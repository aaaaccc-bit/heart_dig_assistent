"""风险分层页面（Risk stratification）。

Layout follows the Pixso screen of the same name: four band cards with a trend
line, a band-to-outcome flow, a factor-contribution panel, a high-risk profile
radar and a list of the highest-risk patients. Every number comes from
:mod:`algorithm.risk_stratification`; this module only paints.
"""

from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from algorithm.clustering import ClusterRun, ClusteringResult
from algorithm.risk_stratification import (
    FactorContribution,
    ProfileAxis,
    RiskCase,
    RiskStratificationResult,
    RiskStratum,
)
from algorithm.stats import format_number
from ui.icons import make_icon_pixmap
from ui.theme import COLORS, DEFAULT_FONT, qcolor

CHEST_PAIN_LABELS = {1: "典型心绞痛", 2: "非典型心绞痛", 3: "非心绞痛", 4: "无症状"}
SEX_LABELS = {1: "男", 0: "女"}

#: Short radar labels: the full names do not fit around a hexagon.
RADAR_LABELS = {
    "sex": "男性",
    "age": "年龄 ≥55",
    "cp": "无症状胸痛",
    "exang": "运动心绞痛",
    "ca": "血管病变",
    "oldpeak": "ST 压低",
}


def _risk_color(key: str) -> str:
    return COLORS.get(f"risk_{key}", COLORS["primary"])


def _font(size: int, bold: bool = False) -> QFont:
    font = QFont(DEFAULT_FONT)
    font.setPixelSize(size)
    font.setBold(bold)
    return font


def _label(
    text: str = "",
    *,
    size: int = 11,
    color: str | None = None,
    bold: bool = False,
) -> QLabel:
    label = QLabel(text)
    label.setFont(_font(size, bold))
    label.setStyleSheet(f"color: {color or COLORS['text']}; background: transparent;")
    return label


def _text(
    painter: QPainter,
    rect: QRectF,
    text: str,
    *,
    size: int = 9,
    color: str | None = None,
    bold: bool = False,
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
) -> None:
    painter.setFont(_font(size, bold))
    painter.setPen(QColor(color or COLORS["text_secondary"]))
    painter.drawText(rect, int(align), text)


def _card(
    title: str,
    subtitle: str = "",
    *,
    extra: QWidget | None = None,
) -> tuple[QFrame, QVBoxLayout]:
    """Rounded surface card with a title, an optional header widget and a body."""

    card = QFrame()
    card.setObjectName("riskCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(10)

    head = QHBoxLayout()
    head.setSpacing(10)
    titles = QVBoxLayout()
    titles.setSpacing(1)
    titles.addWidget(_label(title, size=13, bold=True))
    if subtitle:
        titles.addWidget(_label(subtitle, size=9, color=COLORS["text_secondary"]))
    head.addLayout(titles)
    head.addStretch(1)
    if extra is not None:
        head.addWidget(extra, 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addLayout(head)

    body = QVBoxLayout()
    body.setSpacing(8)
    layout.addLayout(body, 1)
    return card, body


# --------------------------------------------------------------------------- #
# band cards
# --------------------------------------------------------------------------- #
class _AgeSparkline(QWidget):
    """Share of one band across the five age bands, drawn as a soft trend."""

    def __init__(self, points: tuple[float, ...], color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.points = tuple(points)
        self.color = QColor(color)
        self.setFixedSize(88, 34)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if len(self.points) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        inset = 3.0
        width = self.width() - inset * 2
        height = self.height() - inset * 2
        maximum = max(self.points) or 1.0
        step = width / (len(self.points) - 1)
        coordinates = [
            QPointF(inset + index * step, inset + height - value / maximum * height)
            for index, value in enumerate(self.points)
        ]

        line = QPainterPath(coordinates[0])
        for index in range(1, len(coordinates)):
            previous = coordinates[index - 1]
            current = coordinates[index]
            middle = (previous.x() + current.x()) / 2.0
            line.quadTo(middle, previous.y(), middle, (previous.y() + current.y()) / 2.0)
        line.lineTo(coordinates[-1])

        area = QPainterPath(line)
        area.lineTo(coordinates[-1].x(), self.height() - inset)
        area.lineTo(coordinates[0].x(), self.height() - inset)
        area.closeSubpath()

        gradient = QLinearGradient(0, inset, 0, self.height() - inset)
        gradient.setColorAt(0.0, qcolor(self.color.name(), 0.26))
        gradient.setColorAt(1.0, qcolor(self.color.name(), 0.0))
        painter.fillPath(area, gradient)
        painter.setPen(QPen(self.color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(line)
        painter.end()


class _StratumCard(QFrame):
    """One risk band: label, size, share, real positive rate and age trend."""

    def __init__(self, stratum: RiskStratum, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stratum = stratum
        self.color = _risk_color(stratum.key)
        self.setFixedHeight(102)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(22, 14, 16, 12)
        layout.setSpacing(6)

        column = QVBoxLayout()
        column.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(7)
        title_row.addWidget(_label(stratum.label, size=12, color=self.color, bold=True))
        title_row.addWidget(_label(f"{stratum.count} 例", size=10, color=COLORS["text_secondary"]))
        title_row.addStretch(1)
        column.addLayout(title_row)

        column.addWidget(_label(f"{format_number(stratum.percent)}%", size=25, bold=True))
        column.addWidget(
            _label(
                f"实际阳性 {stratum.positive} 例 · {format_number(stratum.positive_rate)}%",
                size=9,
                color=COLORS["text_secondary"],
            )
        )
        column.addStretch(1)
        layout.addLayout(column, 1)
        layout.addWidget(_AgeSparkline(stratum.age_share, self.color), 0, Qt.AlignmentFlag.AlignVCenter)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["surface"]))
        painter.drawRoundedRect(body, 14, 14)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.color))
        painter.drawRoundedRect(QRectF(0.0, 14.0, 5.0, self.height() - 28.0), 2.5, 2.5)
        painter.end()


# --------------------------------------------------------------------------- #
# flow diagram
# --------------------------------------------------------------------------- #
class _FlowChart(QWidget):
    """Predicted band on the left, real diagnosis on the right, ribs in between."""

    def __init__(self, result: RiskStratificationResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.strata = result.strata
        self.flows = {flow.key: flow for flow in result.flows}
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _ribbon(
        painter: QPainter,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        height: float,
        color: QColor,
    ) -> None:
        if height <= 0.0:
            return
        height = max(height, 1.2)
        dx = (x1 - x0) * 0.5
        path = QPainterPath()
        path.moveTo(x0, y0)
        path.cubicTo(x0 + dx, y0, x1 - dx, y1, x1, y1)
        path.lineTo(x1, y1 + height)
        path.cubicTo(x1 - dx, y1 + height, x0 + dx, y0 + height, x0, y0 + height)
        path.closeSubpath()
        painter.fillPath(path, color)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        total = sum(stratum.count for stratum in self.strata) or 1
        positive_total = sum(flow.positive for flow in self.flows.values())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = float(self.width())
        top, bottom = 28.0, float(self.height()) - 8.0
        span = bottom - top
        if span <= 10.0:
            painter.end()
            return
        scale = span / total

        node_width = 12.0
        left_x = 112.0
        right_x = width - 128.0

        _text(painter, QRectF(0, 2, left_x, 16), "预测风险档", size=9)
        _text(
            painter,
            QRectF(right_x + node_width, 2, width - right_x - node_width, 16),
            "实际诊断结局",
            size=9,
            align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        # ribbons first so both columns sit on top of them
        cursor = top
        destinations = []
        for stratum in self.strata:
            flow = self.flows[stratum.key]
            height = stratum.count * scale
            positive_height = flow.positive * scale
            destinations.append((stratum, cursor, cursor + positive_height, height))
            cursor += height

        positive_cursor = top
        negative_cursor = top + positive_total * scale
        for stratum, y, positive_y, height in destinations:
            flow = self.flows[stratum.key]
            color = QColor(_risk_color(stratum.key))
            self._ribbon(
                painter,
                left_x + node_width,
                y,
                right_x,
                positive_cursor,
                flow.positive * scale,
                qcolor(color.name(), 0.40),
            )
            self._ribbon(
                painter,
                left_x + node_width,
                positive_y,
                right_x,
                negative_cursor,
                flow.negative * scale,
                qcolor(color.name(), 0.10),
            )
            positive_cursor += flow.positive * scale
            negative_cursor += flow.negative * scale

        # left column: one rounded strip split by band
        left_path = QPainterPath()
        left_path.addRoundedRect(QRectF(left_x, top, node_width, span), 5, 5)
        painter.save()
        painter.setClipPath(left_path)
        for stratum, y, positive_y, height in destinations:
            flow = self.flows[stratum.key]
            painter.fillRect(QRectF(left_x, y, node_width, height), QColor(_risk_color(stratum.key)))
            painter.fillRect(
                QRectF(left_x, positive_y, node_width, height - flow.positive * scale),
                qcolor(_risk_color(stratum.key), 0.34),
            )
        painter.setPen(QPen(QColor(COLORS["surface"]), 2.0))
        cursor = top
        for stratum in self.strata[:-1]:
            cursor += stratum.count * scale
            painter.drawLine(QPointF(left_x - 1, cursor), QPointF(left_x + node_width + 1, cursor))
        painter.restore()

        # right column: positive above negative
        right_path = QPainterPath()
        right_path.addRoundedRect(QRectF(right_x, top, node_width, span), 5, 5)
        painter.save()
        painter.setClipPath(right_path)
        split = top + positive_total * scale
        painter.fillRect(QRectF(right_x, top, node_width, split - top), QColor(COLORS["danger"]))
        painter.fillRect(QRectF(right_x, split, node_width, bottom - split), QColor(COLORS["success"]))
        painter.setPen(QPen(QColor(COLORS["surface"]), 2.0))
        painter.drawLine(QPointF(right_x - 1, split), QPointF(right_x + node_width + 1, split))
        painter.restore()

        # band labels (outside the strip, so the ribs stay clean)
        for stratum, y, _, height in destinations:
            center = y + height / 2.0
            _text(
                painter,
                QRectF(0, center - 9, left_x - 12, 18),
                f"{stratum.label} · {stratum.count}",
                size=10,
                color=_risk_color(stratum.key),
                bold=True,
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

        # outcome labels
        positive_center = top + positive_total * scale / 2.0
        negative_center = split + (bottom - split) / 2.0
        for center, title, count, color in (
            (positive_center, "实际阳性", positive_total, COLORS["danger"]),
            (negative_center, "实际阴性", total - positive_total, COLORS["success"]),
        ):
            _text(
                painter,
                QRectF(right_x + node_width + 12, center - 16, 112, 15),
                title,
                size=10,
                color=color,
                bold=True,
            )
            _text(
                painter,
                QRectF(right_x + node_width + 12, center - 1, 112, 14),
                f"{count} 例 · {format_number(count / total * 100)}%",
                size=9,
            )
        painter.end()


# --------------------------------------------------------------------------- #
# factor contributions
# --------------------------------------------------------------------------- #
class _ContributionBars(QWidget):
    """Horizontal log-odds shares, longest bar first."""

    def __init__(self, factors: tuple[FactorContribution, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.factors = factors
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = float(self.width())
        height = float(self.height())

        if not self.factors:
            _text(
                painter,
                QRectF(0, 0, width, height),
                "样本量不足以分解风险因素贡献度",
                size=10,
                align=Qt.AlignmentFlag.AlignCenter,
            )
            painter.end()
            return

        label_width = 92.0
        track_left = label_width + 10.0
        track_right = width - 40.0
        rows = len(self.factors)
        row_height = height / rows
        bar_height = min(11.0, row_height * 0.34)
        longest = max(abs(item.share) for item in self.factors) or 1.0

        painter.setPen(QPen(QColor(COLORS["grid"]), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(track_left, 4), QPointF(track_left, height - 4))

        for index, factor in enumerate(self.factors):
            center = row_height * (index + 0.5)
            _text(
                painter,
                QRectF(0, center - 9, label_width, 18),
                factor.label,
                size=10,
                color=COLORS["text"],
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

            length = abs(factor.share) / longest * (track_right - track_left)
            bar = QRectF(track_left, center - bar_height / 2.0, max(length, 2.0), bar_height)
            gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
            gradient.setColorAt(0.0, QColor(COLORS["primary"]))
            gradient.setColorAt(1.0, QColor(COLORS["risk_critical"]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(bar, bar_height / 2.0, bar_height / 2.0)

            _text(
                painter,
                QRectF(track_right + 6, center - 9, 34, 18),
                f"{format_number(factor.share)}%",
                size=9,
                color=COLORS["text"],
                bold=True,
            )
        painter.end()


# --------------------------------------------------------------------------- #
# radar + profile table
# --------------------------------------------------------------------------- #
class _RadarChart(QWidget):
    """Indicator shares of the high-risk bands against the whole sample."""

    def __init__(self, profile: tuple[ProfileAxis, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile = profile
        self.setMinimumSize(260, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _point(self, center: QPointF, radius: float, index: int, value: float) -> QPointF:
        angle = -math.pi / 2.0 + 2.0 * math.pi * index / len(self.profile)
        return QPointF(
            center.x() + math.cos(angle) * radius * value,
            center.y() + math.sin(angle) * radius * value,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if len(self.profile) < 3:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2.0, self.height() / 2.0 - 2.0)
        radius = min(self.width() / 2.0 - 56.0, self.height() / 2.0 - 38.0)
        if radius <= 10.0:
            painter.end()
            return

        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ring in (0.25, 0.5, 0.75, 1.0):
            path = QPainterPath()
            for index in range(len(self.profile)):
                point = self._point(center, radius, index, ring)
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            path.closeSubpath()
            painter.setPen(QPen(QColor(COLORS["grid"]), 1.0))
            painter.drawPath(path)

        painter.setPen(QPen(QColor(COLORS["grid"]), 1.0))
        for index in range(len(self.profile)):
            painter.drawLine(center, self._point(center, radius, index, 1.0))

        for values, color, dashed, width_px, fill in (
            (tuple(axis.overall_value for axis in self.profile), COLORS["text_muted"], True, 1.2, 0.05),
            (tuple(axis.high_value for axis in self.profile), COLORS["risk_high"], False, 2.0, 0.16),
        ):
            path = QPainterPath()
            for index, value in enumerate(values):
                point = self._point(center, radius, index, max(value, 0.02))
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            path.closeSubpath()
            pen = QPen(QColor(color), width_px)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(qcolor(color, fill))
            painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["risk_high"]))
        for index, axis in enumerate(self.profile):
            painter.drawEllipse(self._point(center, radius, index, max(axis.high_value, 0.02)), 2.4, 2.4)

        for index, axis in enumerate(self.profile):
            point = self._point(center, radius + 22.0, index, 1.0)
            if point.x() < center.x() - 6:
                align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            elif point.x() > center.x() + 6:
                align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            else:
                align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            _text(
                painter,
                QRectF(point.x() - 46, point.y() - 9, 92, 18),
                RADAR_LABELS.get(axis.key, axis.label),
                size=9,
                color=COLORS["text_secondary"],
                align=align,
            )
        painter.end()


class _ProfileTable(QWidget):
    """Numbers behind the radar: indicator share per group and the gap."""

    def __init__(self, profile: tuple[ProfileAxis, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        for column, text in enumerate(("指标", "高危人群", "全样本")):
            align = (
                Qt.AlignmentFlag.AlignLeft
                if column == 0
                else Qt.AlignmentFlag.AlignRight
            )
            label = _label(text, size=9, color=COLORS["text_muted"], bold=True)
            label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, 0, column)

        for row, axis in enumerate(profile, start=1):
            name = _label(axis.label, size=9, color=COLORS["text_secondary"])
            name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            high = _label(f"{format_number(axis.high_rate)}%", size=9, color=COLORS["risk_high"], bold=True)
            high.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            overall = _label(f"{format_number(axis.overall_rate)}%", size=9, color=COLORS["text"])
            overall.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            grid.addWidget(name, row, 0)
            grid.addWidget(high, row, 1)
            grid.addWidget(overall, row, 2)

        grid.setColumnStretch(0, 1)
        grid.setRowStretch(len(profile) + 1, 1)


# --------------------------------------------------------------------------- #
# case table
# --------------------------------------------------------------------------- #
class _CaseTable(QWidget):
    """Highest-risk patients of the critical band."""

    COLUMNS = (
        ("#", 26, Qt.AlignmentFlag.AlignLeft),
        ("年龄", 44, Qt.AlignmentFlag.AlignCenter),
        ("性别", 40, Qt.AlignmentFlag.AlignCenter),
        ("胸痛类型", 100, Qt.AlignmentFlag.AlignCenter),
        ("最大心率", 74, Qt.AlignmentFlag.AlignCenter),
        ("ST 压低", 62, Qt.AlignmentFlag.AlignCenter),
        ("预测风险", 68, Qt.AlignmentFlag.AlignRight),
    )

    def __init__(self, cases: tuple[RiskCase, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("riskTableHead")
        header.setFixedHeight(26)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        header_layout.setSpacing(6)
        for text, width, align in self.COLUMNS:
            label = _label(text, size=9, color=COLORS["text_secondary"], bold=True)
            label.setFixedWidth(width)
            label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            header_layout.addWidget(label)
        header_layout.addStretch(1)
        layout.addWidget(header)

        if not cases:
            empty = _label("当前配置下没有极高风险样本", size=10, color=COLORS["text_secondary"])
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setMinimumHeight(60)
            layout.addWidget(empty)
            layout.addStretch(1)
            return

        for case in cases:
            row = QFrame()
            row.setObjectName("riskTableRow")
            row.setFixedHeight(26)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 0, 10, 0)
            row_layout.setSpacing(6)
            probability = f"{case.probability * 100:.1f}%"
            cells = (
                (str(case.rank), COLORS["text_muted"], False),
                (str(case.age), COLORS["text"], False),
                (SEX_LABELS.get(case.sex, str(case.sex)), COLORS["text"], False),
                (CHEST_PAIN_LABELS.get(case.chest_pain, str(case.chest_pain)), COLORS["text"], False),
                (str(case.thalach), COLORS["text"], False),
                (format_number(case.oldpeak), COLORS["text"], False),
                (probability, COLORS["risk_critical"], True),
            )
            for (text, color, bold), (_, width, align) in zip(cells, self.COLUMNS):
                label = _label(text, size=9, color=color, bold=bold)
                label.setFixedWidth(width)
                label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                row_layout.addWidget(label)
            row_layout.addStretch(1)
            layout.addWidget(row)
        layout.addStretch(1)


# --------------------------------------------------------------------------- #
# clustering view
# --------------------------------------------------------------------------- #
#: Cluster colours follow the risk ramp: the riskiest cluster is always red.
_CLUSTER_RAMP = {
    2: ("risk_critical", "risk_low"),
    3: ("risk_critical", "risk_medium", "risk_low"),
    4: ("risk_critical", "risk_high", "risk_medium", "risk_low"),
    5: ("risk_critical", "risk_high", "risk_medium", "risk_low", "teal"),
    6: ("risk_critical", "risk_high", "risk_medium", "risk_low", "teal", "purple"),
}

CLUSTER_DECIMALS = {"age": 0, "thalach": 0, "oldpeak": 1, "ca": 1, "trestbps": 0}

#: Short labels for the method switch.
_METHOD_SHORT = {"kmeans": "KMeans", "single": "最小距离（单链）"}


def _cluster_color(k: int, index: int) -> str:
    ramp = _CLUSTER_RAMP.get(k, _CLUSTER_RAMP[6])
    return COLORS.get(ramp[min(index, len(ramp) - 1)], COLORS["primary"])


class _ModeSwitch(QFrame):
    """Two exclusive pills switching between the model and clustering views."""

    changed = Signal(int)

    def __init__(self, labels: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for index, text in enumerate(labels):
            button = QPushButton(text)
            button.setObjectName("riskMode")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(32)
            button.setChecked(index == 0)
            self.group.addButton(button, index)
            row.addWidget(button)
        self.group.idClicked.connect(self.changed.emit)


class _KSelector(QWidget):
    """``k=2 … k=6`` shortcut buttons for the clustering view."""

    changed = Signal(int)

    def __init__(self, candidates: tuple[int, ...], current: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for k in candidates:
            button = QPushButton(f"k={k}")
            button.setObjectName("riskKButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFixedHeight(24)
            button.setMinimumWidth(40)
            button.setChecked(k == current)
            self.group.addButton(button, k)
            row.addWidget(button)
        self.group.idClicked.connect(self.changed.emit)


class _KScanChart(QWidget):
    """Silhouette and ARI for every candidate ``k``."""

    def __init__(self, result: ClusteringResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        runs = self.result.runs
        if len(runs) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, right, top, bottom = 38.0, 12.0, 30.0, 24.0
        plot = QRectF(left, top, self.width() - left - right, self.height() - top - bottom)
        if plot.width() <= 10.0 or plot.height() <= 10.0:
            painter.end()
            return

        values = [run.metric.silhouette for run in runs] + [run.metric.ari for run in runs]
        axis_max = max(0.3, math.ceil(max(values) * 10.0) / 10.0)

        def y_at(value: float) -> float:
            return plot.bottom() - value / axis_max * plot.height()

        def x_at(index: int) -> float:
            return plot.left() + plot.width() * index / (len(runs) - 1)

        # selected-k band
        selected_index = next((index for index, run in enumerate(runs) if run.k == self.result.selected_k), 0)
        band = plot.width() / max(len(runs) - 1, 1) * 0.72
        center = x_at(selected_index)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(COLORS["primary"], 0.07))
        painter.drawRoundedRect(QRectF(center - band / 2, plot.top(), band, plot.height()), 6, 6)

        painter.setFont(_font(8))
        painter.setPen(QColor(COLORS["text_muted"]))
        for fraction in (0.0, 0.5, 1.0):
            value = axis_max * fraction
            y = y_at(value)
            painter.setPen(QPen(QColor(COLORS["grid"]), 1.0))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            _text(
                painter,
                QRectF(0, y - 7, left - 6, 14),
                f"{value:.2f}",
                size=8,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

        for series, color, dashed in (
            ([run.metric.silhouette for run in runs], COLORS["primary"], False),
            ([run.metric.ari for run in runs], COLORS["risk_high"], True),
        ):
            path = QPainterPath()
            for index, value in enumerate(series):
                point = QPointF(x_at(index), y_at(value))
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            pen = QPen(QColor(color), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            for index, value in enumerate(series):
                radius = 4.0 if index == selected_index else 2.8
                painter.drawEllipse(QPointF(x_at(index), y_at(value)), radius, radius)

        for index, run in enumerate(runs):
            _text(
                painter,
                QRectF(x_at(index) - 22, plot.bottom() + 6, 44, 14),
                f"k={run.k}",
                size=8,
                color=COLORS["text_secondary"],
                bold=run.k == self.result.selected_k,
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )

        # legend
        painter.setPen(QPen(QColor(COLORS["primary"]), 2.0))
        painter.drawLine(QPointF(left, 12), QPointF(left + 12, 12))
        _text(painter, QRectF(left + 16, 5, 66, 14), "轮廓系数", size=8, color=COLORS["text_secondary"])
        pen = QPen(QColor(COLORS["risk_high"]), 2.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(left + 92, 12), QPointF(left + 104, 12))
        _text(painter, QRectF(left + 108, 5, 96, 14), "ARI（与阳性）", size=8, color=COLORS["text_secondary"])
        painter.end()


class _ClusterBars(QWidget):
    """Positive rate per cluster, with the raw counts on the right."""

    def __init__(self, run: ClusterRun, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.run = run
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        groups = self.run.groups
        if not groups:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        k = self.run.k
        row_height = min(74.0, max(30.0, (self.height() - 4.0) / k))
        top = max(2.0, (self.height() - row_height * k) / 2.0)
        label_width = 78.0
        track = QRectF(label_width, 0, self.width() - label_width - 96.0, 9.0)

        for index, group in enumerate(groups):
            center = top + row_height * (index + 0.5)
            color = _cluster_color(k, index)
            _text(
                painter,
                QRectF(0, center - 9, label_width - 8, 18),
                f"{group.label} · {group.count}",
                size=9,
                color=color,
                bold=True,
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

            bar = QRectF(track.left(), center - track.height() / 2, track.width(), track.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["grid"]))
            painter.drawRoundedRect(bar, 4.5, 4.5)
            filled = QRectF(bar)
            filled.setWidth(max(3.0, bar.width() * group.positive_rate / 100.0))
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(filled, 4.5, 4.5)

            _text(
                painter,
                QRectF(bar.right() + 8, center - 16, 88, 14),
                f"{format_number(group.positive_rate)}%",
                size=9,
                color=color,
                bold=True,
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            _text(
                painter,
                QRectF(bar.right() + 8, center, 88, 14),
                f"{group.positive} / {group.count} 例",
                size=8,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
        painter.end()


class _ScatterPlot(QWidget):
    """Two-dimensional PCA projection, coloured by cluster."""

    def __init__(self, result: ClusteringResult, run: ClusterRun, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.run = run
        self.setMinimumSize(280, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        points = self.result.points
        if not points:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = QRectF(34.0, 16.0, self.width() - 46.0, self.height() - 48.0)
        if plot.width() <= 10.0 or plot.height() <= 10.0:
            painter.end()
            return

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = (x_max - x_min) * 0.06 or 1.0
        y_pad = (y_max - y_min) * 0.06 or 1.0
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

        def point_at(x: float, y: float) -> QPointF:
            px = plot.left() + (x - x_min) / (x_max - x_min) * plot.width()
            py = plot.bottom() - (y - y_min) / (y_max - y_min) * plot.height()
            return QPointF(px, py)

        painter.setPen(QPen(QColor(COLORS["grid"]), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())

        for (x, y), cluster in zip(points, self.run.labels):
            color = _cluster_color(self.run.k, cluster)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(color, 0.78))
            painter.drawEllipse(point_at(x, y), 2.8, 2.8)

        _text(
            painter,
            QRectF(plot.left(), plot.bottom() + 6, plot.width(), 14),
            f"PC1（解释度 {format_number(self.result.explained[0] * 100)}%）",
            size=8,
            align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
        )
        _text(
            painter,
            QRectF(plot.left() - 30, 2, 120, 14),
            f"PC2（{format_number(self.result.explained[1] * 100)}%）",
            size=8,
        )
        painter.end()


class _ClusterTable(QWidget):
    """Clinical means per cluster."""

    def __init__(self, run: ClusterRun, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("riskTableHead")
        header.setFixedHeight(24)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(4)
        for text, width, align in self._columns():
            label = _label(text, size=8, color=COLORS["text_secondary"], bold=True)
            label.setFixedWidth(width)
            label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            header_layout.addWidget(label)
        header_layout.addStretch(1)
        layout.addWidget(header)

        for index, group in enumerate(run.groups):
            row = QFrame()
            row.setObjectName("riskTableRow")
            row.setFixedHeight(24)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 0, 8, 0)
            row_layout.setSpacing(4)
            cells = [
                (group.label, _cluster_color(run.k, index), True),
                (str(group.count), COLORS["text"], False),
                (f"{format_number(group.positive_rate)}%", _cluster_color(run.k, index), True),
            ]
            for feature in group.features:
                decimals = CLUSTER_DECIMALS.get(feature.key, 1)
                cells.append((format_number(feature.mean, decimals), COLORS["text"], False))
            for (text, color, bold), (_, width, align) in zip(cells, self._columns()):
                label = _label(text, size=8, color=color, bold=bold)
                label.setFixedWidth(width)
                label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                row_layout.addWidget(label)
            row_layout.addStretch(1)
            layout.addWidget(row)
        layout.addStretch(1)

    @staticmethod
    def _columns() -> tuple[tuple[str, int, Qt.AlignmentFlag], ...]:
        return (
            ("簇", 48, Qt.AlignmentFlag.AlignLeft),
            ("例数", 40, Qt.AlignmentFlag.AlignCenter),
            ("阳性率", 52, Qt.AlignmentFlag.AlignCenter),
            ("年龄", 44, Qt.AlignmentFlag.AlignCenter),
            ("最大心率", 56, Qt.AlignmentFlag.AlignCenter),
            ("ST 压低", 50, Qt.AlignmentFlag.AlignCenter),
            ("血管数", 46, Qt.AlignmentFlag.AlignCenter),
            ("静息血压", 52, Qt.AlignmentFlag.AlignCenter),
        )


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
class RiskStratificationPage(QWidget):
    """The 风险分层 screen: model bands and unsupervised clustering."""

    report_requested = Signal()

    def __init__(
        self,
        result: RiskStratificationResult,
        clusters: ClusteringResult | Sequence[ClusteringResult] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        if clusters is None:
            self.clusters: tuple[ClusteringResult, ...] = ()
        elif isinstance(clusters, ClusteringResult):
            self.clusters = (clusters,)
        else:
            self.clusters = tuple(clusters)
        self._cluster_method = self.clusters[0].method_key if self.clusters else "kmeans"
        self._cluster_k = self.clusters[0].selected_k if self.clusters else 2
        self.setMinimumSize(1120, 830)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._model_view(result))
        if self.clusters:
            self.stack.addWidget(self._cluster_view())
        root.addWidget(self._header(result))
        root.addWidget(self.stack, 1)

    @property
    def active_clusters(self) -> ClusteringResult | None:
        return next(
            (item for item in self.clusters if item.method_key == self._cluster_method),
            None,
        )

    # -- sections ---------------------------------------------------------
    def _header(self, result: RiskStratificationResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(38)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        self._title_label = _label("风险分档总览", size=15, bold=True)
        self._subtitle_label = _label(
            self._model_subtitle(result),
            size=9,
            color=COLORS["text_secondary"],
        )
        titles.addWidget(self._title_label)
        titles.addWidget(self._subtitle_label)
        row.addLayout(titles)
        row.addStretch(1)

        if self.clusters:
            switch = _ModeSwitch(("模型分档", "无监督分群"))
            switch.changed.connect(self._set_mode)
            self._mode_switch = switch
            row.addWidget(switch, 0, Qt.AlignmentFlag.AlignVCenter)
        else:
            self._mode_switch = None

        button = QPushButton("生成风险报告")
        button.setObjectName("riskPrimary")
        button.setFixedHeight(38)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setIcon(make_icon_pixmap("file-text", 15, COLORS["surface"]))
        button.setIconSize(QSize(15, 15))
        button.clicked.connect(self.report_requested.emit)
        row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
        return holder

    @staticmethod
    def _model_subtitle(result: RiskStratificationResult) -> str:
        return (
            f"{result.model} · {result.folds} 折交叉验证 · "
            f"AUC {result.roc_auc:.3f} · Brier {result.brier_score:.3f}"
        )

    def _set_mode(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if self._mode_switch is not None:
            button = self._mode_switch.group.button(index)
            if button is not None and not button.isChecked():
                button.setChecked(True)
        clusters = self.active_clusters
        if index == 1 and clusters is not None:
            run = next(
                (item for item in clusters.runs if item.k == self._cluster_k),
                clusters.runs[0],
            )
            self._title_label.setText("无监督分群总览")
            self._subtitle_label.setText(
                f"{clusters.method} · 扫描 k=2–6 · 当前 k={run.k} · "
                f"轮廓系数 {run.metric.silhouette:.3f} · ARI {run.metric.ari:.3f}"
            )
        else:
            self._title_label.setText("风险分档总览")
            self._subtitle_label.setText(self._model_subtitle(self.result))

    def _model_view(self, result: RiskStratificationResult) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._bands(result))
        layout.addWidget(self._middle(result))
        layout.addWidget(self._bottom(result))
        layout.addStretch(1)
        return view

    def _method_strip(self, active: ClusteringResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(30)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        switch = _ModeSwitch(
            tuple(_METHOD_SHORT.get(item.method_key, item.method_key) for item in self.clusters)
        )
        index = next(
            (
                position
                for position, item in enumerate(self.clusters)
                if item.method_key == active.method_key
            ),
            0,
        )
        button = switch.group.button(index)
        if button is not None:
            button.setChecked(True)
        switch.changed.connect(
            lambda position: self._select_method(self.clusters[position].method_key)
        )
        row.addWidget(switch, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        row.addWidget(
            _label(
                "单链法易产生链式簇；簇规模极不均衡时应以 KMeans 结果为准。",
                size=9,
                color=COLORS["text_muted"],
            )
        )
        return holder

    def _cluster_view(self) -> QWidget:
        clusters = self.active_clusters
        if clusters is None:
            return QWidget()
        run = next(
            (item for item in clusters.runs if item.k == self._cluster_k),
            clusters.runs[0],
        )
        best = max(clusters.runs, key=lambda item: item.metric.silhouette)
        sizes = [group.count for group in run.groups]
        size_text = f"{min(sizes)}–{max(sizes)} 例" if sizes else "—"
        degenerate = bool(sizes) and min(sizes) <= 1

        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        if len(self.clusters) > 1:
            layout.addWidget(self._method_strip(clusters))

        top = QWidget()
        top.setFixedHeight(300)
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(12)

        selector = _KSelector(tuple(item.k for item in clusters.runs), run.k)
        selector.changed.connect(self._select_k)
        scan_card, scan_body = _card(
            "聚类质量（k 扫描）",
            "轮廓系数越高越好；ARI 衡量分群与阳性标签的一致程度",
            extra=selector,
        )
        scan_card.setMinimumWidth(440)
        scan_body.addWidget(_KScanChart(clusters), 1)
        if degenerate:
            warning = _label(
                f"当前分群为 {size_text}：最小距离法把绝大多数样本连成一条链，"
                "单点簇会虚高轮廓系数，不能据此判断分群质量。",
                size=9,
                color=COLORS["danger"],
            )
        else:
            warning = _label(
                f"k={best.k} 时轮廓系数最高（{format_number(best.metric.silhouette, 3)}），"
                "k 继续增大后指标持续下降，说明数据只支持粗分。",
                size=9,
                color=COLORS["text_secondary"],
            )
        warning.setWordWrap(True)
        scan_body.addWidget(warning)
        top_row.addWidget(scan_card, 5)

        bar_card, bar_body = _card(
            "各簇阳性率",
            f"k={run.k} 时各簇的实际阳性比例；簇规模 {size_text}",
        )
        bar_card.setMinimumWidth(360)
        bar_body.addWidget(_ClusterBars(run), 1)
        gap = run.groups[0].positive_rate - run.groups[-1].positive_rate if run.groups else 0.0
        bar_body.addWidget(
            _label(
                f"首末簇阳性率相差 {format_number(gap)} 个百分点；"
                + (
                    "分群由单点簇主导，差异没有统计意义。"
                    if degenerate
                    else "轮廓系数偏低，簇间仍有重叠，应按风险分群解读。"
                ),
                size=9,
                color=COLORS["text_secondary"],
            )
        )
        top_row.addWidget(bar_card, 4)
        layout.addWidget(top)

        bottom = QWidget()
        bottom.setFixedHeight(380)
        bottom_row = QHBoxLayout(bottom)
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(12)

        scatter_card, scatter_body = _card(
            "PCA 投影",
            f"PC1 {format_number(clusters.explained[0] * 100)}% · "
            f"PC2 {format_number(clusters.explained[1] * 100)}%，一个点 = 一例样本",
        )
        scatter_card.setMinimumWidth(520)
        scatter_body.addWidget(_ScatterPlot(clusters, run), 1)
        legend = QHBoxLayout()
        legend.setSpacing(12)
        for index, group in enumerate(run.groups):
            legend.addWidget(
                _label(
                    f'<span style="color:{_cluster_color(run.k, index)};">●</span> {group.label} · {group.count} 例',
                    size=8,
                )
            )
        legend.addStretch(1)
        scatter_body.addLayout(legend)
        bottom_row.addWidget(scatter_card, 5)

        table_card, table_body = _card("簇画像", "各簇临床指标均值")
        table_card.setMinimumWidth(440)
        table_body.addWidget(_ClusterTable(run), 0)
        note = _label(self._cluster_delta_note(run, degenerate), size=9, color=COLORS["text_secondary"])
        note.setWordWrap(True)
        table_body.addWidget(note)
        table_body.addWidget(
            _label("簇按阳性率由高到低编号；数值为簇内均值，缺失值不参与。", size=9, color=COLORS["text_muted"])
        )
        table_body.addStretch(1)
        bottom_row.addWidget(table_card, 4)
        layout.addWidget(bottom)
        layout.addStretch(1)
        return view

    @staticmethod
    def _cluster_delta_note(run: ClusterRun, degenerate: bool = False) -> str:
        """Compare the riskiest cluster with the lowest-risk one."""

        if len(run.groups) < 2:
            return ""
        first, last = run.groups[0], run.groups[-1]
        if degenerate:
            return (
                f"簇 1 只有 {first.count} 例、簇 {run.k} 只有 {last.count} 例，"
                "单点簇的均值没有代表性；这组结果说明单链法在当前数据上不适用。"
            )

        def mean(group, key: str) -> float:
            return next(feature.mean for feature in group.features if feature.key == key)

        return (
            f"簇 1 相对簇 {run.k}：年龄 {mean(first, 'age') - mean(last, 'age'):+.0f} 岁、"
            f"最大心率 {mean(first, 'thalach') - mean(last, 'thalach'):+.0f} bpm、"
            f"ST 压低 {mean(first, 'oldpeak') - mean(last, 'oldpeak'):+.1f} mm；"
            f"轮廓系数仅 {run.metric.silhouette:.3f}，簇间仍有重叠。"
        )

    def _rebuild_cluster_view(self) -> None:
        view = self._cluster_view()
        old = self.stack.widget(1)
        self.stack.insertWidget(1, view)
        self.stack.removeWidget(old)
        old.deleteLater()

    def _select_k(self, k: int) -> None:
        if not self.clusters or k == self._cluster_k:
            return
        self._cluster_k = k
        self._rebuild_cluster_view()
        self._set_mode(1)

    def _select_method(self, key: str) -> None:
        if key == self._cluster_method:
            return
        self._cluster_method = key
        active = self.active_clusters
        if active is not None and self._cluster_k not in {run.k for run in active.runs}:
            self._cluster_k = active.selected_k
        self._rebuild_cluster_view()
        self._set_mode(1)

    @staticmethod
    def _bands(result: RiskStratificationResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(102)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        for stratum in result.strata:
            row.addWidget(_StratumCard(stratum), 1)
        return holder

    def _middle(self, result: RiskStratificationResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(288)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        flow_card, flow_body = _card(
            "风险分层流转",
            "预测风险档 → 实际诊断结局；带宽为样本数，浅色为未患病",
        )
        flow_card.setMinimumWidth(420)
        flow_body.addWidget(_FlowChart(result), 1)
        lowest = result.strata[0]
        flow_body.addWidget(
            _label(
                f"低风险档仍有 {lowest.positive} 例实际阳性（{format_number(lowest.positive_rate)}%），"
                "筛查场景需保留复核环节。",
                size=9,
                color=COLORS["text_secondary"],
            )
        )
        row.addWidget(flow_card, 3)

        factor_card, factor_body = _card(
            "关键风险因素贡献度",
            "高危及极高风险档相对低风险档的 log-odds 分解",
        )
        factor_card.setMinimumWidth(400)
        factor_body.addWidget(_ContributionBars(result.factors), 1)
        factor_body.addWidget(
            _label("数值为各特征占全部绝对贡献的比例；仅说明模型关联方向。", size=9, color=COLORS["text_muted"])
        )
        row.addWidget(factor_card, 2)
        return holder

    def _bottom(self, result: RiskStratificationResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(330)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        high_total = sum(stratum.count for stratum in result.strata if stratum.key in {"high", "critical"})
        profile_card, profile_body = _card(
            "高危人群特征画像",
            f"高风险 + 极高风险档合计 {high_total} 例，指标为各特征在组内的占比",
        )
        profile_card.setMinimumWidth(600)

        content = QHBoxLayout()
        content.setSpacing(10)
        content.addWidget(_RadarChart(result.profile), 3)

        side = QVBoxLayout()
        side.setSpacing(6)
        side.addWidget(
            _label(
                f'<span style="color:{COLORS["risk_high"]};">●</span> 高危人群 {high_total} 例'
                f'　<span style="color:{COLORS["text_muted"]};">◇ 全样本 {result.n} 例</span>',
                size=9,
            )
        )
        side.addWidget(_ProfileTable(result.profile), 1)
        content.addLayout(side, 2)
        profile_body.addLayout(content, 1)
        profile_body.addWidget(self._profile_note(result))
        row.addWidget(profile_card, 5)

        case_card, case_body = _card(
            f"极高风险 {result.critical_count} 例明细",
            f"按融合概率降序，展示前 {len(result.cases)} 例",
        )
        case_card.setMinimumWidth(470)
        case_body.addWidget(_CaseTable(result.cases), 1)
        case_body.addWidget(
            _label(
                "预测风险为 5 折交叉验证的折外概率，不用于个体诊断。",
                size=9,
                color=COLORS["text_muted"],
            )
        )
        row.addWidget(case_card, 4)
        return holder

    @staticmethod
    def _profile_note(result: RiskStratificationResult) -> QLabel:
        """One sentence naming the two indicators with the widest group gap."""

        ranked = sorted(
            result.profile,
            key=lambda axis: axis.high_rate - axis.overall_rate,
            reverse=True,
        )[:2]
        if len(ranked) < 2:
            label = _label("", size=9, color=COLORS["text_secondary"])
            label.setWordWrap(True)
            return label
        first, second = ranked
        text = (
            f"与全样本相比，高危人群在「{first.label}」"
            f"（{format_number(first.high_rate)}% vs {format_number(first.overall_rate)}%）"
            f"和「{second.label}」"
            f"（{format_number(second.high_rate)}% vs {format_number(second.overall_rate)}%）"
            "上差异最大。"
        )
        label = _label(text, size=9, color=COLORS["text_secondary"])
        label.setWordWrap(True)
        return label


def risk_stylesheet() -> str:
    return f"""
QFrame#riskCard {{ background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; border-radius: 16px; }}
QFrame#riskTableHead {{ background: {COLORS["surface_alt"]}; border: none; border-radius: 8px; }}
QFrame#riskTableRow {{ background: transparent; border: none; border-bottom: 1px solid {COLORS["grid"]}; }}
QPushButton#riskPrimary {{
    background: {COLORS["primary"]};
    border: none;
    border-radius: 10px;
    color: {COLORS["surface"]};
    font-size: 11px;
    font-weight: 700;
    padding: 0 18px;
}}
QPushButton#riskPrimary:hover {{ background: {COLORS["primary_hover"]}; }}
QPushButton#riskPrimary:pressed {{ background: {COLORS["primary_pressed"]}; }}
QPushButton#riskMode {{
    background: {COLORS["surface_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 9px;
    color: {COLORS["text_secondary"]};
    font-size: 10px;
    padding: 0 14px;
}}
QPushButton#riskMode:checked {{
    background: {COLORS["primary"]};
    border-color: {COLORS["primary"]};
    color: {COLORS["surface"]};
    font-weight: 700;
}}
QPushButton#riskMode:hover:!checked {{ border-color: {COLORS["primary"]}; color: {COLORS["primary"]}; }}
QPushButton#riskKButton {{
    background: {COLORS["surface_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    color: {COLORS["text_secondary"]};
    font-size: 9px;
}}
QPushButton#riskKButton:checked {{
    background: {COLORS["primary"]};
    border-color: {COLORS["primary"]};
    color: {COLORS["surface"]};
    font-weight: 700;
}}
QPushButton#riskKButton:hover:!checked {{ border-color: {COLORS["primary"]}; color: {COLORS["primary"]}; }}
"""
