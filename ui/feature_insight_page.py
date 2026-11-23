"""特征洞察页: permutation importance, single-feature discrimination,
group differences and the correlation structure of the continuous columns.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from algorithm.feature_insight import FeatureInsightResult
from algorithm.stats import format_number
from ui.icons import make_icon_pixmap
from ui.theme import COLORS, DEFAULT_FONT, qcolor


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


def _card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("insightCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(8)

    head = QVBoxLayout()
    head.setSpacing(1)
    head.addWidget(_label(title, size=13, bold=True))
    if subtitle:
        head.addWidget(_label(subtitle, size=9, color=COLORS["text_secondary"]))
    layout.addLayout(head)

    body = QVBoxLayout()
    body.setSpacing(6)
    layout.addLayout(body, 1)
    return card, body


# --------------------------------------------------------------------------- #
# ranked bars
# --------------------------------------------------------------------------- #
class _RankedBars(QWidget):
    """One horizontal bar per feature, sorted by the value shown."""

    def __init__(
        self,
        items: tuple[tuple[str, float], ...],
        color: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        baseline: float | None = None,
        decimals: int = 3,
        signed: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.items = items
        self.color = color
        self.minimum = minimum
        self.maximum = maximum
        self.baseline = baseline
        self.decimals = decimals
        self.signed = signed
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        if not self.items:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = float(self.width())
        height = float(self.height())
        label_width = 82.0
        value_width = 44.0
        track_left = label_width + 8.0
        track_right = width - value_width - 8.0
        if track_right - track_left < 20.0:
            painter.end()
            return

        rows = len(self.items)
        row_height = height / rows
        bar_height = min(9.0, row_height * 0.52)

        values = [value for _, value in self.items]
        if self.minimum is not None and self.maximum is not None:
            low, high = self.minimum, self.maximum
        else:
            low = min(0.0, min(values))
            high = max(max(values), 0.0)
        if high <= low:
            high = low + 1.0

        def x_at(value: float) -> float:
            clipped = min(max(value, low), high)
            return track_left + (clipped - low) / (high - low) * (track_right - track_left)

        if self.baseline is not None:
            x = x_at(self.baseline)
            painter.setPen(QPen(qcolor(COLORS["text_muted"], 0.55), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, 0), QPointF(x, height))

        zero = x_at(0.0)
        for index, (label_text, value) in enumerate(self.items):
            center = row_height * (index + 0.5)
            _text(
                painter,
                QRectF(0, center - 8, label_width, 16),
                label_text,
                size=8,
                color=COLORS["text_secondary"],
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

            tip = x_at(value)
            left = min(zero, tip)
            length = abs(tip - zero)
            bar = QRectF(
                left,
                center - bar_height / 2.0,
                max(length, 1.5),
                bar_height,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(self.color, 0.85))
            painter.drawRoundedRect(bar, bar_height / 2.0, bar_height / 2.0)

            text = (
                f"{value:+.{self.decimals}f}"
                if self.signed
                else f"{value:.{self.decimals}f}"
            )
            _text(
                painter,
                QRectF(track_right + 6, center - 8, value_width, 16),
                text,
                size=8,
                color=COLORS["text"],
                bold=index == 0,
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
        painter.end()


# --------------------------------------------------------------------------- #
# correlation heat map
# --------------------------------------------------------------------------- #
class _CorrelationHeatmap(QWidget):
    """Spearman correlation of the continuous columns, annotated."""

    def __init__(self, result: FeatureInsightResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.matrix = result.correlation
        self.setMinimumSize(260, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        keys = self.matrix.keys
        if len(keys) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = float(self.width())
        height = float(self.height())
        label_gap = 62.0
        top_gap = 26.0
        legend_height = 20.0
        cell = min(
            (width - label_gap - 12.0) / len(keys),
            (height - top_gap - legend_height - 12.0) / len(keys),
            46.0,
        )
        left = label_gap + max(
            0.0, (width - label_gap - 12.0 - cell * len(keys)) / 2.0
        )
        top = top_gap

        for index, label in enumerate(self.matrix.labels):
            _text(
                painter,
                QRectF(left + cell * index, 0, cell, top_gap - 4.0),
                label,
                size=8,
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            )
            _text(
                painter,
                QRectF(0, top + cell * index, label_gap - 8.0, cell),
                label,
                size=8,
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

        for row, key in enumerate(keys):
            for column in range(len(keys)):
                value = self.matrix.values[row][column]
                rect = QRectF(left + cell * column + 1, top + cell * row + 1, cell - 2, cell - 2)
                base = COLORS["danger"] if value >= 0 else COLORS["primary"]
                alpha = 0.06 + min(abs(value), 1.0) * 0.52
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(qcolor(base, alpha))
                painter.drawRoundedRect(rect, 5, 5)
                _text(
                    painter,
                    rect,
                    f"{value:.2f}",
                    size=8,
                    color=COLORS["text"],
                    bold=abs(value) >= 0.4,
                    align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                )

        legend_y = top + cell * len(keys) + 6.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(COLORS["primary"], 0.5))
        painter.drawRoundedRect(QRectF(left, legend_y, 12, 10), 3, 3)
        _text(painter, QRectF(left + 16, legend_y - 3, 40, 16), "负相关", size=8)
        painter.setBrush(qcolor(COLORS["danger"], 0.5))
        painter.drawRoundedRect(QRectF(left + 62, legend_y, 12, 10), 3, 3)
        _text(painter, QRectF(left + 78, legend_y - 3, 40, 16), "正相关", size=8)
        _text(
            painter,
            QRectF(left + 130, legend_y - 3, width - left - 130, 16),
            "|r| ≥ 0.40 加粗",
            size=8,
            color=COLORS["text_muted"],
        )
        painter.end()


# --------------------------------------------------------------------------- #
# group comparison table
# --------------------------------------------------------------------------- #
class _GroupTable(QWidget):
    """Means of the continuous features in the positive and negative groups."""

    def __init__(self, result: FeatureInsightResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.signals = tuple(
            feature for feature in result.features if feature.kind == "numeric"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("insightTableHead")
        header.setFixedHeight(26)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(6)
        for text, stretch, align in self._columns():
            label = _label(text, size=9, color=COLORS["text_secondary"], bold=True)
            label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            header_layout.addWidget(label, stretch)
        layout.addWidget(header)

        for feature in self.signals:
            row = QFrame()
            row.setObjectName("insightTableRow")
            row.setFixedHeight(30)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 0, 8, 0)
            row_layout.setSpacing(6)
            cells = (
                (feature.label, COLORS["text"], True),
                (format_number(feature.positive_mean, 1), COLORS["text"], False),
                (format_number(feature.negative_mean, 1), COLORS["text"], False),
                (
                    f"{feature.group_delta:+.1f}",
                    COLORS["danger"] if abs(feature.effect_size) >= 0.8 else COLORS["text_secondary"],
                    abs(feature.effect_size) >= 0.8,
                ),
                (
                    f"{feature.effect_size:+.2f}",
                    COLORS["danger"] if abs(feature.effect_size) >= 0.8 else COLORS["text_secondary"],
                    abs(feature.effect_size) >= 0.8,
                ),
            )
            for (text, color, bold), (_, stretch, align) in zip(cells, self._columns()):
                label = _label(text, size=9, color=color, bold=bold)
                label.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                row_layout.addWidget(label, stretch)
            layout.addWidget(row)
        layout.addStretch(1)

    @staticmethod
    def _columns() -> tuple[tuple[str, int, Qt.AlignmentFlag], ...]:
        return (
            ("指标", 2, Qt.AlignmentFlag.AlignLeft),
            ("阳性组", 1, Qt.AlignmentFlag.AlignCenter),
            ("阴性组", 1, Qt.AlignmentFlag.AlignCenter),
            ("差值", 1, Qt.AlignmentFlag.AlignCenter),
            ("Cohen's d", 1, Qt.AlignmentFlag.AlignCenter),
        )


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
class FeatureInsightPage(QWidget):
    """特征洞察: what the model relies on, and what each feature says alone."""

    def __init__(self, result: FeatureInsightResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.setMinimumSize(1120, 830)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)
        root.addWidget(self._header(result))
        root.addWidget(self._top_rows(result))
        root.addWidget(self._lower_row(result))
        root.addWidget(self._note(result))
        root.addStretch(1)

    def _header(self, result: FeatureInsightResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(38)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.addWidget(_label("特征洞察总览", size=15, bold=True))
        titles.addWidget(
            _label(
                f"{len(result.features)} 个特征 · {result.n} 例（阳性 {result.positive}） · "
                f"{result.folds} 折交叉验证 · 置换重要性 + 单变量 AUC",
                size=9,
                color=COLORS["text_secondary"],
            )
        )
        row.addLayout(titles)
        row.addStretch(1)
        return holder

    @staticmethod
    def _top_rows(result: FeatureInsightResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(330)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        importance_card, importance_body = _card(
            "置换重要性",
            "打乱该特征后验证折 AUC 的平均下降；越接近 0 说明模型越不依赖它",
        )
        importance_card.setMinimumWidth(420)
        importance_items = tuple(
            (feature.label, feature.permutation) for feature in result.features
        )
        importance_body.addWidget(
            _RankedBars(importance_items, COLORS["primary"], signed=True),
            1,
        )
        row.addWidget(importance_card, 1)

        discrimination_card, discrimination_body = _card(
            "单变量判别力",
            "只用这一个特征做逻辑回归的 5 折 AUC；条形自 0.40 起算，0.50 = 无判别力",
        )
        discrimination_card.setMinimumWidth(420)
        discrimination_items = tuple(
            sorted(
                ((feature.label, feature.univariate_auc) for feature in result.features),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        discrimination_body.addWidget(
            _RankedBars(
                discrimination_items,
                COLORS["teal"],
                minimum=0.40,
                maximum=0.80,
                baseline=0.50,
                decimals=3,
            ),
            1,
        )
        row.addWidget(discrimination_card, 1)
        return holder

    @staticmethod
    def _lower_row(result: FeatureInsightResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(340)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        heatmap_card, heatmap_body = _card(
            "连续特征相关性",
            "Spearman 秩相关；相关性强的一对特征会互相替代",
        )
        heatmap_card.setMinimumWidth(360)
        heatmap_body.addWidget(_CorrelationHeatmap(result), 1)
        row.addWidget(heatmap_card, 4)

        group_card, group_body = _card(
            "阳性组与阴性组差异",
            "连续特征的组内均值、差值（阳性组 - 阴性组）与标准化效应量",
        )
        group_card.setMinimumWidth(380)
        group_body.addWidget(_GroupTable(result), 0)
        group_body.addWidget(
            _label(
                "|Cohen's d| ≥ 0.8 视为大效应；差值单位与指标单位一致。",
                size=9,
                color=COLORS["text_muted"],
            )
        )
        group_body.addStretch(1)
        row.addWidget(group_card, 5)
        return holder

    @staticmethod
    def _note(result: FeatureInsightResult) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(34)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(
            _icon_label("sparkles", COLORS["primary"]),
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        text = _label(_insight_text(result), size=9, color=COLORS["text_secondary"])
        text.setWordWrap(True)
        row.addWidget(text, 1)
        return holder


def _icon_label(kind: str, color: str) -> QLabel:
    label = QLabel()
    label.setPixmap(make_icon_pixmap(kind, 13, color))
    label.setFixedSize(13, 13)
    return label


def _insight_text(result: FeatureInsightResult) -> str:
    """One sentence pairing the strongest model driver with a contradicting case."""

    features = result.features
    if not features:
        return ""
    top = features[0]
    ranked = sorted(features, key=lambda item: item.univariate_auc, reverse=True)
    strongest_single = ranked[0]
    weakest_model = sorted(features, key=lambda item: item.permutation)[0]

    contrast = strongest_single if strongest_single.key != top.key else None
    if contrast is None:
        contrast = next(
            (
                feature
                for feature in ranked[1:]
                if feature.permutation < 0.25 * max(top.permutation, 1e-9)
            ),
            None,
        )

    text = (
        f"模型最依赖「{top.label}」：置换后 AUC 变化 {top.permutation:+.3f}，"
        f"其单变量 AUC 为 {top.univariate_auc:.3f}。"
    )
    if contrast is not None:
        text += (
            f"「{contrast.label}」单凭自身可达 AUC {contrast.univariate_auc:.3f}，"
            f"但置换重要性仅 {contrast.permutation:+.3f}，相关信息已被其他特征分摊。"
        )
    if weakest_model.permutation <= 0:
        text += f"「{weakest_model.label}」置换后 AUC 不降反升（{weakest_model.permutation:+.3f}），可视为无贡献。"
    return text


def feature_stylesheet() -> str:
    return f"""
QFrame#insightCard {{ background: {COLORS["surface"]}; border: 1px solid {COLORS["border"]}; border-radius: 16px; }}
QFrame#insightTableHead {{ background: {COLORS["surface_alt"]}; border: none; border-radius: 8px; }}
QFrame#insightTableRow {{ background: transparent; border: none; border-bottom: 1px solid {COLORS["grid"]}; }}
"""
