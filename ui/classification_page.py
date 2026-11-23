"""分类评估页：把三个二分类模型放到同一套折外概率上对比。

The page is owned by :mod:`ui.sample_analysis` (fourth sub-tab).  It compares
logistic regression, random forest and the MLP at one decision threshold, shows
the ROC/PR curves, the confusion matrices and the threshold trade-off, and lets
the reader move the threshold without re-fitting anything.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from algorithm.classification import ClassificationEvaluation, ModelEvaluation
from algorithm.stats import format_number
from ui.theme import COLORS, DEFAULT_FONT, qcolor

MODEL_COLORS = {
    "logistic": "primary",
    "forest": "risk_high",
    "neural": "purple",
}


def _model_color(key: str) -> str:
    return COLORS.get(MODEL_COLORS.get(key, "primary"), COLORS["primary"])


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
    card = QFrame()
    card.setObjectName("sampleCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(8)

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
    body.setSpacing(6)
    layout.addLayout(body, 1)
    return card, body


# --------------------------------------------------------------------------- #
# comparison table
# --------------------------------------------------------------------------- #
_COLUMNS = (
    ("ROC-AUC", "auc", 3, True, False),
    ("PR-AUC", "pr_auc", 3, True, False),
    ("Brier", "brier", 3, False, False),
    ("准确率", "accuracy", 1, True, True),
    ("敏感度", "sensitivity", 1, True, True),
    ("特异度", "specificity", 1, True, True),
    ("F1", "f1", 1, True, True),
    ("MCC", "mcc", 3, True, False),
)


class _ComparisonTable(QWidget):
    """Three models by eight metrics at the current threshold."""

    def __init__(self, result: ClassificationEvaluation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.threshold = 0.5
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        self.update()

    def _value(self, model: ModelEvaluation, key: str) -> float:
        if key == "auc":
            return model.roc_auc
        if key == "pr_auc":
            return model.pr_auc
        if key == "brier":
            return model.brier_score
        return float(getattr(model.at(self.threshold), key))

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        width = float(self.width())
        height = float(self.height())
        margin = 4.0
        model_width = 112.0
        column_width = (width - margin * 2 - model_width) / len(_COLUMNS)
        header_height = 24.0
        note_height = 16.0
        row_height = (height - header_height - note_height - 6.0) / len(self.result.models)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        row_height = min(
            46.0,
            (height - header_height - note_height - 6.0) / len(self.result.models),
        )
        block_top = max(
            2.0,
            (height - note_height - (header_height + row_height * len(self.result.models))) / 2.0,
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["surface_alt"]))
        painter.drawRoundedRect(
            QRectF(margin, block_top, width - margin * 2, header_height), 7, 7
        )
        _text(
            painter,
            QRectF(margin + 10, block_top, model_width, header_height),
            "模型",
            size=9,
            color=COLORS["text_secondary"],
            bold=True,
        )
        for index, (title, _, _, _, _) in enumerate(_COLUMNS):
            left = margin + model_width + column_width * index
            _text(
                painter,
                QRectF(left, block_top, column_width, header_height),
                title,
                size=9,
                color=COLORS["text_secondary"],
                bold=True,
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )

        # best value per column, to highlight it
        best = []
        for _, key, _, higher, _ in _COLUMNS:
            values = [self._value(model, key) for model in self.result.models]
            best.append(max(values) if higher else min(values))

        for row, model in enumerate(self.result.models):
            top = block_top + header_height + row_height * row
            _text(
                painter,
                QRectF(margin + 10, top, model_width, row_height),
                model.label,
                size=9,
                color=COLORS["text"],
                bold=True,
            )
            for index, (_, key, decimals, _, percent) in enumerate(_COLUMNS):
                value = self._value(model, key)
                left = margin + model_width + column_width * index
                highlight = abs(value - best[index]) < 1e-12
                if highlight:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(qcolor(COLORS["primary"], 0.09))
                    painter.drawRoundedRect(
                        QRectF(left + 4, top + row_height / 2 - 12, column_width - 8, 24),
                        7,
                        7,
                    )
                text = (
                    f"{format_number(value * 100)}%"
                    if percent
                    else format_number(value, decimals)
                )
                _text(
                    painter,
                    QRectF(left, top, column_width, row_height),
                    text,
                    size=9,
                    color=COLORS["primary"] if highlight else COLORS["text"],
                    bold=highlight,
                    align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                )

        _text(
            painter,
            QRectF(margin, height - note_height, width - margin * 2, note_height),
            f"判定阈值 {format_number(self.threshold, 2)}；Brier 越低越好，其余越高越好；AUC 与阈值无关。",
            size=8,
            color=COLORS["text_muted"],
        )
        painter.end()


# --------------------------------------------------------------------------- #
# ROC / PR curves
# --------------------------------------------------------------------------- #
class _CurvePanel(QWidget):
    """ROC on the left, precision-recall on the right, shared legend on top."""

    def __init__(self, result: ClassificationEvaluation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        width = float(self.width())
        height = float(self.height())
        legend_height = 18.0
        gap = 16.0
        panel_width = (width - gap * 2) / 2
        panel_height = height - legend_height - 4.0
        if panel_width <= 40.0 or panel_height <= 40.0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # legend
        short = {"logistic": "Logistic", "forest": "随机森林", "neural": "MLP"}
        step = min(142.0, width / 3.0)
        x = 0.0
        for model in self.result.models:
            color = _model_color(model.key)
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawLine(QPointF(x, 9), QPointF(x + 14, 9))
            _text(
                painter,
                QRectF(x + 18, 0, step - 18, 18),
                f"{short.get(model.key, model.label)} {model.roc_auc:.3f}",
                size=8,
                color=COLORS["text_secondary"],
            )
            x += step

        for side, title in ((0, "ROC 曲线"), (1, "PR 曲线")):
            left = gap + side * (panel_width + gap)
            plot = QRectF(left + 28, legend_height + 18, panel_width - 40, panel_height - 48)
            _text(
                painter,
                QRectF(left, legend_height, panel_width, 14),
                title,
                size=9,
                color=COLORS["text"],
                bold=True,
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )

            painter.setPen(QPen(QColor(COLORS["grid"]), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(plot.bottomLeft(), plot.bottomRight())
            painter.drawLine(plot.bottomLeft(), plot.topLeft())

            for fraction in (0.0, 0.5, 1.0):
                y = plot.bottom() - plot.height() * fraction
                painter.setPen(QPen(QColor(COLORS["grid"]), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                _text(
                    painter,
                    QRectF(plot.left() - 26, y - 6, 22, 12),
                    f"{fraction:.1f}",
                    size=7,
                    color=COLORS["text_muted"],
                    align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                )

            if side == 0:
                painter.setPen(QPen(QColor(COLORS["border_strong"]), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(plot.bottomLeft(), plot.topRight())
            else:
                prevalence = self.result.positive / self.result.n
                y = plot.bottom() - plot.height() * prevalence
                painter.setPen(QPen(QColor(COLORS["border_strong"]), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

            for model in self.result.models:
                color = _model_color(model.key)
                if side == 0:
                    xs, ys = model.roc_fpr, model.roc_tpr
                else:
                    xs, ys = model.pr_recall, model.pr_precision
                path = QPainterPath()
                for index, (x_value, y_value) in enumerate(zip(xs, ys)):
                    point = QPointF(
                        plot.left() + plot.width() * min(max(x_value, 0.0), 1.0),
                        plot.bottom() - plot.height() * min(max(y_value, 0.0), 1.0),
                    )
                    if index == 0:
                        path.moveTo(point)
                    else:
                        path.lineTo(point)
                painter.setPen(QPen(QColor(color), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)

            _text(
                painter,
                QRectF(plot.left(), plot.bottom() + 6, plot.width(), 12),
                "1 - 特异度" if side == 0 else "召回率",
                size=7,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )
        painter.end()


# --------------------------------------------------------------------------- #
# confusion matrices
# --------------------------------------------------------------------------- #
class _ConfusionPanel(QWidget):
    """One 2x2 matrix per model at the current threshold."""

    def __init__(self, result: ClassificationEvaluation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.threshold = 0.5
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        width = float(self.width())
        height = float(self.height())
        count = len(self.result.models)
        if count == 0:
            return
        column_width = width / count
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cell_height = 40.0
        content_height = 16.0 + 12.0 + 2.0 * (cell_height + 4.0) + 4.0 + 28.0
        painter.translate(0.0, max(0.0, (height - content_height) / 2.0))

        for index, model in enumerate(self.result.models):
            left = column_width * index
            metrics = model.at(self.threshold)
            label_width = 34.0
            cell_width = (column_width - label_width - 16.0) / 2.0

            _text(
                painter,
                QRectF(left, 0, column_width - 8, 16),
                model.label,
                size=9,
                color=_model_color(model.key),
                bold=True,
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )
            _text(
                painter,
                QRectF(left + label_width, 18, cell_width, 12),
                "预测阳性",
                size=7,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )
            _text(
                painter,
                QRectF(left + label_width + cell_width, 18, cell_width, 12),
                "预测阴性",
                size=7,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )

            cells = (
                ("实际阳性", metrics.true_positive, COLORS["success"]),
                (None, metrics.false_negative, COLORS["danger"]),
                ("实际阴性", metrics.false_positive, COLORS["danger"]),
                (None, metrics.true_negative, COLORS["text_secondary"]),
            )
            for cell_index, (row_label, value, color) in enumerate(cells):
                row = cell_index // 2
                column = cell_index % 2
                top = 32.0 + row * (cell_height + 4.0)
                if row_label is not None:
                    _text(
                        painter,
                        QRectF(left, top, label_width, cell_height),
                        row_label,
                        size=7,
                        color=COLORS["text_muted"],
                        align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    )
                rect = QRectF(
                    left + label_width + column * (cell_width + 2.0),
                    top,
                    cell_width,
                    cell_height,
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(qcolor(color, 0.10))
                painter.drawRoundedRect(rect, 6, 6)
                _text(
                    painter,
                    rect,
                    str(value),
                    size=12,
                    color=color,
                    bold=True,
                    align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                )

            summary_top = 32.0 + 2 * (cell_height + 4.0) + 4.0
            _text(
                painter,
                QRectF(left, summary_top, column_width - 8, 14),
                f"敏感度 {format_number(metrics.sensitivity * 100)}% · "
                f"特异度 {format_number(metrics.specificity * 100)}%",
                size=8,
                color=COLORS["text_secondary"],
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )
            _text(
                painter,
                QRectF(left, summary_top + 14, column_width - 8, 14),
                f"准确率 {format_number(metrics.accuracy * 100)}% · "
                f"PPV {format_number(metrics.ppv * 100)}%",
                size=8,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )
        painter.end()


# --------------------------------------------------------------------------- #
# threshold trade-off
# --------------------------------------------------------------------------- #
class _ThresholdTradeoff(QWidget):
    """Sensitivity / specificity / F1 of one model across the threshold grid."""

    def __init__(
        self,
        result: ClassificationEvaluation,
        model_key: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.model_key = model_key
        self.threshold = 0.5
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_model(self, model_key: str) -> None:
        self.model_key = model_key
        self.update()

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        model = self.result.model(self.model_key)
        width = float(self.width())
        height = float(self.height())
        if width <= 60.0 or height <= 60.0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = QRectF(38.0, 46.0, width - 50.0, height - 84.0)

        _text(painter, QRectF(0, 0, width, 16), f"{model.label} 的阈值权衡", size=9, color=COLORS["text"], bold=True)

        series = (
            ("敏感度", [item.sensitivity for item in model.thresholds], COLORS["danger"]),
            ("特异度", [item.specificity for item in model.thresholds], COLORS["success"]),
            ("F1", [item.f1 for item in model.thresholds], COLORS["primary"]),
        )
        x = 0.0
        for title, _, color in series:
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawLine(QPointF(x, 24), QPointF(x + 12, 24))
            _text(painter, QRectF(x + 16, 15, 54, 18), title, size=8, color=COLORS["text_secondary"])
            x += 62

        painter.setPen(QPen(QColor(COLORS["grid"]), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        for fraction in (0.0, 0.5, 1.0):
            line_y = plot.bottom() - plot.height() * fraction
            painter.setPen(QPen(QColor(COLORS["grid"]), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(plot.left(), line_y), QPointF(plot.right(), line_y))
            _text(
                painter,
                QRectF(plot.left() - 30, line_y - 6, 26, 12),
                f"{fraction:.1f}",
                size=7,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )

        thresholds = self.result.thresholds
        low, high = min(thresholds), max(thresholds)

        def x_at(threshold: float) -> float:
            return plot.left() + (threshold - low) / (high - low) * plot.width()

        # active threshold marker
        marker = x_at(self.threshold)
        painter.setPen(QPen(qcolor(COLORS["text_muted"], 0.55), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(marker, plot.top()), QPointF(marker, plot.bottom()))
        _text(
            painter,
            QRectF(marker - 24, plot.top() + 2, 48, 12),
            f"{self.threshold:.2f}",
            size=7,
            color=COLORS["text_secondary"],
            align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
        )

        for _, values, color in series:
            path = QPainterPath()
            for index, (threshold, value) in enumerate(zip(thresholds, values)):
                point = QPointF(x_at(threshold), plot.bottom() - plot.height() * min(max(value, 0.0), 1.0))
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            painter.setPen(QPen(QColor(color), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            for threshold, value in zip(thresholds, values):
                if abs(threshold - self.threshold) < 1e-9:
                    painter.drawEllipse(
                        QPointF(x_at(threshold), plot.bottom() - plot.height() * min(max(value, 0.0), 1.0)),
                        3.2,
                        3.2,
                    )

        for index, threshold in enumerate(thresholds):
            if index % 2:
                continue
            _text(
                painter,
                QRectF(x_at(threshold) - 16, plot.bottom() + 6, 32, 12),
                f"{threshold:.2f}",
                size=7,
                color=COLORS["text_muted"],
                align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            )
        _text(
            painter,
            QRectF(plot.left(), plot.bottom() + 18, plot.width(), 12),
            "判定阈值",
            size=7,
            color=COLORS["text_muted"],
            align=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
        )
        painter.end()


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
class ClassificationEvaluationPage(QWidget):
    """分类评估: three models, one threshold grid, one set of folds."""

    def __init__(self, result: ClassificationEvaluation, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.threshold = 0.50
        self.setMinimumSize(900, 740)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)
        root.addWidget(self._header(result))
        root.addWidget(self._compare_row(result))
        root.addWidget(self._lower_row(result))
        root.addStretch(1)

    def _header(self, result: ClassificationEvaluation) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(40)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.addWidget(_label("分类评估", size=15, bold=True))
        titles.addWidget(
            _label(
                f"同一 {result.folds} 折分层交叉验证 · 折外概率 · "
                f"{result.n} 例（阳性 {result.positive}）",
                size=9,
                color=COLORS["text_secondary"],
            )
        )
        row.addLayout(titles)
        row.addStretch(1)

        row.addWidget(_label("判定阈值", size=9, color=COLORS["text_secondary"]))
        combo = QComboBox()
        combo.setFixedHeight(30)
        combo.setMinimumWidth(86)
        combo.addItems([f"{value:.2f}" for value in result.thresholds])
        combo.setCurrentIndex(result.thresholds.index(0.50))
        combo.setStyleSheet(
            f"QComboBox {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 7px; padding: 0 8px; color: {COLORS['text']}; font-size: 10px; }}"
        )
        combo.currentIndexChanged.connect(lambda index: self._set_threshold(result.thresholds[index]))
        row.addWidget(combo, 0, Qt.AlignmentFlag.AlignVCenter)
        return holder

    def _compare_row(self, result: ClassificationEvaluation) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(330)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        table_card, table_body = _card("模型对比", "同一阈值下的判别指标；每列最优值高亮")
        table_card.setMinimumWidth(520)
        self._table = _ComparisonTable(result)
        table_body.addWidget(self._table, 1)
        row.addWidget(table_card, 4)

        curve_card, curve_body = _card("ROC 与 PR 曲线", "每条曲线都由同一份折外概率计算")
        curve_card.setMinimumWidth(360)
        curve_body.addWidget(_CurvePanel(result), 1)
        row.addWidget(curve_card, 3)
        return holder

    def _lower_row(self, result: ClassificationEvaluation) -> QWidget:
        holder = QWidget()
        holder.setFixedHeight(400)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        matrix_card, matrix_body = _card(
            "混淆矩阵",
            "每个模型在当前阈值下的四格计数；绿=正确、红=错误",
        )
        matrix_card.setMinimumWidth(520)
        self._confusion = _ConfusionPanel(result)
        matrix_body.addWidget(self._confusion, 1)
        matrix_body.addWidget(
            _label(
                "TP：实际阳性且判为阳性；FN：实际阳性但判为阴性；阈值改变时四格同步更新。",
                size=9,
                color=COLORS["text_muted"],
            )
        )
        row.addWidget(matrix_card, 4)

        selector = QComboBox()
        selector.setFixedHeight(28)
        selector.setMinimumWidth(120)
        for model in result.models:
            selector.addItem(model.label, model.key)
        selector.setStyleSheet(
            f"QComboBox {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 7px; padding: 0 8px; color: {COLORS['text']}; font-size: 10px; }}"
        )

        tradeoff_card, tradeoff_body = _card(
            "阈值权衡",
            "移动阈值会同时改变敏感度与特异度",
            extra=selector,
        )
        tradeoff_card.setMinimumWidth(360)
        self._tradeoff = _ThresholdTradeoff(result, result.models[0].key)
        tradeoff_body.addWidget(self._tradeoff, 1)
        selector.currentIndexChanged.connect(
            lambda index: self._tradeoff.set_model(result.models[index].key)
        )
        row.addWidget(tradeoff_card, 3)
        return holder

    def _set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        self._table.set_threshold(threshold)
        self._confusion.set_threshold(threshold)
        self._tradeoff.set_threshold(threshold)


__all__ = ["ClassificationEvaluationPage"]
