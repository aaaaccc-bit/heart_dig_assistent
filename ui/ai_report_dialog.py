"""Report generation and review panel for the analysis reports page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import math
import os
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QInputDialog,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.provider_settings import load_report_model, report_model_key
from ai.report_generator import AIReportCancelled, AIReportError, AIReportService
from ai.security_settings import load_security_settings
from ui.dashboard_data import DATASET_NAME, DashboardData
from ui.markdown_viewer import MarkdownViewer
from ui.theme import COLORS, DEFAULT_FONT, theme_manager
from algorithm.regression import RegressionResult
from algorithm.random_forest import RandomForestResult


def _theme_color(color: str) -> str:
    aliases = {
        "#17283B": "text",
        "#26384C": "text",
        "#7C8B9D": "text_muted",
        "#53677B": "text_secondary",
        "#0D9B92": "teal",
        "#C24545": "danger",
        "#60758A": "text_secondary",
        "#2B6EDE": "primary",
        "#0B8B64": "success",
    }
    return COLORS.get(aliases.get(color, ""), color)


@dataclass(frozen=True)
class ReportMetric:
    label: str
    value: str
    note: str


@dataclass(frozen=True)
class ReportHighlight:
    title: str
    detail: str


@dataclass(frozen=True)
class CardioReportRecord:
    title: str
    kind_label: str
    dataset_name: str
    created_at: datetime
    trajectory_count: int
    group_count: int
    metrics: tuple[ReportMetric, ...]
    highlights: tuple[ReportHighlight, ...]
    parameters: dict


def _report_number(value):
    """Convert NaN/inf to ``None`` so the prompt contains valid JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _report_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_number(item) for item in value]
    return value


def build_cardio_report_record(
    data: DashboardData,
    regression: RegressionResult | None = None,
    random_forest: RandomForestResult | None = None,
    risk: object | None = None,
    clusters: object | None = None,
    evaluation: object | None = None,
    features: object | None = None,
) -> CardioReportRecord:
    """Adapt the current workbench result to the reference AI report contract.

    Every analysis is optional: the record only carries the sections whose
    results were passed in, and the report prompt is fed the same structure.
    """

    metrics = tuple(
        ReportMetric(metric.label, f"{metric.value}{metric.unit}", metric.badge)
        for metric in data.kpis
    )
    highlights = [
        ReportHighlight("总体结果", f"样本 {data.total_count} 例，阳性 {data.positive_count} 例，阳性率 {data.positive_rate:.1f}%"),
        ReportHighlight("年龄分层", data.insight),
        ReportHighlight("性别与胸痛类型", data.positive_note),
    ]
    parameters = {
        "阳性率": round(data.positive_rate, 2),
        "年龄段": [
            {"分组": band.label, "样本数": band.total, "阳性数": band.positive, "阳性率": band.rate}
            for band in data.age_bands
        ],
        "风险因素": [
            {"指标": factor.label, "比例": round(factor.percent, 2), "摘要": factor.summary}
            for factor in data.risk_factors
        ],
        "组间比较": [
            {"指标": row.indicator, "阳性组": row.positive, "阴性组": row.negative, "差值": row.delta}
            for row in data.comparisons
        ],
        "连续特征分布": [
            {
                "指标": item.label,
                "最小值": item.minimum,
                "Q1": item.q1,
                "中位数": item.median,
                "Q3": item.q3,
                "最大值": item.maximum,
            }
            for item in data.distributions
        ],
    }
    if regression is not None:
        parameters["回归分析"] = {
            "模型": "二元 Logistic 回归" if len(regression.predictors) == 2 else "多元 Logistic 回归",
            "自变量": list(regression.predictors),
            "样本数": regression.n,
            "阳性数": regression.positive,
            "阴性数": regression.negative,
            "系数与优势比": [
                {
                    "变量": item.name,
                    "系数": round(item.coefficient, 6),
                    "OR": round(item.odds_ratio, 6),
                    "95% CI": [round(item.ci_low, 6), round(item.ci_high, 6)],
                    "p值": round(item.p_value, 8),
                }
                for item in regression.coefficients
            ],
            "拟合优度": {
                "LR统计量": round(regression.fit.lr_statistic, 6),
                "LR检验p值": round(regression.fit.lr_p_value, 8),
                "AIC": round(regression.fit.aic, 6),
                "BIC": round(regression.fit.bic, 6),
                "McFadden pseudo-R²": round(regression.fit.mcfadden_r2, 6),
                "HL检验p值": round(regression.fit.hosmer_lemeshow_p_value, 8),
            },
            "验证指标": {
                "ROC-AUC": round(regression.validation.roc_auc, 6),
                "PR-AUC": round(regression.validation.pr_auc, 6),
                "Brier score": round(regression.validation.brier_score, 6),
                "校准截距": round(regression.validation.calibration_intercept, 6),
                "校准斜率": round(regression.validation.calibration_slope, 6),
            },
        }
        highlights.append(
            ReportHighlight(
                "回归分析",
                f"{'二元' if len(regression.predictors) == 2 else '多元'} Logistic 回归 "
                f"ROC-AUC {regression.validation.roc_auc:.3f}，"
                f"Brier {regression.validation.brier_score:.3f}",
            )
        )
    if random_forest is not None:
        validation = random_forest.validation
        parameters["随机森林分析"] = {
            "模型": "Random Forest 二分类模型",
            "自变量": list(random_forest.predictors),
            "样本数": random_forest.n,
            "阳性数": random_forest.positive,
            "阴性数": random_forest.negative,
            "模型参数": {
                "树数量": random_forest.n_estimators,
                "最小叶节点样本数": random_forest.min_samples_leaf,
                "交叉验证折数": random_forest.folds,
                "随机种子": random_forest.seed,
                "数据指纹": random_forest.dataset_fingerprint,
            },
            "验证指标": {
                "ROC-AUC": round(validation.roc_auc, 6),
                "PR-AUC": round(validation.pr_auc, 6),
                "Brier score": round(validation.brier_score, 6),
                "校准后 ROC-AUC": round(validation.calibrated_roc_auc, 6),
                "校准后 PR-AUC": round(validation.calibrated_pr_auc, 6),
                "校准后 Brier score": round(validation.calibrated_brier_score, 6),
                "校准截距": round(validation.calibration_intercept, 6),
                "校准斜率": round(validation.calibration_slope, 6),
                "各折 ROC-AUC": [round(value, 6) for value in validation.cv_roc_auc],
                "各折 PR-AUC": [round(value, 6) for value in validation.cv_pr_auc],
                "各折 Brier score": [round(value, 6) for value in validation.cv_brier_score],
            },
            "特征重要性": [
                {"特征": name, "重要性": round(value, 6)}
                for name, value in random_forest.feature_importance
            ],
            "阈值评估": [
                {
                    "阈值": round(row[0], 3),
                    "Accuracy": round(row[1], 6),
                    "Sensitivity": round(row[2], 6),
                    "Specificity": round(row[3], 6),
                    "PPV": round(row[4], 6),
                    "NPV": round(row[5], 6),
                    "F1": round(row[6], 6),
                }
                for row in validation.threshold_metrics
            ],
            "决策曲线净获益": [
                {
                    "阈值": round(threshold, 3),
                    "模型": round(model_value, 6),
                    "全部干预": round(treat_all, 6),
                    "全部不干预": round(treat_none, 6),
                }
                for threshold, model_value, treat_all, treat_none in zip(
                    validation.decision_curve_thresholds,
                    validation.decision_curve_net_benefit,
                    validation.decision_curve_treat_all,
                    validation.decision_curve_treat_none,
                )
            ],
        }
        highlights.append(
            ReportHighlight(
                "随机森林",
                f"ROC-AUC {validation.roc_auc:.3f}，PR-AUC {validation.pr_auc:.3f}，"
                f"校准后 Brier {validation.calibrated_brier_score:.3f}",
            )
        )
    if risk is not None:
        parameters["风险分层"] = {
            "模型": risk.model,
            "概率阈值": list(risk.thresholds),
            "样本数": risk.n,
            "阳性数": risk.positive,
            "验证指标": {
                "ROC-AUC": round(risk.roc_auc, 6),
                "Brier score": round(risk.brier_score, 6),
                "交叉验证折数": risk.folds,
                "随机种子": risk.seed,
                "数据指纹": risk.dataset_fingerprint,
            },
            "分档": [
                {
                    "档位": stratum.label,
                    "例数": stratum.count,
                    "占比": round(stratum.percent, 2),
                    "实际阳性": stratum.positive,
                    "实际阳性率": round(stratum.positive_rate, 2),
                }
                for stratum in risk.strata
            ],
            "关键风险因素": [
                {"因素": factor.label, "贡献占比": round(factor.share, 2)}
                for factor in risk.factors
            ],
            "高危人群画像": [
                {
                    "指标": axis.label,
                    "高危人群": round(axis.high_rate, 2),
                    "全样本": round(axis.overall_rate, 2),
                }
                for axis in risk.profile
            ],
        }
        extremes = next(
            (stratum for stratum in risk.strata if stratum.key == "critical"), None
        )
        if extremes is None:
            extreme_text = "四档风险分层已完成"
        else:
            extreme_text = (
                f"极高风险 {extremes.count} 例（{extremes.percent:.1f}%），"
                f"实际阳性率 {extremes.positive_rate:.1f}%"
            )
        highlights.append(ReportHighlight("风险分档", extreme_text))
    if clusters is not None:
        results = clusters if isinstance(clusters, (tuple, list)) else (clusters,)
        parameters["无监督分群"] = [
            {
                "算法": item.method,
                "候选簇数": [run.k for run in item.runs],
                "选定簇数": item.selected.k,
                "轮廓系数": round(item.selected.metric.silhouette, 6),
                "ARI（与阳性标签）": round(item.selected.metric.ari, 6),
                "PCA解释度": [round(value, 6) for value in item.explained],
                "各簇": [
                    {
                        "簇": group.label,
                        "例数": group.count,
                        "阳性率": round(group.positive_rate, 2),
                        **{
                            feature.label: round(feature.mean, 2)
                            for feature in group.features
                        },
                    }
                    for group in item.selected.groups
                ],
            }
            for item in results
        ]
        primary = results[0]
        first_group = primary.selected.groups[0]
        last_group = primary.selected.groups[-1]
        if primary.has_singletons:
            suffix = "；出现单点簇（链式效应），不宜据此分型"
        elif any(item.has_singletons for item in results):
            suffix = "；另一方法出现单点簇"
        else:
            suffix = ""
        highlights.append(
            ReportHighlight(
                "无监督分群",
                f"{primary.method.split('（')[0]} k={primary.selected.k}："
                f"首末簇阳性率 {first_group.positive_rate:.1f}% / {last_group.positive_rate:.1f}%"
                + suffix,
            )
        )
    if evaluation is not None:
        parameters["分类评估"] = {
            "验证方式": f"{evaluation.folds} 折分层交叉验证（折外概率）",
            "判定阈值": 0.5,
            "模型对比": [
                {
                    "模型": model.label,
                    "ROC-AUC": round(model.roc_auc, 6),
                    "PR-AUC": round(model.pr_auc, 6),
                    "Brier": round(model.brier_score, 6),
                    "准确率": round(model.at(0.5).accuracy, 6),
                    "敏感度": round(model.at(0.5).sensitivity, 6),
                    "特异度": round(model.at(0.5).specificity, 6),
                    "F1": round(model.at(0.5).f1, 6),
                    "MCC": round(model.at(0.5).mcc, 6),
                }
                for model in evaluation.models
            ],
        }
        best = max(evaluation.models, key=lambda model: model.at(0.5).mcc)
        best_metrics = best.at(0.5)
        highlights.append(
            ReportHighlight(
                "分类评估",
                f"{best.label} 综合最优：MCC {best_metrics.mcc:.3f}，"
                f"准确率 {best_metrics.accuracy:.1%}，敏感度 {best_metrics.sensitivity:.1%}",
            )
        )
    if features is not None:
        signals = features.features
        parameters["特征洞察"] = {
            "方法": f"置换重要性（{features.folds} 折内打乱）+ 单变量 5 折 AUC",
            "特征": [
                {
                    "特征": signal.label,
                    "类型": "连续" if signal.kind == "numeric" else "分类",
                    "置换重要性": round(signal.permutation, 4),
                    "单变量AUC": round(signal.univariate_auc, 3),
                    **(
                        {
                            "阳性组均值": round(signal.positive_mean, 2),
                            "阴性组均值": round(signal.negative_mean, 2),
                            "Cohen's d": round(signal.effect_size, 2),
                        }
                        if signal.kind == "numeric"
                        else {}
                    ),
                }
                for signal in signals
            ],
            "相关矩阵": {
                "指标": list(features.correlation.labels),
                "Spearman": [
                    [round(value, 3) for value in row]
                    for row in features.correlation.values
                ],
            },
        }
        top = signals[0]
        strongest = max(signals, key=lambda signal: signal.univariate_auc)
        highlights.append(
            ReportHighlight(
                "特征洞察",
                f"模型最依赖「{top.label}」（置换 {top.permutation:+.3f}）；"
                f"单变量最强「{strongest.label}」（AUC {strongest.univariate_auc:.3f}）",
            )
        )
    return CardioReportRecord(
        title=("Cleveland 心脏风险分析报告" if data.dataset == DATASET_NAME
               else f"{data.dataset} · 心脏风险分析报告"),
        kind_label="心脏病风险与模型评估分析",
        dataset_name=data.dataset,
        created_at=datetime.now(),
        trajectory_count=data.total_count,
        group_count=len(data.age_bands),
        metrics=metrics,
        highlights=tuple(highlights),
        parameters=_report_number(parameters),
    )


class AIReportWorker(QObject):
    """Runs the report workflow on a background thread."""

    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(dict)

    def __init__(
        self,
        service: AIReportService,
        record,
        provider,
        api_key: str,
        model: str | None,
        instruction: str,
    ):
        super().__init__()
        self.service = service
        self.record = record
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.instruction = instruction
        self._cancel_requested = Event()

    def request_cancellation(self):
        self._cancel_requested.set()

    @Slot()
    def run(self):
        try:
            kwargs = {"cancel_requested": self._cancel_requested.is_set}
            parameters = inspect.signature(self.service.generate).parameters
            if "progress_callback" in parameters:
                kwargs["progress_callback"] = self.progress.emit
            if "safety_mode" in parameters:
                security = load_security_settings()
                kwargs["safety_mode"] = security["mode"]
                kwargs["safety_strength"] = security["strength"]
            text = self.service.generate(
                self.record, self.provider, self.api_key, self.model, self.instruction, **kwargs
            )
        except AIReportCancelled:
            self.cancelled.emit()
        except AIReportError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # SSL / 连接重置 / 解码错误等都不能漏
            self.failed.emit(f"生成失败：{exc}")
        else:
            self.completed.emit(text)


class AIReportPanel(QWidget):
    preparation_requested = Signal()
    skip_preparation_requested = Signal()
    generation_finished = Signal()

    def __init__(self, record=None, parent=None, *, service: AIReportService | None = None):
        super().__init__(parent)
        self.record = record
        self.service = service or AIReportService(retrieval_enabled=True)
        self.result_text: str | None = None
        self._thread: QThread | None = None
        self._worker: AIReportWorker | None = None
        self._reset_after_generation = False
        self._status_color = "#7C8B9D"
        self.approval_status = "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at: str | None = None

        self.setObjectName("reportPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(12)
        title = QHBoxLayout()
        title.addWidget(self._label("生成与审阅", 17, "text", True))
        title.addStretch(1)
        self.record_label = self._label("", 12, "text_secondary")
        title.addWidget(self.record_label)
        root.addLayout(title)

        context = QHBoxLayout()
        self.model_summary = self._label("", 12, "text_secondary")
        self.model_summary.setWordWrap(True)
        context.addWidget(self.model_summary, 1)
        self.prepare_button = QPushButton("准备报告")
        self.prepare_button.setObjectName("reportButton")
        self.prepare_button.clicked.connect(self.preparation_requested.emit)
        context.addWidget(self.prepare_button)
        self.skip_button = QPushButton("跳过等待")
        self.skip_button.setObjectName("reportButton")
        self.skip_button.clicked.connect(self.skip_preparation_requested.emit)
        self.skip_button.hide()
        context.addWidget(self.skip_button)
        root.addLayout(context)

        root.addWidget(self._label("报告要求（可选）", 12, "text", True))
        self.instruction_edit = QTextEdit()
        self.instruction_edit.setPlaceholderText("例如：重点解释阳性组与阴性组的差异，并用审慎语气说明统计局限性。")
        self.instruction_edit.setFixedHeight(74)
        root.addWidget(self.instruction_edit)

        root.addWidget(self._label("报告预览", 12, "text", True))
        self.result_edit = MarkdownViewer()
        self.result_edit.setReadOnly(True)
        self.result_edit.setMinimumHeight(200)
        root.addWidget(self.result_edit, 1)
        self.status_label = self._label("", 11, "text_secondary")
        root.addWidget(self.status_label)
        progress_row = QHBoxLayout()
        self.agent_progress_label = self._label("等待生成", 11, "text_secondary")
        progress_row.addWidget(self.agent_progress_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(190)
        progress_row.addWidget(self.progress_bar)
        root.addLayout(progress_row)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setFixedSize(84, 36)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.cancel_button.hide()
        actions.addWidget(self.cancel_button)
        self.clear_button = QPushButton("清空")
        self.clear_button.setFixedSize(84, 36)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear_result)
        actions.addWidget(self.clear_button)
        self.approve_button = QPushButton("通过审核")
        self.approve_button.setFixedSize(96, 36)
        self.approve_button.clicked.connect(self.approve_report)
        self.approve_button.setEnabled(False)
        actions.addWidget(self.approve_button)
        self.revise_button = QPushButton("修订报告")
        self.revise_button.setFixedSize(96, 36)
        self.revise_button.clicked.connect(self.revise_report)
        self.revise_button.setEnabled(False)
        actions.addWidget(self.revise_button)
        self.reject_button = QPushButton("退回")
        self.reject_button.setFixedSize(72, 36)
        self.reject_button.clicked.connect(self.reject_report)
        self.reject_button.setEnabled(False)
        actions.addWidget(self.reject_button)
        self.export_button = QPushButton("导出 Markdown")
        self.export_button.setMinimumWidth(130)
        self.export_button.setFixedHeight(36)
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)
        actions.addWidget(self.export_button)
        self.generate_button = QPushButton("生成分析报告")
        self.generate_button.setFixedSize(126, 36)
        self.generate_button.setCursor(Qt.PointingHandCursor)
        self.generate_button.clicked.connect(self.generate_report)
        actions.addWidget(self.generate_button)
        root.addLayout(actions)
        for button in (self.cancel_button, self.clear_button, self.approve_button,
                       self.revise_button, self.reject_button, self.export_button):
            button.setObjectName("reportButton")
        self.generate_button.setObjectName("reportPrimary")
        self.refresh_report_model()
        self.set_record(record)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setStyleSheet(f"""
        QWidget#reportPanel {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 18px; }}
        QLabel[reportRole="text"] {{ color: {COLORS['text']}; }}
        QLabel[reportRole="text_secondary"] {{ color: {COLORS['text_secondary']}; }}
        QTextEdit {{ background: {COLORS['input']}; color: {COLORS['text']};
            border: 1px solid {COLORS['border']}; border-radius: 9px; padding: 8px;
            font-family: "{DEFAULT_FONT}"; font-size: 12px; }}
        QTextEdit:focus {{ border-color: {COLORS['primary']}; }}
        QProgressBar {{ background: {COLORS['grid']}; border: none; border-radius: 4px; height: 8px; }}
        QProgressBar::chunk {{ background: {COLORS['primary']}; border-radius: 4px; }}
        QPushButton#reportButton, QPushButton#reportPrimary {{
            min-height: 35px; padding: 0 13px; border-radius: 8px; font-size: 12px; font-weight: 600;
        }}
        QPushButton#reportButton {{ background: {COLORS['surface']}; color: {COLORS['text']};
            border: 1px solid {COLORS['border']}; }}
        QPushButton#reportButton:hover {{ background: {COLORS['surface_hover']}; border-color: {COLORS['border_strong']}; }}
        QPushButton#reportButton:disabled {{ color: {COLORS['disabled_text']}; }}
        QPushButton#reportPrimary {{ background: {COLORS['primary']};
            color: {COLORS['navy'] if theme_manager.is_dark else '#FFFFFF'}; border: 1px solid {COLORS['primary']}; }}
        QPushButton#reportPrimary:hover {{ background: {COLORS['primary_hover']}; }}
        QPushButton#reportPrimary:disabled {{ background: {COLORS['disabled_surface']};
            color: {COLORS['disabled_text']}; border-color: {COLORS['border']}; }}
        """)
        self.result_edit.apply_theme()
        self._set_status(self.status_label.text(), self._status_color)

    @staticmethod
    def _label(text: str, size: int, role: str, bold: bool = False) -> QLabel:
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setProperty("reportRole", role)
        font = label.font()
        font.setFamily(DEFAULT_FONT)
        font.setPixelSize(size)
        font.setBold(bold)
        label.setFont(font)
        return label

    def refresh_report_model(self) -> None:
        config = load_report_model()
        provider = config.provider
        model = provider.default_model if provider.is_custom else config.model or "自动选择模型"
        self.model_summary.setText(f"当前模型 · {provider.name} / {model}   ·   在「系统设置」的「报告模型」中修改")

    def set_record(self, record) -> None:
        self.record = record
        self.record_label.setText(record.title if record is not None else "等待分析结果")
        self.prepare_button.setVisible(record is None and not self.is_running())
        self.skip_button.hide()
        self.generate_button.setEnabled(record is not None and not self.is_running())

    def set_preparing(self) -> None:
        self.prepare_button.hide()
        self.skip_button.show()
        self.record_label.setText("正在汇总分析结果…")
        self._set_status("", "#7C8B9D")

    def set_preparation_error(self, message: str) -> None:
        self.set_record(None)
        self._set_status(message, "#C24545")

    def reset_record(self) -> None:
        self.set_record(None)
        if self.is_running():
            self._reset_after_generation = True
            self.cancel_generation()
        else:
            self.clear_result()
            self._set_status("", "#7C8B9D")

    # ------------------------------------------------------------ generation
    def clear_result(self):
        """Clear the current Markdown result so another report can be generated."""
        if self.is_running():
            return
        self.result_text = None
        self.approval_status = "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self._set_review_buttons(False)
        self.result_edit.clear()
        self.progress_bar.setValue(0)
        self.agent_progress_label.setText("等待生成")
        self._set_status("已清空报告内容，可再次生成。", "#7C8B9D")

    def generate_report(self):
        if self.is_running() or self.record is None:
            return
        settings = load_report_model()
        config = settings.provider
        api_key = report_model_key(config)
        try:
            config.request_headers(api_key)
        except ValueError as exc:
            self._set_status(f"{exc} 请在「系统设置」的「报告模型」中更新 {config.name} 的设置。", "#C24545")
            return
        self.result_text = None
        self.approval_status = "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self._set_review_buttons(False)
        self.result_edit.clear()
        self.progress_bar.setValue(0)
        self.agent_progress_label.setText("正在启动 Agent 编排…")
        self._set_status("正在请求模型，请稍候…", "#2B6EDE")
        self._set_running(True)

        thread = QThread(self)
        worker = AIReportWorker(
            self.service,
            self.record,
            config,
            api_key,
            settings.model,
            self.instruction_edit.toPlainText(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.progress.connect(self._on_progress)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        # 模型初始化/下载属于耗时的后台工作，使用低优先级线程，避免
        # 首次生成报告时占用过多 CPU 影响主界面和用户操作。
        thread.start(QThread.LowPriority)

    def cancel_generation(self):
        if not self.is_running():
            return
        self._worker.request_cancellation()
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("正在取消…")
        self._set_status("正在取消本次请求…", "#7C8B9D")

    @Slot(str)
    def _on_completed(self, text: str):
        if self._reset_after_generation:
            return
        self.result_text = text
        self.approval_status = "blocked" if "## 安全硬阻断" in text else "pending_review"
        self.report_version = "1.0"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self.result_edit.set_markdown(text)
        if self.approval_status == "blocked":
            self._set_status("检测到高风险内容，报告已硬阻断，禁止审批和导出。", "#C24545")
            self.agent_progress_label.setText("✕ 安全硬阻断 · 禁止审批和导出")
        elif "## 安全审核提示" in text or "## 机器审核结果" in text:
            self._set_status("报告已生成，需人工复核后再导出。", "#C24545")
            self.agent_progress_label.setText("△ 等待人工审核 · 可修订或退回")
        else:
            self._set_status("报告生成完成，等待人工审核。", "#0B8B64")
            self.agent_progress_label.setText("✓ Agent 编排完成 · 等待人工审核")
        self.progress_bar.setValue(100)

        self._set_review_buttons(self.approval_status != "blocked")

    def _set_review_buttons(self, enabled: bool):
        self.approve_button.setEnabled(enabled and self.approval_status in {"pending_review", "rejected"})
        self.revise_button.setEnabled(enabled and bool(self.result_text))
        self.reject_button.setEnabled(enabled and self.approval_status in {"pending_review", "approved"})
        self.export_button.setEnabled(enabled and self.approval_status == "approved")

    def approve_report(self):
        if not self.result_text or self.is_running():
            return
        if self.approval_status == "blocked":
            self._set_status("安全硬阻断报告不可审批或导出。", "#C24545")
            return
        self.approval_status = "approved"
        self.reviewer = os.getenv("CARDIOSCOPE_REVIEWER", "local_user")
        self.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._set_review_buttons(True)
        self._set_status("已通过人工审核，可以导出报告。", "#0B8B64")

    def revise_report(self):
        if not self.result_text or self.is_running() or self.approval_status == "blocked":
            return
        editor = QDialog(self)
        editor.setWindowTitle("修订 AI 报告")
        editor.resize(760, 560)
        layout = QVBoxLayout(editor)
        text_edit = QTextEdit(editor)
        text_edit.setPlainText(self.result_text)
        layout.addWidget(text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, parent=editor)
        buttons.accepted.connect(editor.accept)
        buttons.rejected.connect(editor.reject)
        layout.addWidget(buttons)
        if editor.exec() != QDialog.Accepted:
            return
        revised = text_edit.toPlainText().strip()
        if not revised:
            self._set_status("修订内容不能为空。", "#C24545")
            return
        self.result_text = revised
        major, minor = (self.report_version.split(".", 1) + ["0"])[:2]
        self.report_version = f"{major}.{int(minor) + 1}"
        self.approval_status = "pending_review"
        self.rejection_reason = ""
        self.reviewer = ""
        self.reviewed_at = None
        self.result_edit.set_markdown(revised)
        self._set_review_buttons(True)
        self._set_status(f"已保存人工修订，当前版本 {self.report_version}，请重新审核。", "#2B6EDE")

    def reject_report(self):
        if not self.result_text or self.is_running() or self.approval_status == "blocked":
            return
        reason, ok = QInputDialog.getText(self, "退回 AI 报告", "退回原因（必填）：")
        reason = reason.strip()
        if not ok or not reason:
            return
        self.approval_status = "rejected"
        self.rejection_reason = reason
        self.reviewer = os.getenv("CARDIOSCOPE_REVIEWER", "local_user")
        self.reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._set_review_buttons(True)
        self._set_status(f"报告已退回：{reason}", "#C24545")

    def export_report(self):
        if self.approval_status != "approved" or not self.result_text:
            self._set_status("导出前必须先通过人工审核。", "#C24545")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 AI 报告", "ai_report.md", "Markdown 文件 (*.md)")
        if not path:
            return
        Path(path).write_text(self.result_text, encoding="utf-8")
        self._set_status("报告已导出。", "#0B8B64")

    @Slot(str)
    def _on_failed(self, message: str):
        self._set_status(message, "#C24545")
        self.agent_progress_label.setText("✕ Agent 编排已停止，请检查错误信息")

    @Slot()
    def _on_cancelled(self):
        self._set_status("已取消本次 AI 报告生成。", "#7C8B9D")
        self.agent_progress_label.setText("已取消 Agent 编排")

    @Slot(dict)
    def _on_progress(self, event: dict):
        kind = event.get("kind")
        agent = event.get("agent", "报告编排")
        completed = int(event.get("completed", 0) or 0)
        total = max(1, int(event.get("total", 1) or 1))
        if kind == "graph_started":
            self.agent_progress_label.setText(f"正在启动主管 Agent · 共 {total} 个节点")
            self.progress_bar.setValue(0)
        elif kind == "wave_started":
            wave = event.get("wave", "?")
            agents = event.get("agents") or []
            self.agent_progress_label.setText(
                f"第 {wave} 波并行执行 · {len(agents)} 个节点 · {completed}/{total}"
            )
        elif kind == "agent_started":
            self.agent_progress_label.setText(f"● 正在执行：{agent} · {completed}/{total}")
        elif kind == "agent_retry":
            attempt = event.get("attempt", 2)
            self.agent_progress_label.setText(f"↻ {agent} 第 {attempt} 次重试 · {completed}/{total}")
        elif kind == "agent_finished":
            status = event.get("status", "completed")
            icon = {"completed": "✓", "degraded": "△", "skipped": "跳过"}.get(status, "✕")
            self.agent_progress_label.setText(f"{icon} {agent} · {completed}/{total}")
        elif kind == "model_selected":
            self._set_status(f"{agent} · {event['model']}", "#2B6EDE")
        elif kind == "prompt_built":
            self._set_status(
                f"{agent} · 输入 {int(event.get('prompt_chars', 0)):,} 字符",
                "#2B6EDE",
            )
        elif kind == "model_waiting":
            self.agent_progress_label.setText(f"● {agent} · 正在等待模型响应")
        elif kind == "model_thinking":
            self.agent_progress_label.setText(f"● {agent} · 模型正在思考")
        elif kind == "model_chunk":
            chars = int(event.get("chars", 0) or 0)
            self.agent_progress_label.setText(f"● {agent} 生成中 · 已接收 {chars:,} 字符")
        elif kind == "wave_finished":
            elapsed_ms = float(event.get("elapsed_ms", 0) or 0)
            wave = event.get("wave", "?")
            self.agent_progress_label.setText(
                f"✓ 第 {wave} 波完成 · {elapsed_ms / 1000:.2f} 秒 · {completed}/{total}"
            )
        elif kind == "graph_finished":
            elapsed_ms = float(event.get("elapsed_ms", 0) or 0)
            self._set_status(
                f"Agent 编排完成 · 总耗时 {elapsed_ms / 1000:.2f} 秒",
                "#0B8B64",
            )
        percent = min(100, int(completed * 100 / total))
        self.progress_bar.setValue(max(self.progress_bar.value(), percent))

    @Slot()
    def _on_thread_finished(self):
        self._thread = None
        self._worker = None
        self._set_running(False)
        if self._reset_after_generation:
            self._reset_after_generation = False
            self.reset_record()
        self.generation_finished.emit()

    # --------------------------------------------------------------- helpers
    def _set_running(self, running: bool):
        self.instruction_edit.setEnabled(not running)
        self.generate_button.setEnabled(not running and self.record is not None)
        self.clear_button.setEnabled(not running)
        if running:
            for button in (self.approve_button, self.revise_button, self.reject_button, self.export_button):
                button.setEnabled(False)
        else:
            self._set_review_buttons(bool(self.result_text) and self.approval_status != "blocked")
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        if running:
            self.cancel_button.setText("取消")

    def _set_status(self, text: str, color: str):
        self._status_color = color
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {_theme_color(color)}; background: transparent;"
            f" font-family: '{DEFAULT_FONT}'; font-size: 9px;"
        )

    def is_running(self) -> bool:
        return self._thread is not None


__all__ = ["AIReportPanel", "AIReportWorker", "CardioReportRecord", "build_cardio_report_record"]
