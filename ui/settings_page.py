"""Settings page for appearance, report safety and the custom report API."""

from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRunnable, QSize, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from ai.provider_settings import (
    ReportModelSettings, build_preset_model, load_report_model, report_key_env_name,
    report_key_from_env, report_model_key, report_session_key, save_custom_provider,
    save_report_model,
)
from ai.providers import CUSTOM_PROVIDER_NAME, PROVIDER_PRESETS, ProviderConfig
from ai.report_generator import AIReportError, AIReportService
from ai.security_settings import load_security_settings, save_security_settings
from ui.icons import IconWidget, make_icon_pixmap
from ui.theme import COLORS, DEFAULT_FONT, theme_manager


_AUTO_MODEL = "自动选择"


class _ProbeSignals(QObject):
    finished = Signal(bool, str, object)


class _ConnectionProbe(QRunnable):
    def __init__(self, config: ReportModelSettings, api_key: str, *, fetch_models: bool) -> None:
        super().__init__()
        self.config = config
        self.api_key = api_key
        self.fetch_models = fetch_models
        self.signals = _ProbeSignals()

    def run(self) -> None:
        try:
            if self.fetch_models:
                models = AIReportService.list_models(self.config.provider, self.api_key)
                if not models:
                    self.signals.finished.emit(False, "接口没有返回模型，可手动输入模型 ID。", None)
                else:
                    self.signals.finished.emit(True, f"已获取 {len(models)} 个模型，请从下拉列表选择。", models)
            else:
                AIReportService.test_connection(self.config.provider, self.api_key, model=self.config.model)
                self.signals.finished.emit(True, "连接成功，模型已返回文本。", None)
        except (ValueError, AIReportError) as exc:
            self.signals.finished.emit(False, str(exc), None)


class _SettingsComboBox(QComboBox):
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = COLORS["text_secondary"] if self.isEnabled() else COLORS["disabled_text"]
            pen = QPen(QColor(color))
            pen.setWidthF(1.5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            x, y = self.width() - 14, self.height() / 2
            painter.drawLine(QPointF(x - 3, y - 1), QPointF(x, y + 2))
            painter.drawLine(QPointF(x, y + 2), QPointF(x + 3, y - 1))


def _label(text: str, size: int = 12, *, color: str = "text", bold: bool = False) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    font = QFont(DEFAULT_FONT)
    font.setPixelSize(size)
    font.setBold(bold)
    label.setFont(font)
    label.setStyleSheet(f"color: {COLORS[color]}; background: transparent;")
    return label


def _description(text: str) -> QLabel:
    label = _label(text, color="text_secondary")
    label.setWordWrap(True)
    return label


def _button(text: str, *, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setObjectName("settingsPrimary" if primary else "settingsButton")
    return button


def _card(title: str, detail: str, icon: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("settingsCard")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 20)
    layout.setSpacing(12)
    header = QHBoxLayout()
    header.setSpacing(12)
    tile = QFrame()
    tile.setObjectName("settingsTile")
    tile.setFixedSize(40, 40)
    tile_layout = QHBoxLayout(tile)
    tile_layout.setContentsMargins(0, 0, 0, 0)
    tile_layout.addWidget(IconWidget(icon, 19, COLORS["primary"]), 0, Qt.AlignmentFlag.AlignCenter)
    header.addWidget(tile, 0, Qt.AlignmentFlag.AlignTop)
    labels = QVBoxLayout()
    labels.setSpacing(4)
    labels.addWidget(_label(title, 16, bold=True))
    labels.addWidget(_description(detail))
    header.addLayout(labels, 1)
    layout.addLayout(header)
    layout.addSpacing(5)
    return frame, layout


class SettingsPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._probe: _ConnectionProbe | None = None
        self.setObjectName("settingsPage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_stylesheet())
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(0)

        heading = _label("报告配置", 14, bold=True)
        heading.setFixedHeight(37)
        root.addWidget(heading)
        divider = QFrame()
        divider.setObjectName("settingsDivider")
        divider.setFixedHeight(1)
        root.addWidget(divider)
        root.addSpacing(16)

        columns = QHBoxLayout()
        columns.setSpacing(16)
        columns.addWidget(self._api_card(), 7, Qt.AlignmentFlag.AlignTop)
        side = QVBoxLayout()
        side.setSpacing(16)
        side.addWidget(self._appearance_card())
        side.addWidget(self._safety_card())
        side.addStretch(1)
        columns.addLayout(side, 4)
        root.addLayout(columns)
        root.addStretch(1)
        self.provider_picker.setCurrentText(load_report_model().provider.name)
        self._load_provider()

    def _api_card(self) -> QFrame:
        card, layout = _card("报告模型", "管理 AI 分析报告使用的供应商和连接参数。", "sparkles")
        self.active_label = _label("", 12, color="primary", bold=True)
        layout.addWidget(self.active_label)

        fields = QHBoxLayout()
        fields.setSpacing(12)
        self.provider_picker = _SettingsComboBox()
        self.provider_picker.addItems([*PROVIDER_PRESETS, CUSTOM_PROVIDER_NAME])
        fields.addLayout(self._field("供应商", self.provider_picker), 1)
        self.protocol = _SettingsComboBox()
        self.protocol.addItem("OpenAI 兼容", "openai")
        self.protocol.addItem("Anthropic 兼容", "anthropic")
        fields.addLayout(self._field("API 协议", self.protocol), 1)
        layout.addLayout(fields)

        self.model = _SettingsComboBox()
        self.model.setEditable(True)
        self.model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(12)
        self.model.lineEdit().setPlaceholderText("选择或输入模型 ID")
        self.fetch_button = _button("获取模型")
        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_row.addLayout(self._field("模型", self.model), 1)
        model_row.addWidget(self.fetch_button, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(model_row)

        self.address = QLineEdit()
        self.address.setPlaceholderText("https://api.example.com/v1")
        layout.addLayout(self._field("API 地址", self.address))
        self.address_hint = _description("")
        layout.addWidget(self.address_hint)

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("输入 API Key")
        layout.addLayout(self._field("API Key", self.api_key))
        self.key_hint = _description("")
        layout.addWidget(self.key_hint)

        note = _description("生成报告会向所选服务发送分析摘要。测试连接仅发送一条短消息。")
        note.setObjectName("settingsNote")
        note.setStyleSheet("")
        layout.addWidget(note)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.remove_button = _button("移除自定义 API")
        self.probe_button = _button("测试连接")
        self.save_button = _button("保存并用于报告", primary=True)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        actions.addWidget(self.probe_button)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self.api_status = _description("")
        self.api_status.setMinimumHeight(18)
        layout.addWidget(self.api_status)

        self.provider_picker.currentIndexChanged.connect(self._load_provider)
        self.protocol.activated.connect(self._connection_edited)
        self.address.textEdited.connect(self._connection_edited)
        self.model.activated.connect(self._mark_edited)
        self.model.lineEdit().textEdited.connect(self._mark_edited)
        self.api_key.textEdited.connect(self._clear_model_results)
        self.fetch_button.clicked.connect(self._fetch_models)
        self.remove_button.clicked.connect(self._remove_custom_provider)
        self.probe_button.clicked.connect(self._test_connection)
        self.save_button.clicked.connect(self._save_report_config)
        return card

    @staticmethod
    def _field(text: str, widget: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(6)
        label = _label(text, 12, bold=True)
        label.setBuddy(widget)
        widget.setAccessibleName(text)
        layout.addWidget(label)
        layout.addWidget(widget)
        return layout

    def _appearance_card(self) -> QFrame:
        card, layout = _card("界面外观", "选择整个工作台的显示主题。", "sun")
        row = QHBoxLayout()
        row.setSpacing(8)
        group = QButtonGroup(card)
        for mode, text, icon in (("light", "浅色", "sun"), ("dark", "深色", "moon")):
            button = _button(text)
            button.setCheckable(True)
            button.setChecked(mode == theme_manager.mode)
            button.setIcon(make_icon_pixmap(icon, 15, COLORS["text_secondary"]))
            button.setIconSize(QSize(15, 15))
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _checked=False, mode=mode: theme_manager.set_mode(mode))
            group.addButton(button)
            row.addWidget(button)
        layout.addLayout(row)
        return card

    def _safety_card(self) -> QFrame:
        card, layout = _card("报告安全", "设定 AI 报告的高风险处理方式。", "shield")
        safety = load_security_settings()
        self.safety_mode = _SettingsComboBox()
        self.safety_mode.addItem("阻断高风险报告", "hard_block")
        self.safety_mode.addItem("标记并转人工复核", "human_review")
        self.safety_mode.setCurrentIndex(self.safety_mode.findData(safety["mode"]))
        layout.addLayout(self._field("高风险处理", self.safety_mode))

        row = QHBoxLayout()
        row.addWidget(_label("安全对齐强度", 12, bold=True))
        row.addStretch(1)
        self.strength_label = _label(str(safety["strength"]), 12, color="primary", bold=True)
        row.addWidget(self.strength_label)
        layout.addLayout(row)
        self.safety_strength = QSlider(Qt.Orientation.Horizontal)
        self.safety_strength.setAccessibleName("安全对齐强度")
        self.safety_strength.setRange(0, 100)
        self.safety_strength.setValue(int(safety["strength"]))
        layout.addWidget(self.safety_strength)
        layout.addWidget(_description("选择阻断策略时，强度达到 50 才会阻断高风险报告。修改即时保存。"))
        self.safety_mode.currentIndexChanged.connect(self._save_safety)
        self.safety_strength.valueChanged.connect(self._save_safety)
        return card

    def _save_safety(self) -> None:
        strength = self.safety_strength.value()
        self.strength_label.setText(str(strength))
        save_security_settings(mode=self.safety_mode.currentData(), strength=strength)

    def _update_connection_hints(self) -> None:
        provider_name = self.provider_picker.currentText()
        suffix = "messages" if self.protocol.currentData() == "anthropic" else "chat/completions"
        self.address_hint.setText(f"支持基础地址或以 /{suffix} 结尾的完整地址。")
        if provider_name == CUSTOM_PROVIDER_NAME:
            self.key_hint.setText(
                f"手输密钥仅在本次会话有效，也可读取 {report_key_env_name(provider_name)}。本机免鉴权服务可留空。"
            )
        elif self.address.text().strip().rstrip("/") not in ("", PROVIDER_PRESETS[provider_name].base_url):
            self.key_hint.setText("使用覆盖地址时，请输入该服务的密钥。手输密钥仅在本次会话有效。")
        else:
            self.key_hint.setText(f"手输密钥仅在本次会话有效，也可读取 {report_key_env_name(provider_name)}。")

    def _raw_fields(self) -> tuple[str, str, str, str, str | None]:
        name = self.provider_picker.currentText()
        model = self.model.currentText().strip()
        if name != CUSTOM_PROVIDER_NAME and (not model or model == _AUTO_MODEL):
            model = None
        return name, self.protocol.currentData(), self.address.text(), self.api_key.text(), model

    def draft(self) -> tuple[str, str, str, str, str | None] | None:
        fields = self._raw_fields()
        return fields if fields != self._saved_fields or fields[0] != self._active_name else None

    def restore_draft(self, fields: tuple[str, str, str, str, str | None]) -> None:
        provider_name, protocol, address, key, model = fields
        self.provider_picker.setCurrentText(provider_name)
        self.protocol.setCurrentIndex(self.protocol.findData(protocol))
        self.address.setText(address)
        self.api_key.setText(key)
        self._populate_models(model)
        self._update_connection_hints()
        self._mark_edited()

    def _load_provider(self) -> None:
        active = load_report_model()
        self._active_name = active.provider.name
        self.active_label.setText(f"当前用于报告 · {self._active_name}")
        name = self.provider_picker.currentText()
        saved = load_report_model(name)
        is_custom = name == CUSTOM_PROVIDER_NAME
        provider = saved.provider if saved is not None else None
        self.protocol.setCurrentIndex(self.protocol.findData(provider.protocol) if provider is not None else 0)
        self.protocol.setEnabled(is_custom)
        model = None
        if saved is not None:
            model = provider.default_model if is_custom else saved.model
        self._populate_models(model)
        self.address.setText(provider.base_url if provider is not None else "")
        self.api_key.setText(report_session_key(provider) if provider is not None else "")
        self.remove_button.setVisible(is_custom)
        self.remove_button.setEnabled(is_custom and saved is not None)
        self._saved_fields = self._raw_fields()
        self._update_connection_hints()
        self._mark_edited()

    def _populate_models(self, selected: str | None, fetched: tuple[str, ...] = ()) -> None:
        self._fetched_models = fetched
        preset = PROVIDER_PRESETS.get(self.provider_picker.currentText())
        entries = fetched
        if preset is not None:
            entries = (_AUTO_MODEL, *(fetched or preset.models))
            if not selected:
                selected = _AUTO_MODEL
        if selected and selected not in entries:
            entries = (*entries, selected)
        self.model.clear()
        self.model.addItems(entries)
        self.model.setCurrentIndex(self.model.findText(selected) if selected else -1)

    def _clear_model_results(self) -> None:
        self._populate_models(self._raw_fields()[-1])
        self._mark_edited()

    def _connection_edited(self) -> None:
        _, protocol, address, key, _ = self._saved_fields
        if key and self.api_key.text() == key and (
            self.protocol.currentData() != protocol or self.address.text() != address
        ):
            self.api_key.clear()
        self._update_connection_hints()
        self._clear_model_results()

    def _mark_edited(self) -> None:
        if self._raw_fields() != self._saved_fields:
            self._status("有未保存的修改。")
        elif self.provider_picker.currentText() != self._active_name:
            self._status("保存后，AI 分析报告将使用此供应商。")
        else:
            self._status("")

    def _status(self, text: str, color: str = "text_secondary") -> None:
        self.api_status.setText(text)
        self.api_status.setStyleSheet(f"color: {COLORS[color]}; background: transparent;")

    def _read_config(self) -> ReportModelSettings:
        name, protocol, address, _, model = self._raw_fields()
        if name == CUSTOM_PROVIDER_NAME:
            return ReportModelSettings(ProviderConfig.custom(protocol, address, model or ""))
        return build_preset_model(name, address, model)

    @Slot()
    def _save_report_config(self) -> None:
        try:
            config = self._read_config()
            save_report_model(config, self.api_key.text())
        except (ValueError, OSError) as exc:
            self._status(str(exc), "danger")
            return
        fetched = self._fetched_models
        self._load_provider()
        if fetched:
            self._populate_models(self._raw_fields()[-1], fetched)
        provider = config.provider
        if provider.requires_api_key and not report_model_key(provider):
            self._status("配置已保存。生成报告前请提供对应的 API Key。")
        else:
            self._status("配置已保存，AI 分析报告将使用此模型服务。", "success")

    @Slot()
    def _remove_custom_provider(self) -> None:
        try:
            save_custom_provider(None)
        except OSError as exc:
            self._status(str(exc), "danger")
        else:
            self._load_provider()
            self._status(f"已移除自定义 API。报告当前使用 {self._active_name}。")

    def _set_probing(self, probing: bool) -> None:
        is_custom = self.provider_picker.currentText() == CUSTOM_PROVIDER_NAME
        for widget in (self.provider_picker, self.address, self.model, self.api_key,
                       self.fetch_button, self.probe_button, self.save_button):
            widget.setEnabled(not probing)
        self.protocol.setEnabled(is_custom and not probing)
        self.remove_button.setEnabled(is_custom and not probing and load_report_model(CUSTOM_PROVIDER_NAME) is not None)

    @Slot()
    def _test_connection(self) -> None:
        self._start_probe(fetch_models=False)

    @Slot()
    def _fetch_models(self) -> None:
        self._start_probe(fetch_models=True)

    def _model_list_config(self) -> ReportModelSettings:
        name, protocol, address, _, _ = self._raw_fields()
        if name == CUSTOM_PROVIDER_NAME:
            base_url = ProviderConfig.normalize_base_url(protocol, address)
            return ReportModelSettings(ProviderConfig(name, protocol, base_url, "", ()))
        return build_preset_model(name, address)

    def _start_probe(self, *, fetch_models: bool) -> None:
        try:
            config = self._model_list_config() if fetch_models else self._read_config()
            provider = config.provider
            env_key = "" if provider.has_overridden_preset_url else report_key_from_env(provider.name)
            key = self.api_key.text().strip() or env_key
            provider.request_headers(key)
        except ValueError as exc:
            self._status(str(exc), "danger")
            return
        self._set_probing(True)
        self._status("正在获取模型…" if fetch_models else "正在测试连接…", "primary")
        self._probe = _ConnectionProbe(config, key, fetch_models=fetch_models)
        self._probe.signals.finished.connect(self._finish_probe)
        QThreadPool.globalInstance().start(self._probe)

    @Slot(bool, str, object)
    def _finish_probe(self, success: bool, message: str, models: tuple[str, ...] | None) -> None:
        self._probe = None
        self._set_probing(False)
        if models:
            self._populate_models(self._raw_fields()[-1], models)
        self._status(message, "success" if success else "danger")


def _stylesheet() -> str:
    c = COLORS
    return f"""
QWidget#settingsPage {{ background: {c['page']}; }}
QFrame#settingsDivider {{ background: {c['border']}; }}
QFrame#settingsCard {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 18px; }}
QFrame#settingsTile {{ background: {c['primary_light']}; border-radius: 11px; }}
QLabel#settingsNote {{ background: {c['surface_alt']}; color: {c['text_secondary']};
    border: 1px solid {c['border']}; border-radius: 9px; padding: 9px 11px; font-size: 12px; }}
QLineEdit, QComboBox {{ background: {c['input']}; color: {c['text']}; font-size: 13px;
    border: 1px solid {c['border']}; border-radius: 8px; min-height: 35px; padding: 0 9px; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {c['primary']}; }}
QComboBox {{ padding-right: 28px; }}
QComboBox QLineEdit {{ border: none; background: transparent; min-height: 0; padding: 0; }}
QComboBox::drop-down {{ subcontrol-origin: padding; width: 26px; border: none; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; color: {c['text']};
    selection-background-color: {c['primary_light']}; selection-color: {c['text']}; }}
QPushButton {{ min-height: 35px; padding: 0 13px; border-radius: 8px; font-size: 12px; font-weight: 600; }}
QPushButton#settingsButton {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; }}
QPushButton#settingsButton:hover {{ background: {c['surface_hover']}; border-color: {c['border_strong']}; }}
QPushButton#settingsButton:checked {{ background: {c['primary_light']}; color: {c['primary']}; border-color: {c['primary']}; }}
QPushButton#settingsButton:disabled {{ color: {c['disabled_text']}; border-color: {c['border']}; }}
QPushButton#settingsPrimary {{ background: {c['primary']}; color: {c['navy'] if theme_manager.is_dark else '#FFFFFF'};
    border: 1px solid {c['primary']}; }}
QPushButton#settingsPrimary:hover {{ background: {c['primary_hover']}; }}
QPushButton#settingsPrimary:disabled {{ background: {c['disabled_surface']}; color: {c['disabled_text']};
    border-color: {c['border']}; }}
QSlider::groove:horizontal {{ height: 5px; border-radius: 2px; background: {c['border']}; }}
QSlider::sub-page:horizontal {{ border-radius: 2px; background: {c['primary']}; }}
QSlider::handle:horizontal {{ background: {c['primary']}; width: 14px; margin: -5px 0;
    border: 1px solid {c['surface']}; border-radius: 7px; }}
"""
