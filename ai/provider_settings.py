"""Local report model preferences. Hand-entered credentials stay in memory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

from PySide6.QtCore import QSettings

from .providers import CUSTOM_PROVIDER_NAME, PROVIDER_PRESETS, ProviderConfig
from .security_settings import APPLICATION, ORGANIZATION

CUSTOM_API_KEY_ENV = "CARDIOSCOPE_CUSTOM_API_KEY"
_SETTINGS_KEY = "ai/custom_provider"
_ACTIVE_KEY = "ai/report_provider"
_PRESET_KEY = "ai/report_presets"
_DEFAULT_PROVIDER = next(iter(PROVIDER_PRESETS))
_KEY_ENV = {
    "OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY", "DeepSeek": "DEEPSEEK_API_KEY",
    "Kimi": "MOONSHOT_API_KEY", "GLM": "GLM_API_KEY", "豆包": "DOUBAO_API_KEY",
    CUSTOM_PROVIDER_NAME: CUSTOM_API_KEY_ENV,
}
_preset_credentials: dict[str, tuple[str, str]] = {}
_session_credential: tuple[ProviderConfig, str] | None = None


@dataclass(frozen=True)
class CustomProviderSettings:
    provider: ProviderConfig
    use_by_default: bool = False


@dataclass(frozen=True)
class ReportModelSettings:
    provider: ProviderConfig
    model: str | None = None


def load_custom_provider() -> CustomProviderSettings | None:
    raw = QSettings(ORGANIZATION, APPLICATION).value(_SETTINGS_KEY, "", type=str)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return CustomProviderSettings(
            ProviderConfig.custom(payload["protocol"], payload["base_url"], payload["model"]),
            payload.get("use_by_default") is True,
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def save_custom_provider(config: CustomProviderSettings | None, api_key: str = "") -> None:
    settings = QSettings(ORGANIZATION, APPLICATION)
    _write_custom_provider(settings, config)
    if config is None and settings.value(_ACTIVE_KEY, "", type=str) == CUSTOM_PROVIDER_NAME:
        settings.setValue(_ACTIVE_KEY, _DEFAULT_PROVIDER)
    _sync(settings)
    _remember_custom_key(config, api_key)


def _write_custom_provider(settings: QSettings, config: CustomProviderSettings | None) -> None:
    if config is None:
        settings.remove(_SETTINGS_KEY)
    else:
        provider = config.provider
        if provider.name != CUSTOM_PROVIDER_NAME:
            raise ValueError("请选择自定义 API。")
        settings.setValue(_SETTINGS_KEY, json.dumps({
            "protocol": provider.protocol,
            "base_url": provider.base_url,
            "model": provider.default_model,
            "use_by_default": config.use_by_default,
        }))


def _sync(settings: QSettings) -> None:
    settings.sync()
    if settings.status() != QSettings.Status.NoError:
        raise OSError("无法保存报告模型设置，请检查本地设置的写入权限。")


def _remember_custom_key(config: CustomProviderSettings | None, api_key: str) -> None:
    global _session_credential
    _session_credential = (config.provider, api_key.strip()) if config and api_key.strip() else None


def session_api_key(provider: ProviderConfig) -> str:
    credential = _session_credential
    if credential and credential[0] == provider:
        return credential[1]
    return ""


def custom_api_key_from_env() -> str:
    return report_key_from_env(CUSTOM_PROVIDER_NAME)


def resolve_custom_api_key(provider: ProviderConfig) -> str:
    saved = load_custom_provider()
    if saved and saved.provider == provider:
        return session_api_key(provider) or custom_api_key_from_env()
    return ""


def build_preset_model(name: str, address: str, model: str | None = None) -> ReportModelSettings:
    preset = PROVIDER_PRESETS[name]
    if model is not None:
        model = ProviderConfig.normalize_model_id(model)
    address = address.strip() or preset.base_url
    normalized = ProviderConfig.normalize_base_url(preset.protocol, address)
    return ReportModelSettings(replace(preset, base_url=normalized), model)


def load_report_model(name: str | None = None) -> ReportModelSettings | None:
    """Load a provider's saved values, or the configuration used by reports."""
    settings = QSettings(ORGANIZATION, APPLICATION)
    custom = load_custom_provider() if name in (None, CUSTOM_PROVIDER_NAME) else None
    if name is None:
        name = settings.value(_ACTIVE_KEY, "", type=str)
        if not name and custom is not None and custom.use_by_default:
            name = CUSTOM_PROVIDER_NAME
        if name not in PROVIDER_PRESETS and (name != CUSTOM_PROVIDER_NAME or custom is None):
            name = _DEFAULT_PROVIDER
    if name == CUSTOM_PROVIDER_NAME:
        return ReportModelSettings(custom.provider) if custom is not None else None
    raw = settings.value(f"{_PRESET_KEY}/{name}", "", type=str)
    if raw:
        try:
            payload = json.loads(raw)
            return build_preset_model(name, payload["base_url"], payload.get("model"))
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
    return ReportModelSettings(PROVIDER_PRESETS[name])


def save_report_model(config: ReportModelSettings, api_key: str = "") -> None:
    provider = config.provider
    key = api_key.strip()
    if key:
        provider.request_headers(key)
    settings = QSettings(ORGANIZATION, APPLICATION)
    if provider.is_custom:
        custom = CustomProviderSettings(provider, use_by_default=True)
        _write_custom_provider(settings, custom)
    else:
        settings.setValue(f"{_PRESET_KEY}/{provider.name}", json.dumps({
            "base_url": provider.base_url,
            "model": config.model,
        }))
    settings.setValue(_ACTIVE_KEY, provider.name)
    _sync(settings)
    if provider.is_custom:
        _remember_custom_key(custom, key)
    elif key:
        _preset_credentials[provider.name] = (provider.base_url, key)
    else:
        _preset_credentials.pop(provider.name, None)


def report_key_env_name(provider_name: str) -> str:
    return _KEY_ENV.get(provider_name, "")


def report_key_from_env(provider_name: str) -> str:
    name = report_key_env_name(provider_name)
    return os.getenv(name, "").strip() if name else ""


def report_session_key(provider: ProviderConfig) -> str:
    if provider.is_custom:
        return session_api_key(provider)
    credential = _preset_credentials.get(provider.name)
    if credential is not None and credential[0] == provider.base_url:
        return credential[1]
    return ""


def report_model_key(provider: ProviderConfig) -> str:
    if provider.is_custom:
        return resolve_custom_api_key(provider)
    return report_session_key(provider) or (
        "" if provider.has_overridden_preset_url else report_key_from_env(provider.name)
    )
