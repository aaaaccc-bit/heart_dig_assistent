from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from PySide6.QtCore import QSettings

from ai.provider_settings import (
    CustomProviderSettings, build_preset_model, load_custom_provider, load_report_model,
    report_model_key, resolve_custom_api_key, save_custom_provider, save_report_model,
)
from ai.providers import ProviderConfig
from ai.report_generator import AIReportError, AIReportService


class CustomProviderTest(unittest.TestCase):
    def test_normalizes_full_url_and_validates_credentials(self):
        local = ProviderConfig.custom("openai", "http://[::1]:8080/v1/chat/completions/", "org/model")
        self.assertEqual(local.base_url, "http://[::1]:8080/v1")
        self.assertEqual(local.endpoint, "http://[::1]:8080/v1/chat/completions")
        self.assertNotIn("Authorization", local.request_headers(""))
        cloud = ProviderConfig.custom("anthropic", "https://api.example.test/v1", "test")
        self.assertEqual(cloud.request_headers("key")["x-api-key"], "key")
        with self.assertRaises(ValueError):
            cloud.request_headers("")
        for url in ("http://api.example.test/v1", "https://user@api.example.test/v1", "https://api.example.test/v1?key=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ProviderConfig.custom("openai", url, "test")

    def test_configuration_persists_without_the_session_key(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "settings.ini"

            class TestSettings(QSettings):
                def __init__(self, *_args):
                    super().__init__(str(path), QSettings.Format.IniFormat)

            with patch("ai.provider_settings.QSettings", TestSettings), patch("ai.provider_settings._session_credential", None):
                provider = ProviderConfig.custom("openai", "https://api.example.test/v1", "test")
                saved = CustomProviderSettings(provider, use_by_default=True)
                save_custom_provider(saved, "secret-key")
                self.assertEqual(load_custom_provider(), saved)
                self.assertEqual(resolve_custom_api_key(provider), "secret-key")
                self.assertNotIn("secret-key", path.read_text(encoding="utf-8"))
                other = ProviderConfig.custom("openai", "https://other.example.test/v1", "test")
                self.assertEqual(resolve_custom_api_key(other), "")
                save_custom_provider(None)
                self.assertIsNone(load_custom_provider())

    def test_report_selection_preserves_legacy_custom_profile_and_scopes_preset_keys(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "settings.ini"

            class TestSettings(QSettings):
                def __init__(self, *_args):
                    super().__init__(str(path), QSettings.Format.IniFormat)

            with (patch("ai.provider_settings.QSettings", TestSettings),
                  patch("ai.provider_settings._session_credential", None),
                  patch("ai.provider_settings._preset_credentials", {}),
                  patch.dict("os.environ", {"DEEPSEEK_API_KEY": "official-key"})):
                custom = ProviderConfig.custom("openai", "https://example.test/v1", "test")
                save_custom_provider(CustomProviderSettings(custom, True))
                self.assertEqual(load_report_model().provider, custom)
                gateway = build_preset_model("DeepSeek", "https://gateway.test/v1/chat/completions", "deepseek-chat")
                save_report_model(gateway)
                self.assertEqual(load_report_model(), gateway)
                self.assertEqual(load_report_model(custom.name).provider, custom)
                self.assertEqual(report_model_key(gateway.provider), "")
                save_report_model(gateway, "gateway-key")
                self.assertEqual(report_model_key(gateway.provider), "gateway-key")
                self.assertEqual(report_model_key(build_preset_model("DeepSeek", "").provider), "official-key")
                self.assertNotIn("gateway-key", path.read_text(encoding="utf-8"))

    def test_model_list_uses_a_get_without_a_selected_model(self):
        base = "http://127.0.0.1:9000/gateway/v1"
        provider = ProviderConfig("自定义 API", "openai", base, "", ())
        data = {"data": [{"id": "org/report-model"}, {"id": "org/report-model"}, {"id": ""}, {}]}
        with patch("ai.report_generator.open_custom_request", return_value=BytesIO(json.dumps(data).encode())) as request:
            self.assertEqual(AIReportService.list_models(provider, ""), ("org/report-model",))
        sent = request.call_args.args[0]
        self.assertEqual(sent.get_method(), "GET")
        self.assertEqual(sent.full_url, base + "/models")
        self.assertIsNone(sent.data)
        self.assertNotIn("Authorization", sent.headers)
        selected = build_preset_model("DeepSeek", "", "org/report-model")
        self.assertEqual(selected.model, "org/report-model")

    def test_model_list_handles_anthropic_pagination(self):
        provider = ProviderConfig.custom("anthropic", "https://example.test/v1/messages", "unused")
        pages = [
            {"data": [{"id": "alpha/model"}], "has_more": True, "last_id": "alpha/model"},
            {"data": [{"id": "beta/model"}], "has_more": False},
        ]
        with patch("ai.report_generator.open_custom_request", side_effect=[BytesIO(json.dumps(page).encode()) for page in pages]) as request:
            self.assertEqual(AIReportService.list_models(provider, "test-key"), ("alpha/model", "beta/model"))
        first, second = (item.args[0] for item in request.call_args_list)
        self.assertEqual(first.full_url, "https://example.test/v1/models")
        self.assertEqual(second.full_url, first.full_url + "?after_id=alpha%2Fmodel")
        self.assertEqual(first.get_header("X-api-key"), "test-key")
        self.assertEqual(first.get_header("Anthropic-version"), "2023-06-01")

    def test_unsupported_model_list_prompts_manual_model_input(self):
        provider = ProviderConfig.custom("openai", "https://example.test/v1", "test")
        failure = HTTPError("https://example.test/v1/models", 404, "Not Found", {}, BytesIO())
        with patch("ai.report_generator.open_custom_request", side_effect=failure):
            with self.assertRaisesRegex(AIReportError, "手动输入模型 ID"):
                AIReportService.list_models(provider, "test-key")

    def test_probe_sends_only_a_short_prompt_and_validates_response(self):
        provider = ProviderConfig.custom("openai", "https://api.example.test/v1", "test")
        content = {"choices": [{"message": {"content": "OK"}}]}
        with patch("ai.report_generator.open_custom_request", return_value=BytesIO(json.dumps(content).encode())) as request:
            AIReportService.test_connection(provider, "secret-key")
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, provider.endpoint)
        self.assertEqual(json.loads(sent.data)["messages"], [{"role": "user", "content": "Reply with OK."}])
        with patch("ai.report_generator.open_custom_request", return_value=BytesIO(b'{"status": "ok"}')):
            with self.assertRaises(AIReportError):
                AIReportService.test_connection(provider, "secret-key")
