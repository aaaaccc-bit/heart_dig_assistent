"""Provider presets for OpenAI and Anthropic compatible chat APIs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from io import BufferedReader, RawIOBase
from time import monotonic
from urllib.parse import urlsplit
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, Request, build_opener

CUSTOM_PROVIDER_NAME = "自定义 API"
_PROTOCOL_ROUTES = {"openai": "/chat/completions", "anthropic": "/messages"}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    protocol: str
    base_url: str
    default_model: str
    models: tuple[str, ...]
    analysis_model: str | None = None

    @classmethod
    def custom(cls, protocol: str, base_url: str, model: str) -> ProviderConfig:
        address = cls.normalize_base_url(protocol, base_url)
        model = cls.normalize_model_id(model)
        return cls(CUSTOM_PROVIDER_NAME, protocol, address, model, (model,))

    @staticmethod
    def normalize_base_url(protocol: str, base_url: str) -> str:
        if protocol not in _PROTOCOL_ROUTES:
            raise ValueError("请选择支持的 API 协议。")
        address = base_url.strip().rstrip("/")
        invalid_url = "请输入有效的 HTTPS 地址，或本机 HTTP 地址。"
        if not address or any(char.isspace() or ord(char) < 32 for char in address) or "\\" in address:
            raise ValueError(invalid_url)
        try:
            parsed = urlsplit(address)
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError(invalid_url) from exc
        if (not host or port == 0 or parsed.username is not None or "?" in address
                or "#" in address or parsed.scheme not in {"https", "http"}):
            raise ValueError(invalid_url)
        if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
            raise ValueError("远程 API 请使用 HTTPS。本机接口可使用 HTTP。")
        route = _PROTOCOL_ROUTES[protocol]
        if parsed.path.endswith(route):
            address = address[:-len(route)].rstrip("/")
        elif any(parsed.path.endswith(other) for other in _PROTOCOL_ROUTES.values()):
            raise ValueError("地址与所选 API 协议不匹配。")
        return address

    @staticmethod
    def normalize_model_id(model: str) -> str:
        model = model.strip()
        if not model or any(char.isspace() or ord(char) < 32 for char in model):
            raise ValueError("请输入供应商提供的模型 ID。")
        return model

    @property
    def is_custom(self) -> bool:
        return self.name == CUSTOM_PROVIDER_NAME

    @property
    def has_overridden_preset_url(self) -> bool:
        preset = PROVIDER_PRESETS.get(self.name)
        return preset is not None and self.base_url.rstrip("/") != preset.base_url

    @property
    def uses_user_endpoint(self) -> bool:
        return self.is_custom or self.has_overridden_preset_url

    @property
    def requires_api_key(self) -> bool:
        return not self.is_custom or urlsplit(self.base_url).hostname not in _LOOPBACK_HOSTS

    @property
    def endpoint(self) -> str:
        if self.is_custom:
            config = self.custom(self.protocol, self.base_url, self.default_model)
            return config.base_url + _PROTOCOL_ROUTES[self.protocol]
        return self.base_url.rstrip("/") + _PROTOCOL_ROUTES[self.protocol]

    def request_headers(self, api_key: str) -> dict[str, str]:
        key = api_key.strip()
        if not key and self.requires_api_key:
            raise ValueError("请填写 API Key，或配置对应的环境变量。")
        if key and any(not 33 <= ord(char) <= 126 for char in key):
            raise ValueError("API Key 包含空格或无效字符。")
        headers = {"content-type": "application/json"}
        if self.protocol == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
            if key:
                headers["x-api-key"] = key
        elif key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def report_model(self, selected: str | None, *, advanced: bool) -> str:
        if selected:
            return selected
        if advanced and self.analysis_model and self.analysis_model in self.models:
            return self.analysis_model
        return self.default_model


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward credentials or report data to a different URL.


def open_custom_request(request: Request, *, timeout: float):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


class _CancellableBody(RawIOBase):
    """Check cancellation between short socket reads, retaining the real I/O timeout."""

    def __init__(self, sock, check_cancelled: Callable[[], None]):
        super().__init__()
        self._sock = sock
        # Keep the socket alive when urllib closes its connection reference.
        self._file_ref = sock.makefile("rb", buffering=0)
        self._check_cancelled = check_cancelled

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        timeout = self._sock.gettimeout()
        deadline = monotonic() + timeout if timeout is not None else None
        try:
            while True:
                self._check_cancelled()
                remaining = deadline - monotonic() if deadline is not None else 0.05
                if remaining <= 0:
                    raise TimeoutError("timed out")
                self._sock.settimeout(min(remaining, 0.05))
                try:
                    return self._sock.recv_into(buffer)
                except TimeoutError:
                    continue
        finally:
            self._sock.settimeout(timeout)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._file_ref.close()


def cancellable_request_opener(*, check_cancelled: Callable[[], None], no_redirect: bool):
    """Build a urllib opener that can cancel while receiving headers or body bytes."""

    class CancellableResponse(HTTPResponse):
        def __init__(self, sock, *args, **kwargs):
            super().__init__(sock, *args, **kwargs)
            original = self.fp
            self.fp = BufferedReader(_CancellableBody(sock, check_cancelled))
            original.close()

    class CancellableHTTPConnection(HTTPConnection):
        response_class = CancellableResponse

    class CancellableHTTPSConnection(HTTPSConnection):
        response_class = CancellableResponse

    class CancellableHTTPHandler(HTTPHandler):
        def http_open(self, req):
            return self.do_open(CancellableHTTPConnection, req)

    class CancellableHTTPSHandler(HTTPSHandler):
        def https_open(self, req):
            return self.do_open(CancellableHTTPSConnection, req, context=self._context)

    handlers = [CancellableHTTPHandler(), CancellableHTTPSHandler()]
    if no_redirect:
        handlers.append(_NoRedirect())
    return build_opener(*handlers).open


PROVIDER_PRESETS = {
    "OpenAI": ProviderConfig(
        "OpenAI", "openai", "https://api.openai.com/v1", "gpt-4o-mini", ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"),
        analysis_model="gpt-4.1",
    ),
    "Anthropic": ProviderConfig(
        "Anthropic", "anthropic", "https://api.anthropic.com/v1", "claude-3-5-sonnet-latest", ("claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"),
    ),
    "DeepSeek": ProviderConfig(
        "DeepSeek", "openai", "https://api.deepseek.com/v1", "deepseek-chat", ("deepseek-chat", "deepseek-reasoner"),
        analysis_model="deepseek-reasoner",
    ),
    "Kimi": ProviderConfig(
        "Kimi", "openai", "https://api.moonshot.cn/v1", "moonshot-v1-8k", ("moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"),
        analysis_model="moonshot-v1-32k",
    ),
    "GLM": ProviderConfig(
        "GLM", "openai", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", ("glm-4-flash", "glm-4-plus"),
        analysis_model="glm-4-plus",
    ),
    "豆包": ProviderConfig(
        "豆包", "openai", "https://ark.cn-beijing.volces.com/api/v3", "doubao-1-5-pro-32k", ("doubao-1-5-pro-32k", "doubao-1-5-lite-32k"),
    ),
}
