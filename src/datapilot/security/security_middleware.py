# -*- coding: utf-8 -*-
"""
安全中间件

提供以下安全功能:
1. HTTPS 重定向
2. HSTS (HTTP Strict Transport Security)
3. 安全响应头 (CSP, X-Frame-Options, etc.)
"""

from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..config.settings import get_settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全响应头中间件

    添加以下安全头:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    - Content-Security-Policy: 基本 CSP 策略
    - Strict-Transport-Security: HSTS (可选)
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if not self.settings.security_headers_enabled:
            return response

        # 基本安全头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # 权限策略
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), "
            "gyroscope=(), magnetometer=(), microphone=(), "
            "payment=(), usb=()"
        )

        # Content-Security-Policy (基本策略)
        # 生产环境应根据实际需求调整
        if self.settings.is_production:
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' wss: https:; "
                "frame-ancestors 'none';"
            )
            response.headers["Content-Security-Policy"] = csp

        # HSTS (仅在启用时添加)
        if self.settings.hsts_enabled:
            hsts_value = f"max-age={self.settings.hsts_max_age}; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value

        return response


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """
    HTTPS 重定向中间件

    将所有 HTTP 请求重定向到 HTTPS
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()

    async def dispatch(self, request: Request, call_next):
        if not self.settings.https_redirect:
            return await call_next(request)

        # 检查是否已经是 HTTPS
        # 支持代理服务器的 X-Forwarded-Proto 头
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        is_https = (
            request.url.scheme == "https" or
            forwarded_proto.lower() == "https"
        )

        if not is_https:
            # 构建 HTTPS URL
            https_url = request.url.replace(scheme="https")
            return RedirectResponse(
                url=str(https_url),
                status_code=301  # 永久重定向
            )

        return await call_next(request)


class TrustedHostMiddleware(BaseHTTPMiddleware):
    """
    可信主机中间件

    验证请求的 Host 头是否在允许列表中
    防止 Host 头注入攻击
    """

    def __init__(self, app: ASGIApp, allowed_hosts: list[str] = None):
        super().__init__(app)
        self.settings = get_settings()
        self.allowed_hosts = allowed_hosts or self._get_allowed_hosts()

    def _get_allowed_hosts(self) -> list[str]:
        """从配置获取允许的主机列表"""
        hosts = ["localhost", "127.0.0.1"]

        # 从 CORS 配置提取主机
        for origin in self.settings.cors_origins_list:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(origin)
                if parsed.hostname:
                    hosts.append(parsed.hostname)
            except Exception:
                pass

        return hosts

    async def dispatch(self, request: Request, call_next):
        # 开发环境跳过检查
        if self.settings.is_development:
            return await call_next(request)

        host = request.headers.get("host", "").split(":")[0]

        # 检查是否在允许列表中
        if host not in self.allowed_hosts and "*" not in self.allowed_hosts:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid host header"}
            )

        return await call_next(request)


def setup_security_middleware(app):
    """
    设置安全中间件

    Args:
        app: FastAPI 应用实例
    """
    settings = get_settings()

    # 安全响应头 (始终添加)
    if settings.security_headers_enabled:
        app.add_middleware(SecurityHeadersMiddleware)

    # HTTPS 重定向 (生产环境)
    if settings.https_redirect:
        app.add_middleware(HTTPSRedirectMiddleware)

    # 可信主机检查 (生产环境)
    if settings.is_production:
        app.add_middleware(TrustedHostMiddleware)


__all__ = [
    "SecurityHeadersMiddleware",
    "HTTPSRedirectMiddleware",
    "TrustedHostMiddleware",
    "setup_security_middleware",
]
