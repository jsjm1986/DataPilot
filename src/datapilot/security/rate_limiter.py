# -*- coding: utf-8 -*-
"""
API 速率限制中间件

基于滑动窗口算法的速率限制实现:
- 支持按 IP 地址限制
- 支持按用户 ID 限制
- 支持按 API Key 限制
- 可配置不同端点的限制策略
"""

import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..config.settings import get_settings


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    enabled: bool = True


@dataclass
class RateLimitState:
    """速率限制状态"""
    minute_requests: list = field(default_factory=list)
    hour_requests: list = field(default_factory=list)
    last_cleanup: float = field(default_factory=time.time)


class RateLimiter:
    """
    滑动窗口速率限制器

    支持多种限制策略:
    - 每分钟请求数限制
    - 每小时请求数限制
    - 突发请求限制
    """

    def __init__(self):
        self._states: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()
        self._configs: dict[str, RateLimitConfig] = {}
        self._default_config = RateLimitConfig()
        self._load_configs()

    def _load_configs(self):
        """加载速率限制配置"""
        settings = get_settings()

        # 默认配置
        self._default_config = RateLimitConfig(
            requests_per_minute=getattr(settings, 'rate_limit_per_minute', 60),
            requests_per_hour=getattr(settings, 'rate_limit_per_hour', 1000),
            burst_limit=getattr(settings, 'rate_limit_burst', 10),
            enabled=getattr(settings, 'rate_limit_enabled', True),
        )

        # 端点特定配置
        self._configs = {
            # 查询端点 - 较严格的限制
            "/api/v1/query": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=500,
                burst_limit=5,
            ),
            # 执行端点 - 较严格的限制
            "/api/v1/execute": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=500,
                burst_limit=5,
            ),
            # 配置生成 - 非常严格的限制 (LLM 调用)
            "/api/v1/config/generate": RateLimitConfig(
                requests_per_minute=5,
                requests_per_hour=50,
                burst_limit=2,
            ),
            # 健康检查 - 宽松限制
            "/api/v1/health": RateLimitConfig(
                requests_per_minute=120,
                requests_per_hour=5000,
                burst_limit=20,
            ),
        }

    def get_config(self, path: str) -> RateLimitConfig:
        """获取端点的速率限制配置"""
        # 精确匹配
        if path in self._configs:
            return self._configs[path]

        # 前缀匹配
        for prefix, config in self._configs.items():
            if path.startswith(prefix):
                return config

        return self._default_config

    def _get_client_key(self, request: Request) -> str:
        """获取客户端标识键"""
        # 优先使用 API Key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key[:16]}"

        # 其次使用用户 ID (从认证中获取)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # 最后使用 IP 地址
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    async def check_rate_limit(self, request: Request) -> tuple[bool, Optional[dict]]:
        """
        检查请求是否超过速率限制

        Returns:
            (allowed, info): allowed 为 True 表示允许请求，info 包含限制信息
        """
        config = self.get_config(request.url.path)

        if not config.enabled:
            return True, None

        client_key = self._get_client_key(request)
        current_time = time.time()

        async with self._lock:
            state = self._states[client_key]

            # 清理过期记录
            minute_ago = current_time - 60
            hour_ago = current_time - 3600

            state.minute_requests = [t for t in state.minute_requests if t > minute_ago]
            state.hour_requests = [t for t in state.hour_requests if t > hour_ago]

            # 检查限制
            minute_count = len(state.minute_requests)
            hour_count = len(state.hour_requests)

            # 检查突发限制 (最近 1 秒内的请求)
            second_ago = current_time - 1
            burst_count = len([t for t in state.minute_requests if t > second_ago])

            info = {
                "limit_minute": config.requests_per_minute,
                "limit_hour": config.requests_per_hour,
                "limit_burst": config.burst_limit,
                "remaining_minute": max(0, config.requests_per_minute - minute_count),
                "remaining_hour": max(0, config.requests_per_hour - hour_count),
                "reset_minute": int(minute_ago + 60),
                "reset_hour": int(hour_ago + 3600),
            }

            # 检查是否超限
            if burst_count >= config.burst_limit:
                info["exceeded"] = "burst"
                info["retry_after"] = 1
                return False, info

            if minute_count >= config.requests_per_minute:
                info["exceeded"] = "minute"
                info["retry_after"] = int(60 - (current_time - state.minute_requests[0]))
                return False, info

            if hour_count >= config.requests_per_hour:
                info["exceeded"] = "hour"
                info["retry_after"] = int(3600 - (current_time - state.hour_requests[0]))
                return False, info

            # 记录请求
            state.minute_requests.append(current_time)
            state.hour_requests.append(current_time)

            return True, info

    async def cleanup_expired(self):
        """清理过期的状态记录"""
        current_time = time.time()
        hour_ago = current_time - 3600

        async with self._lock:
            expired_keys = []
            for key, state in self._states.items():
                if not state.hour_requests or state.hour_requests[-1] < hour_ago:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._states[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI 速率限制中间件"""

    def __init__(self, app, limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.limiter = limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        # 跳过 WebSocket 请求
        if request.url.path.startswith("/ws"):
            return await call_next(request)

        # 跳过静态文件和文档
        skip_paths = ["/docs", "/redoc", "/openapi.json", "/favicon.ico"]
        if any(request.url.path.startswith(p) for p in skip_paths):
            return await call_next(request)

        # 检查速率限制
        allowed, info = await self.limiter.check_rate_limit(request)

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "error": "too_many_requests",
                    "exceeded": info.get("exceeded"),
                    "retry_after": info.get("retry_after"),
                },
                headers={
                    "Retry-After": str(info.get("retry_after", 60)),
                    "X-RateLimit-Limit-Minute": str(info.get("limit_minute")),
                    "X-RateLimit-Remaining-Minute": str(info.get("remaining_minute")),
                    "X-RateLimit-Reset-Minute": str(info.get("reset_minute")),
                },
            )

        # 执行请求
        response = await call_next(request)

        # 添加速率限制头
        if info:
            response.headers["X-RateLimit-Limit-Minute"] = str(info.get("limit_minute"))
            response.headers["X-RateLimit-Remaining-Minute"] = str(info.get("remaining_minute"))
            response.headers["X-RateLimit-Reset-Minute"] = str(info.get("reset_minute"))

        return response


# 全局速率限制器实例
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取速率限制器实例"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limit(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    burst_limit: int = 10,
):
    """
    速率限制装饰器

    用于单独限制特定端点:

    @app.get("/api/expensive")
    @rate_limit(requests_per_minute=10)
    async def expensive_endpoint():
        ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            limiter = get_rate_limiter()

            # 临时覆盖配置
            original_config = limiter.get_config(request.url.path)
            temp_config = RateLimitConfig(
                requests_per_minute=requests_per_minute,
                requests_per_hour=requests_per_hour,
                burst_limit=burst_limit,
            )
            limiter._configs[request.url.path] = temp_config

            try:
                allowed, info = await limiter.check_rate_limit(request)
                if not allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "Rate limit exceeded",
                            "exceeded": info.get("exceeded"),
                            "retry_after": info.get("retry_after"),
                        },
                        headers={"Retry-After": str(info.get("retry_after", 60))},
                    )
                return await func(request, *args, **kwargs)
            finally:
                # 恢复原配置
                if original_config:
                    limiter._configs[request.url.path] = original_config

        return wrapper
    return decorator


__all__ = [
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitMiddleware",
    "get_rate_limiter",
    "rate_limit",
]
