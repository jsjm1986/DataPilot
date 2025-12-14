"""
DataPilot 应用入口
FastAPI 应用主文件
"""

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .api.routes import router
from .api.config_routes import router as config_router
from .api.system_routes import router as system_router
from .api.agents_routes import router as agents_router
from .api.dspy_routes import router as dspy_router
from .api.cache_routes import router as cache_router
from .api.audit_routes import router as audit_router
from .api.cost_routes import router as cost_router
from .api.websocket import websocket_endpoint
from .config.settings import get_settings
from .observability.metrics import setup_metrics
from .security.rate_limiter import RateLimitMiddleware
from .security.security_middleware import setup_security_middleware

# 配置日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()

    # 启动时
    logger.info(
        "DataPilot 启动中...",
        env=settings.datapilot_env,
        debug=settings.datapilot_debug,
    )

    # 初始化审计存储 (启动后台刷新任务)
    from .observability.audit import init_audit_store, shutdown_audit_store
    await init_audit_store()
    logger.info("审计存储已初始化")

    yield

    # 关闭时
    logger.info("DataPilot 关闭中...")

    # 关闭审计存储 (刷新剩余数据)
    await shutdown_audit_store()
    logger.info("审计存储已关闭")

    # 清理数据库连接
    from .db.connector import get_db_manager
    db_manager = get_db_manager()
    await db_manager.close_all()

    logger.info("DataPilot 已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = get_settings()

    app = FastAPI(
        title="DataPilot",
        description="企业级 Agentic BI 平台 - 自然语言转 SQL",
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # CORS 中间件 (严格模式配置)
    if settings.cors_strict_mode and settings.is_production:
        # 生产环境严格 CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
            max_age=settings.cors_max_age,
        )
    else:
        # 开发环境宽松 CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 安全中间件 (HTTPS 重定向、安全响应头)
    setup_security_middleware(app)

    # 速率限制中间件 (生产环境启用)
    if not settings.is_development or getattr(settings, 'rate_limit_enabled', False):
        app.add_middleware(RateLimitMiddleware)
        logger.info("Rate limiting middleware enabled")

    # 注册路由
    setup_metrics(app)
    app.include_router(router)
    app.include_router(config_router)  # 元数据配置路由
    app.include_router(system_router)  # 系统配置路由
    app.include_router(agents_router)  # Agents 监控路由
    app.include_router(dspy_router)    # DSPy 训练路由
    app.include_router(cache_router)   # 缓存管理路由
    app.include_router(audit_router)   # 审计日志路由
    app.include_router(cost_router)    # 成本分析路由

    # WebSocket 端点
    @app.websocket("/ws/{client_id}")
    async def websocket_route(websocket: WebSocket, client_id: str):
        await websocket_endpoint(websocket, client_id)

    # 根路由
    @app.get("/")
    async def root():
        return {
            "name": "DataPilot",
            "version": "0.1.0",
            "description": "企业级 Agentic BI 平台",
            "docs": "/docs",
        }

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.datapilot.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=1 if settings.api_reload else settings.api_workers,
    )
