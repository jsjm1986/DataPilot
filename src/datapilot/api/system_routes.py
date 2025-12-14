# -*- coding: utf-8 -*-
"""
系统配置 API 路由

提供系统配置的查看、修改和连接测试功能
"""

from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config.manager import (
    get_config_manager,
    CONFIG_CATEGORIES,
)

router = APIRouter(prefix="/api/v1/system", tags=["System Configuration"])


# ============================================
# 请求/响应模型
# ============================================

class UpdateConfigRequest(BaseModel):
    """更新配置请求"""
    values: dict[str, Any] = Field(..., description="配置值")


class TestConnectionRequest(BaseModel):
    """测试连接请求"""
    type: str = Field(..., description="连接类型: database, llm, redis, vector")
    config: dict[str, Any] = Field(..., description="连接配置")


class ConfigResponse(BaseModel):
    """配置响应"""
    success: bool = True
    data: Any = None
    message: str = ""


# ============================================
# 路由定义
# ============================================

@router.get("/config")
async def get_all_config():
    """
    获取所有系统配置

    返回按分类组织的配置，敏感信息会被隐藏
    """
    manager = get_config_manager()

    return {
        "success": True,
        "data": {
            "config": manager.get_all_config(mask_sensitive=True),
            "categories": manager.get_categories(),
            "schema": manager.get_schema(),
        }
    }


@router.get("/config/schema")
async def get_config_schema():
    """
    获取配置 Schema

    返回所有配置字段的元数据定义
    """
    manager = get_config_manager()

    return {
        "success": True,
        "data": {
            "schema": manager.get_schema(),
            "categories": manager.get_categories(),
        }
    }


@router.get("/config/categories")
async def get_config_categories():
    """
    获取配置分类列表
    """
    return {
        "success": True,
        "data": CONFIG_CATEGORIES,
    }


@router.get("/config/{category}")
async def get_category_config(category: str):
    """
    获取指定分类的配置

    Args:
        category: 配置分类 (app, llm, embedding, database, vector, cache, security, api, judge)
    """
    if category not in CONFIG_CATEGORIES:
        raise HTTPException(status_code=404, detail=f"配置分类不存在: {category}")

    manager = get_config_manager()
    config = manager.get_category_config(category, mask_sensitive=True)

    # 获取该分类的 schema
    schema = [s for s in manager.get_schema() if s["category"] == category]

    return {
        "success": True,
        "data": {
            "category": category,
            "category_info": CONFIG_CATEGORIES[category],
            "config": config,
            "schema": schema,
        }
    }


@router.put("/config/{category}")
async def update_category_config(category: str, request: UpdateConfigRequest):
    """
    更新指定分类的配置

    Args:
        category: 配置分类
        request: 更新请求，包含配置值

    注意:
    - 部分配置支持热更新，立即生效
    - 部分配置需要重启服务才能生效
    - 配置会同时写入 .env 文件
    """
    if category not in CONFIG_CATEGORIES:
        raise HTTPException(status_code=404, detail=f"配置分类不存在: {category}")

    manager = get_config_manager()

    try:
        result = manager.update_category_config(category, request.values)
        return {
            "success": True,
            "data": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@router.post("/config/test-connection")
async def test_connection(request: TestConnectionRequest):
    """
    测试连接

    支持的连接类型:
    - database: 数据库连接 (MySQL, PostgreSQL, SQLite)
    - llm: LLM API 连接 (DeepSeek)
    - redis: Redis 连接
    - vector: 向量数据库连接 (LanceDB, Qdrant)
    """
    valid_types = ["database", "llm", "embedding", "rerank", "redis", "vector"]
    if request.type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"无效的连接类型: {request.type}，支持: {', '.join(valid_types)}"
        )

    manager = get_config_manager()
    result = await manager.test_connection(request.type, request.config)

    return {
        "success": result.get("success", False),
        "data": result,
    }


@router.get("/env/template")
async def get_env_template():
    """
    获取 .env 文件模板

    返回所有配置项的环境变量模板
    """
    manager = get_config_manager()
    schema = manager.get_schema()

    lines = [
        "# DataPilot 配置文件",
        "# 生成时间: 自动生成",
        "",
    ]

    current_category = None
    for field in schema:
        # 添加分类注释
        if field["category"] != current_category:
            current_category = field["category"]
            category_info = CONFIG_CATEGORIES.get(current_category, {})
            lines.append("")
            lines.append(f"# ============================================")
            lines.append(f"# {category_info.get('name', current_category)}")
            lines.append(f"# ============================================")

        # 添加字段注释和默认值
        env_name = field["name"].upper()
        default = field.get("default", "")
        description = field.get("description", "")
        required = "必填" if field.get("required") else "可选"
        hot_reload = "支持热更新" if field.get("hot_reload") else "需重启"

        lines.append(f"# {description} [{required}, {hot_reload}]")

        if field.get("sensitive"):
            lines.append(f"# {env_name}=your_secret_here")
        else:
            lines.append(f"{env_name}={default}")

    return {
        "success": True,
        "data": {
            "template": "\n".join(lines),
        }
    }


@router.get("/status")
async def get_system_status():
    """
    获取系统状态

    返回各组件的连接状态
    """
    manager = get_config_manager()

    # 获取当前配置
    config = manager.get_all_config(mask_sensitive=False)

    status = {
        "database": {"status": "unknown", "type": config.get("database", {}).get("default_db_type", "sqlite")},
        "llm": {"status": "unknown", "configured": bool(config.get("llm", {}).get("deepseek_api_key"))},
        "embedding": {"status": "unknown", "configured": bool(config.get("embedding", {}).get("embedding_api_key"))},
        "cache": {"status": "unknown", "url": config.get("cache", {}).get("redis_url", "")},
        "vector": {"status": "unknown", "backend": config.get("vector", {}).get("vector_backend", "auto")},
    }

    # 测试数据库连接
    db_config = config.get("database", {})
    db_config["db_type"] = db_config.get("default_db_type", "sqlite")
    db_result = await manager.test_connection("database", db_config)
    status["database"]["status"] = "connected" if db_result.get("success") else "disconnected"
    status["database"]["message"] = db_result.get("message") or db_result.get("error", "")

    return {
        "success": True,
        "data": status,
    }


__all__ = ["router"]
