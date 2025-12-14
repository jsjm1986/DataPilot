# -*- coding: utf-8 -*-
"""
配置管理器

提供配置的读取、验证、更新和热重载功能
"""

import os
import re
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache

from .settings import Settings, get_settings, clear_settings_cache


@dataclass
class ConfigField:
    """配置字段定义"""
    name: str
    label: str
    type: str  # input, password, number, select, switch, slider, textarea
    category: str
    default: Any = None
    required: bool = False
    hot_reload: bool = False  # 是否支持热更新
    options: list = field(default_factory=list)  # select 类型的选项
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    description: str = ""
    sensitive: bool = False  # 是否敏感信息


# 配置字段元数据定义
CONFIG_SCHEMA: list[ConfigField] = [
    # ============================================
    # 应用配置
    # ============================================
    ConfigField(
        name="datapilot_env",
        label="运行环境",
        type="select",
        category="app",
        default="development",
        options=["development", "staging", "production"],
        description="应用运行环境",
    ),
    ConfigField(
        name="datapilot_debug",
        label="调试模式",
        type="switch",
        category="app",
        default=True,
        description="启用调试模式",
    ),
    ConfigField(
        name="datapilot_log_level",
        label="日志级别",
        type="select",
        category="app",
        default="INFO",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
        description="日志输出级别",
    ),

    # ============================================
    # LLM 配置
    # ============================================
    ConfigField(
        name="deepseek_api_key",
        label="LLM API Key",
        type="password",
        category="llm",
        required=True,
        hot_reload=True,
        sensitive=True,
        description="大语言模型 API 密钥",
    ),
    ConfigField(
        name="deepseek_base_url",
        label="LLM Base URL",
        type="input",
        category="llm",
        default="https://api.deepseek.com/v1",
        hot_reload=True,
        description="LLM 服务地址，支持私有部署",
    ),
    ConfigField(
        name="deepseek_model",
        label="模型名称",
        type="input",
        category="llm",
        default="deepseek-chat",
        hot_reload=True,
        description="使用的模型名称",
    ),
    ConfigField(
        name="deepseek_temperature",
        label="Temperature",
        type="slider",
        category="llm",
        default=0.1,
        min_value=0,
        max_value=2,
        step=0.1,
        hot_reload=True,
        description="生成温度参数 (0-2)",
    ),
    ConfigField(
        name="deepseek_max_tokens",
        label="Max Tokens",
        type="number",
        category="llm",
        default=4096,
        min_value=100,
        max_value=32000,
        hot_reload=True,
        description="最大生成 token 数",
    ),

    # ============================================
    # Embedding 配置
    # ============================================
    ConfigField(
        name="embedding_api_key",
        label="Embedding API Key",
        type="password",
        category="embedding",
        hot_reload=True,
        sensitive=True,
        description="SiliconFlow Embedding API 密钥",
    ),
    ConfigField(
        name="embedding_base_url",
        label="Embedding Base URL",
        type="input",
        category="embedding",
        default="https://api.siliconflow.cn/v1",
        hot_reload=True,
        description="Embedding API 基础 URL",
    ),
    ConfigField(
        name="embedding_model",
        label="Embedding 模型",
        type="input",
        category="embedding",
        default="BAAI/bge-large-zh-v1.5",
        hot_reload=True,
        description="Embedding 模型名称",
    ),

    # ============================================
    # Rerank 配置
    # ============================================
    ConfigField(
        name="rerank_api_key",
        label="Rerank API Key",
        type="password",
        category="embedding",
        hot_reload=True,
        sensitive=True,
        description="SiliconFlow Rerank API 密钥",
    ),
    ConfigField(
        name="rerank_base_url",
        label="Rerank Base URL",
        type="input",
        category="embedding",
        default="https://api.siliconflow.cn/v1",
        hot_reload=True,
        description="Rerank API 基础 URL",
    ),
    ConfigField(
        name="rerank_model",
        label="Rerank 模型",
        type="input",
        category="embedding",
        default="Qwen/Qwen3-Reranker-8B",
        hot_reload=True,
        description="Rerank 模型名称",
    ),

    # ============================================
    # MySQL 配置
    # ============================================
    ConfigField(
        name="mysql_host",
        label="MySQL 主机",
        type="input",
        category="database",
        default="localhost",
        description="MySQL 服务器地址",
    ),
    ConfigField(
        name="mysql_port",
        label="MySQL 端口",
        type="number",
        category="database",
        default=3306,
        min_value=1,
        max_value=65535,
        description="MySQL 服务器端口",
    ),
    ConfigField(
        name="mysql_user",
        label="MySQL 用户名",
        type="input",
        category="database",
        default="datapilot",
        description="MySQL 连接用户名",
    ),
    ConfigField(
        name="mysql_password",
        label="MySQL 密码",
        type="password",
        category="database",
        sensitive=True,
        description="MySQL 连接密码",
    ),
    ConfigField(
        name="mysql_database",
        label="MySQL 数据库",
        type="input",
        category="database",
        default="ecommerce",
        description="MySQL 数据库名称",
    ),

    # ============================================
    # PostgreSQL 配置
    # ============================================
    ConfigField(
        name="postgres_host",
        label="PostgreSQL 主机",
        type="input",
        category="database",
        default="localhost",
        description="PostgreSQL 服务器地址",
    ),
    ConfigField(
        name="postgres_port",
        label="PostgreSQL 端口",
        type="number",
        category="database",
        default=5432,
        min_value=1,
        max_value=65535,
        description="PostgreSQL 服务器端口",
    ),
    ConfigField(
        name="postgres_user",
        label="PostgreSQL 用户名",
        type="input",
        category="database",
        default="datapilot",
        description="PostgreSQL 连接用户名",
    ),
    ConfigField(
        name="postgres_password",
        label="PostgreSQL 密码",
        type="password",
        category="database",
        sensitive=True,
        description="PostgreSQL 连接密码",
    ),
    ConfigField(
        name="postgres_database",
        label="PostgreSQL 数据库",
        type="input",
        category="database",
        default="sales",
        description="PostgreSQL 数据库名称",
    ),

    # ============================================
    # SQLite 配置
    # ============================================
    ConfigField(
        name="sqlite_database",
        label="SQLite 数据库路径",
        type="input",
        category="database",
        default="data/datapilot.db",
        description="SQLite 数据库文件路径",
    ),

    # ============================================
    # SQL Server 配置
    # ============================================
    ConfigField(
        name="sqlserver_host",
        label="SQL Server 主机",
        type="input",
        category="database",
        default="localhost",
        description="SQL Server 服务器地址",
    ),
    ConfigField(
        name="sqlserver_port",
        label="SQL Server 端口",
        type="number",
        category="database",
        default=1433,
        min_value=1,
        max_value=65535,
        description="SQL Server 服务器端口",
    ),
    ConfigField(
        name="sqlserver_user",
        label="SQL Server 用户名",
        type="input",
        category="database",
        default="sa",
        description="SQL Server 连接用户名",
    ),
    ConfigField(
        name="sqlserver_password",
        label="SQL Server 密码",
        type="password",
        category="database",
        sensitive=True,
        description="SQL Server 连接密码",
    ),
    ConfigField(
        name="sqlserver_database",
        label="SQL Server 数据库",
        type="input",
        category="database",
        default="master",
        description="SQL Server 数据库名称",
    ),
    ConfigField(
        name="sqlserver_driver",
        label="SQL Server ODBC 驱动",
        type="input",
        category="database",
        default="ODBC Driver 17 for SQL Server",
        description="SQL Server ODBC 驱动名称",
    ),

    # ============================================
    # ClickHouse 配置
    # ============================================
    ConfigField(
        name="clickhouse_host",
        label="ClickHouse 主机",
        type="input",
        category="database",
        default="localhost",
        description="ClickHouse 服务器地址",
    ),
    ConfigField(
        name="clickhouse_port",
        label="ClickHouse HTTP 端口",
        type="number",
        category="database",
        default=8123,
        min_value=1,
        max_value=65535,
        description="ClickHouse HTTP 接口端口",
    ),
    ConfigField(
        name="clickhouse_user",
        label="ClickHouse 用户名",
        type="input",
        category="database",
        default="default",
        description="ClickHouse 连接用户名",
    ),
    ConfigField(
        name="clickhouse_password",
        label="ClickHouse 密码",
        type="password",
        category="database",
        sensitive=True,
        description="ClickHouse 连接密码",
    ),
    ConfigField(
        name="clickhouse_database",
        label="ClickHouse 数据库",
        type="input",
        category="database",
        default="default",
        description="ClickHouse 数据库名称",
    ),

    # ============================================
    # DuckDB 配置
    # ============================================
    ConfigField(
        name="duckdb_database",
        label="DuckDB 数据库路径",
        type="input",
        category="database",
        default="data/datapilot.duckdb",
        description="DuckDB 数据库文件路径",
    ),

    ConfigField(
        name="default_db_type",
        label="默认数据库类型",
        type="select",
        category="database",
        default="sqlite",
        options=["mysql", "postgresql", "sqlite", "sqlserver", "clickhouse", "duckdb"],
        description="默认使用的数据库类型",
    ),

    # ============================================
    # 向量数据库配置
    # ============================================
    ConfigField(
        name="vector_backend",
        label="向量数据库后端",
        type="select",
        category="vector",
        default="auto",
        options=["lancedb", "qdrant", "auto"],
        description="向量数据库后端选择",
    ),
    ConfigField(
        name="lancedb_path",
        label="LanceDB 路径",
        type="input",
        category="vector",
        default="data/lancedb",
        description="LanceDB 数据存储路径",
    ),
    ConfigField(
        name="qdrant_url",
        label="Qdrant URL",
        type="input",
        category="vector",
        default="http://localhost:6333",
        description="Qdrant 服务地址",
    ),
    ConfigField(
        name="qdrant_api_key",
        label="Qdrant API Key",
        type="password",
        category="vector",
        sensitive=True,
        description="Qdrant API 密钥",
    ),
    ConfigField(
        name="qdrant_hnsw_m",
        label="HNSW M 参数",
        type="number",
        category="vector",
        default=16,
        min_value=4,
        max_value=64,
        description="HNSW 索引参数，越大越精确但越慢",
    ),
    ConfigField(
        name="qdrant_ef_search",
        label="EF Search 参数",
        type="number",
        category="vector",
        default=128,
        min_value=16,
        max_value=512,
        description="搜索参数，越大越精确但越慢",
    ),

    # ============================================
    # Redis 配置
    # ============================================
    ConfigField(
        name="redis_url",
        label="Redis URL",
        type="input",
        category="cache",
        default="redis://localhost:6379",
        description="Redis 连接 URL",
    ),
    ConfigField(
        name="redis_password",
        label="Redis 密码",
        type="password",
        category="cache",
        sensitive=True,
        description="Redis 连接密码",
    ),

    # ============================================
    # 语义缓存配置
    # ============================================
    ConfigField(
        name="cache_similarity_threshold",
        label="缓存相似度阈值",
        type="slider",
        category="cache",
        default=0.85,
        min_value=0.5,
        max_value=1.0,
        step=0.05,
        hot_reload=True,
        description="语义缓存相似度阈值 (0.5-1.0)",
    ),
    ConfigField(
        name="cache_ttl_minutes",
        label="缓存过期时间(分钟)",
        type="number",
        category="cache",
        default=30,
        min_value=1,
        max_value=1440,
        hot_reload=True,
        description="缓存过期时间",
    ),
    ConfigField(
        name="cache_eviction",
        label="缓存淘汰策略",
        type="select",
        category="cache",
        default="lru",
        options=["lru", "lfu", "fifo"],
        hot_reload=True,
        description="缓存淘汰策略",
    ),

    # ============================================
    # 安全配置
    # ============================================
    ConfigField(
        name="cors_origins",
        label="CORS 允许源",
        type="textarea",
        category="security",
        default="http://localhost:5173,http://localhost:3000",
        hot_reload=True,
        description="允许的跨域源，多个用逗号分隔",
    ),
    ConfigField(
        name="cors_strict_mode",
        label="CORS 严格模式",
        type="switch",
        category="security",
        default=False,
        hot_reload=True,
        description="生产环境启用严格 CORS",
    ),
    ConfigField(
        name="jwt_secret_key",
        label="JWT 密钥",
        type="password",
        category="security",
        sensitive=True,
        description="JWT 签名密钥",
    ),
    ConfigField(
        name="admin_api_key",
        label="管理员 API Key",
        type="password",
        category="security",
        sensitive=True,
        hot_reload=True,
        description="管理员 API 访问密钥",
    ),
    ConfigField(
        name="user_api_key",
        label="用户 API Key",
        type="password",
        category="security",
        sensitive=True,
        hot_reload=True,
        description="用户 API 访问密钥",
    ),
    ConfigField(
        name="cors_allow_credentials",
        label="CORS 允许凭证",
        type="switch",
        category="security",
        default=True,
        hot_reload=True,
        description="是否允许携带 Cookie",
    ),
    ConfigField(
        name="cors_max_age",
        label="预检请求缓存时间",
        type="number",
        category="security",
        default=600,
        min_value=0,
        max_value=86400,
        hot_reload=True,
        description="CORS 预检请求缓存时间(秒)",
    ),
    ConfigField(
        name="https_redirect",
        label="HTTPS 强制重定向",
        type="switch",
        category="security",
        default=False,
        description="将 HTTP 请求重定向到 HTTPS",
    ),
    ConfigField(
        name="hsts_enabled",
        label="启用 HSTS",
        type="switch",
        category="security",
        default=False,
        description="HTTP 严格传输安全",
    ),
    ConfigField(
        name="hsts_max_age",
        label="HSTS 最大年龄",
        type="number",
        category="security",
        default=31536000,
        min_value=0,
        max_value=63072000,
        description="HSTS 最大年龄(秒)，默认 1 年",
    ),
    ConfigField(
        name="security_headers_enabled",
        label="启用安全响应头",
        type="switch",
        category="security",
        default=True,
        description="添加 X-Frame-Options 等安全头",
    ),
    ConfigField(
        name="allow_mock_users",
        label="允许模拟用户",
        type="switch",
        category="security",
        default=False,
        description="开发环境允许模拟用户登录",
    ),

    # ============================================
    # API 服务配置
    # ============================================
    ConfigField(
        name="api_host",
        label="API 监听地址",
        type="input",
        category="api",
        default="0.0.0.0",
        description="API 服务监听地址",
    ),
    ConfigField(
        name="api_port",
        label="API 端口",
        type="number",
        category="api",
        default=8000,
        min_value=1,
        max_value=65535,
        description="API 服务端口",
    ),
    ConfigField(
        name="api_workers",
        label="工作进程数",
        type="number",
        category="api",
        default=4,
        min_value=1,
        max_value=32,
        description="API 工作进程数",
    ),
    ConfigField(
        name="api_reload",
        label="自动重载",
        type="switch",
        category="api",
        default=True,
        description="开发模式下代码修改自动重启",
    ),
    ConfigField(
        name="frontend_url",
        label="前端应用地址",
        type="input",
        category="api",
        default="http://localhost:5173",
        hot_reload=True,
        description="前端应用地址，用于 CORS 配置",
    ),
    ConfigField(
        name="langsmith_api_key",
        label="LangSmith API Key",
        type="password",
        category="api",
        sensitive=True,
        hot_reload=True,
        description="LangSmith 追踪服务密钥（可选）",
    ),
    ConfigField(
        name="langsmith_project",
        label="LangSmith 项目名",
        type="input",
        category="api",
        default="datapilot",
        hot_reload=True,
        description="LangSmith 追踪数据归属的项目",
    ),
    ConfigField(
        name="langsmith_endpoint",
        label="LangSmith 端点",
        type="input",
        category="api",
        default="https://api.smith.langchain.com",
        hot_reload=True,
        description="LangSmith API 地址",
    ),

    # ============================================
    # MCP 配置
    # ============================================
    ConfigField(
        name="mcp_timeout_seconds",
        label="MCP 超时时间",
        type="number",
        category="mcp",
        default=5,
        min_value=1,
        max_value=60,
        hot_reload=True,
        description="MCP 工具调用最长等待时间(秒)",
    ),
    ConfigField(
        name="mcp_retries",
        label="MCP 重试次数",
        type="number",
        category="mcp",
        default=2,
        min_value=0,
        max_value=10,
        hot_reload=True,
        description="MCP 工具调用失败后重试次数",
    ),
    ConfigField(
        name="mcp_row_limit",
        label="MCP 行数限制",
        type="number",
        category="mcp",
        default=1000,
        min_value=10,
        max_value=10000,
        hot_reload=True,
        description="单次查询最多返回多少行",
    ),
    ConfigField(
        name="mcp_page_limit",
        label="MCP 分页限制",
        type="number",
        category="mcp",
        default=200,
        min_value=10,
        max_value=1000,
        hot_reload=True,
        description="分页时每页最多多少行",
    ),

    # ============================================
    # 沙箱配置
    # ============================================
    ConfigField(
        name="e2b_api_key",
        label="E2B API Key",
        type="password",
        category="sandbox",
        sensitive=True,
        hot_reload=True,
        description="E2B 云端沙箱服务密钥",
    ),
    ConfigField(
        name="sandbox_timeout_seconds",
        label="沙箱执行超时",
        type="number",
        category="sandbox",
        default=10,
        min_value=5,
        max_value=120,
        hot_reload=True,
        description="代码最长执行时间(秒)",
    ),

    # ============================================
    # 可观测性配置
    # ============================================
    ConfigField(
        name="audit_retention_days",
        label="审计日志保留天数",
        type="number",
        category="observability",
        default=90,
        min_value=7,
        max_value=365,
        description="查询历史保留多久",
    ),
    ConfigField(
        name="prometheus_port",
        label="Prometheus 端口",
        type="number",
        category="observability",
        default=9090,
        min_value=1,
        max_value=65535,
        description="Prometheus 指标暴露端口",
    ),
]

# 配置分类定义
CONFIG_CATEGORIES = {
    "app": {"name": "应用配置", "icon": "setting", "order": 1},
    "llm": {"name": "LLM 配置", "icon": "robot", "order": 2},
    "embedding": {"name": "Embedding 配置", "icon": "api", "order": 3},
    "database": {"name": "数据库配置", "icon": "database", "order": 4},
    "vector": {"name": "向量数据库", "icon": "cluster", "order": 5},
    "cache": {"name": "缓存配置", "icon": "thunderbolt", "order": 6},
    "security": {"name": "安全配置", "icon": "safety", "order": 7},
    "api": {"name": "API 配置", "icon": "cloud", "order": 8},
    "mcp": {"name": "MCP 配置", "icon": "tool", "order": 9},
    "sandbox": {"name": "沙箱配置", "icon": "code", "order": 10},
    "observability": {"name": "可观测性", "icon": "monitor", "order": 11},
}


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self._settings: Optional[Settings] = None
        self._env_path = Path(".env")
        self._runtime_overrides: dict[str, Any] = {}

    @property
    def settings(self) -> Settings:
        """获取当前配置"""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def get_schema(self) -> list[dict]:
        """获取配置 Schema"""
        return [
            {
                "name": f.name,
                "label": f.label,
                "type": f.type,
                "category": f.category,
                "default": f.default,
                "required": f.required,
                "hot_reload": f.hot_reload,
                "options": f.options,
                "min_value": f.min_value,
                "max_value": f.max_value,
                "step": f.step,
                "description": f.description,
                "sensitive": f.sensitive,
            }
            for f in CONFIG_SCHEMA
        ]

    def get_categories(self) -> dict:
        """获取配置分类"""
        return CONFIG_CATEGORIES

    def get_all_config(self, mask_sensitive: bool = True) -> dict:
        """
        获取所有配置

        Args:
            mask_sensitive: 是否隐藏敏感信息

        Returns:
            配置字典，按分类组织
        """
        result = {}

        for category in CONFIG_CATEGORIES:
            result[category] = self.get_category_config(category, mask_sensitive)

        return result

    def get_category_config(self, category: str, mask_sensitive: bool = True) -> dict:
        """
        获取指定分类的配置

        Args:
            category: 配置分类
            mask_sensitive: 是否隐藏敏感信息

        Returns:
            配置字典
        """
        config = {}
        fields = [f for f in CONFIG_SCHEMA if f.category == category]

        for field in fields:
            # 优先使用运行时覆盖值
            if field.name in self._runtime_overrides:
                value = self._runtime_overrides[field.name]
            else:
                value = getattr(self.settings, field.name, field.default)

            # 隐藏敏感信息
            if mask_sensitive and field.sensitive and value:
                if isinstance(value, str) and len(value) > 4:
                    value = value[:2] + "*" * (len(value) - 4) + value[-2:]
                else:
                    value = "******"

            config[field.name] = value

        return config

    def update_category_config(self, category: str, values: dict) -> dict:
        """
        更新指定分类的配置

        Args:
            category: 配置分类
            values: 新配置值

        Returns:
            更新结果
        """
        fields = {f.name: f for f in CONFIG_SCHEMA if f.category == category}
        updated = []
        need_restart = []

        for name, value in values.items():
            if name not in fields:
                continue

            field = fields[name]

            # 验证值
            if not self._validate_value(field, value):
                raise ValueError(f"配置项 {name} 的值无效")

            # 检查是否支持热更新
            if field.hot_reload:
                self._runtime_overrides[name] = value
                updated.append(name)
            else:
                need_restart.append(name)

        # 更新 .env 文件
        self._update_env_file(values)

        # 清除配置缓存并重新加载
        clear_settings_cache()
        self._settings = get_settings()

        return {
            "success": True,
            "updated": updated,
            "need_restart": need_restart,
            "message": self._build_update_message(updated, need_restart),
        }

    def _validate_value(self, field: ConfigField, value: Any) -> bool:
        """验证配置值"""
        if field.required and not value:
            return False

        if field.type == "number":
            try:
                num_value = float(value)
                if field.min_value is not None and num_value < field.min_value:
                    return False
                if field.max_value is not None and num_value > field.max_value:
                    return False
            except (TypeError, ValueError):
                return False

        if field.type == "select" and field.options:
            if value not in field.options:
                return False

        return True

    def _update_env_file(self, values: dict):
        """更新 .env 文件"""
        # 读取现有内容
        env_content = ""
        if self._env_path.exists():
            env_content = self._env_path.read_text(encoding="utf-8")

        # 更新或添加配置
        for name, value in values.items():
            env_name = name.upper()
            pattern = rf"^{env_name}=.*$"

            if value is None:
                value = ""
            elif isinstance(value, bool):
                value = str(value).lower()
            else:
                value = str(value)

            # 如果值包含特殊字符，添加引号
            if " " in value or "," in value:
                value = f'"{value}"'

            new_line = f"{env_name}={value}"

            if re.search(pattern, env_content, re.MULTILINE):
                env_content = re.sub(pattern, new_line, env_content, flags=re.MULTILINE)
            else:
                env_content = env_content.rstrip() + f"\n{new_line}\n"

        # 写入文件
        self._env_path.write_text(env_content, encoding="utf-8")

    def _build_update_message(self, updated: list, need_restart: list) -> str:
        """构建更新消息"""
        messages = []

        if updated:
            messages.append(f"已热更新: {', '.join(updated)}")

        if need_restart:
            messages.append(f"需要重启生效: {', '.join(need_restart)}")

        return "; ".join(messages) if messages else "无配置更新"

    async def test_connection(self, conn_type: str, config: dict) -> dict:
        """
        测试连接

        Args:
            conn_type: 连接类型 (database, llm, redis, vector)
            config: 连接配置

        Returns:
            测试结果
        """
        try:
            if conn_type == "database":
                return await self._test_database_connection(config)
            elif conn_type == "llm":
                return await self._test_llm_connection(config)
            elif conn_type == "embedding":
                return await self._test_embedding_connection(config)
            elif conn_type == "rerank":
                return await self._test_rerank_connection(config)
            elif conn_type == "redis":
                return await self._test_redis_connection(config)
            elif conn_type == "vector":
                return await self._test_vector_connection(config)
            else:
                return {"success": False, "error": f"未知连接类型: {conn_type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _is_masked_value(self, value: str) -> bool:
        """检查值是否是脱敏后的值（包含星号）"""
        return bool(value and '*' in value)

    def _get_db_config_value(self, config: dict, key: str, default: any = "") -> any:
        """获取数据库配置值，如果是脱敏值则使用已保存的配置"""
        value = config.get(key, "")
        if isinstance(value, str) and self._is_masked_value(value):
            return getattr(self.settings, key, default)
        return value if value else getattr(self.settings, key, default)

    async def _test_database_connection(self, config: dict) -> dict:
        """测试数据库连接"""
        db_type = config.get("db_type") or getattr(self.settings, "default_db_type", "sqlite")

        try:
            if db_type == "sqlite":
                import aiosqlite
                db_path = self._get_db_config_value(config, "sqlite_database", "data/datapilot.db")
                async with aiosqlite.connect(db_path) as db:
                    await db.execute("SELECT 1")
                return {"success": True, "message": "SQLite 连接成功"}

            elif db_type == "mysql":
                import aiomysql
                conn = await aiomysql.connect(
                    host=self._get_db_config_value(config, "mysql_host", "localhost"),
                    port=int(self._get_db_config_value(config, "mysql_port", 3306)),
                    user=self._get_db_config_value(config, "mysql_user", ""),
                    password=self._get_db_config_value(config, "mysql_password", ""),
                    db=self._get_db_config_value(config, "mysql_database", ""),
                )
                await conn.ensure_closed()
                return {"success": True, "message": "MySQL 连接成功"}

            elif db_type == "postgresql":
                import asyncpg
                conn = await asyncpg.connect(
                    host=self._get_db_config_value(config, "postgres_host", "localhost"),
                    port=int(self._get_db_config_value(config, "postgres_port", 5432)),
                    user=self._get_db_config_value(config, "postgres_user", ""),
                    password=self._get_db_config_value(config, "postgres_password", ""),
                    database=self._get_db_config_value(config, "postgres_database", ""),
                )
                await conn.close()
                return {"success": True, "message": "PostgreSQL 连接成功"}

            elif db_type == "sqlserver":
                try:
                    import pyodbc
                except ImportError:
                    return {"success": False, "error": "SQL Server 驱动未安装。请运行: pip install datapilot[sqlserver]"}
                driver = self._get_db_config_value(config, "sqlserver_driver", "ODBC Driver 17 for SQL Server")
                host = self._get_db_config_value(config, "sqlserver_host", "localhost")
                port = self._get_db_config_value(config, "sqlserver_port", 1433)
                database = self._get_db_config_value(config, "sqlserver_database", "master")
                user = self._get_db_config_value(config, "sqlserver_user", "sa")
                password = self._get_db_config_value(config, "sqlserver_password", "")
                conn_str = (
                    f"DRIVER={{{driver}}};"
                    f"SERVER={host},{port};"
                    f"DATABASE={database};"
                    f"UID={user};"
                    f"PWD={password};"
                )
                conn = pyodbc.connect(conn_str, timeout=5)
                conn.close()
                return {"success": True, "message": "SQL Server 连接成功"}

            elif db_type == "clickhouse":
                try:
                    import clickhouse_connect
                except ImportError:
                    return {"success": False, "error": "ClickHouse 驱动未安装。请运行: pip install datapilot[clickhouse]"}
                client = clickhouse_connect.get_client(
                    host=self._get_db_config_value(config, "clickhouse_host", "localhost"),
                    port=int(self._get_db_config_value(config, "clickhouse_port", 8123)),
                    username=self._get_db_config_value(config, "clickhouse_user", "default"),
                    password=self._get_db_config_value(config, "clickhouse_password", ""),
                    database=self._get_db_config_value(config, "clickhouse_database", "default"),
                )
                result = client.query("SELECT 1")
                client.close()
                return {"success": True, "message": "ClickHouse 连接成功"}

            elif db_type == "duckdb":
                try:
                    import duckdb
                except ImportError:
                    return {"success": False, "error": "DuckDB 驱动未安装。请运行: pip install datapilot[duckdb]"}
                db_path = self._get_db_config_value(config, "duckdb_database", "data/datapilot.duckdb")
                conn = duckdb.connect(db_path)
                conn.execute("SELECT 1")
                conn.close()
                return {"success": True, "message": "DuckDB 连接成功"}

        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "未知数据库类型"}

    async def _test_llm_connection(self, config: dict) -> dict:
        """测试 LLM 连接"""
        try:
            import httpx
            import logging
            logger = logging.getLogger(__name__)

            # 如果前端传递的是脱敏值（包含星号），使用已保存的配置
            api_key = config.get("deepseek_api_key", "")
            logger.info(f"[LLM Test] Received api_key: '{api_key[:10]}...' (masked: {self._is_masked_value(api_key) if api_key else 'empty'})")

            if not api_key or self._is_masked_value(api_key):
                api_key = getattr(self.settings, "deepseek_api_key", "")
                logger.info(f"[LLM Test] Using saved api_key: '{api_key[:10]}...' if api_key else 'NOT SET'")

            base_url = config.get("deepseek_base_url") or getattr(self.settings, "deepseek_base_url", "https://api.deepseek.com/v1")

            if not api_key:
                return {"success": False, "error": "未配置 LLM API Key"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )

                if response.status_code == 200:
                    return {"success": True, "message": "DeepSeek API 连接成功"}
                elif response.status_code == 401:
                    return {"success": False, "error": "API Key 无效"}
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}"}

        except httpx.TimeoutException:
            return {"success": False, "error": "连接超时，请检查网络或 API 地址"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_embedding_connection(self, config: dict) -> dict:
        """测试 Embedding API 连接"""
        try:
            import httpx

            # 如果前端传递的是脱敏值（包含星号）或为空，使用已保存的配置
            api_key = config.get("embedding_api_key", "")
            if not api_key or self._is_masked_value(api_key):
                api_key = getattr(self.settings, "embedding_api_key", "")

            base_url = config.get("embedding_base_url") or getattr(self.settings, "embedding_base_url", "https://api.siliconflow.cn/v1")
            model = config.get("embedding_model") or getattr(self.settings, "embedding_model", "BAAI/bge-m3")

            if not api_key:
                return {"success": False, "error": "未配置 Embedding API Key"}

            # 测试 embedding 接口
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "input": "test",
                    },
                    timeout=15,
                )

                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        dim = len(data["data"][0].get("embedding", []))
                        return {"success": True, "message": f"Embedding API 连接成功，向量维度: {dim}"}
                    return {"success": True, "message": "Embedding API 连接成功"}
                elif response.status_code == 401:
                    return {"success": False, "error": "API Key 无效"}
                elif response.status_code == 404:
                    return {"success": False, "error": f"模型不存在: {model}"}
                else:
                    error_detail = ""
                    try:
                        error_detail = response.json().get("error", {}).get("message", "")
                    except:
                        pass
                    return {"success": False, "error": f"HTTP {response.status_code}: {error_detail}"}

        except httpx.TimeoutException:
            return {"success": False, "error": "连接超时，请检查网络或 API 地址"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_rerank_connection(self, config: dict) -> dict:
        """测试 Rerank API 连接"""
        try:
            import httpx

            # 如果前端传递的是脱敏值（包含星号）或为空，使用已保存的配置
            api_key = config.get("rerank_api_key", "")
            if not api_key or self._is_masked_value(api_key):
                api_key = getattr(self.settings, "rerank_api_key", "")

            base_url = config.get("rerank_base_url") or getattr(self.settings, "rerank_base_url", "https://api.siliconflow.cn/v1")
            model = config.get("rerank_model") or getattr(self.settings, "rerank_model", "Qwen/Qwen3-Reranker-8B")

            if not api_key:
                return {"success": False, "error": "未配置 Rerank API Key"}

            # 测试 rerank 接口
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "query": "test query",
                        "documents": ["test document 1", "test document 2"],
                    },
                    timeout=15,
                )

                if response.status_code == 200:
                    return {"success": True, "message": f"Rerank API 连接成功，模型: {model}"}
                elif response.status_code == 401:
                    return {"success": False, "error": "API Key 无效"}
                elif response.status_code == 404:
                    return {"success": False, "error": f"模型不存在: {model}"}
                else:
                    error_detail = ""
                    try:
                        error_detail = response.json().get("error", {}).get("message", "")
                    except:
                        pass
                    return {"success": False, "error": f"HTTP {response.status_code}: {error_detail}"}

        except httpx.TimeoutException:
            return {"success": False, "error": "连接超时，请检查网络或 API 地址"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_redis_connection(self, config: dict) -> dict:
        """测试 Redis 连接"""
        try:
            import redis.asyncio as redis

            url = config.get("redis_url") or getattr(self.settings, "redis_url", "redis://localhost:6379")

            # 如果前端传递的是脱敏值（包含星号）或为空，使用已保存的配置
            password = config.get("redis_password", "")
            if self._is_masked_value(password):
                password = getattr(self.settings, "redis_password", "")

            client = redis.from_url(url, password=password if password else None)
            await client.ping()
            await client.close()

            return {"success": True, "message": "Redis 连接成功"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _test_vector_connection(self, config: dict) -> dict:
        """测试向量数据库连接"""
        backend = config.get("vector_backend") or getattr(self.settings, "vector_backend", "lancedb")

        try:
            if backend == "lancedb" or backend == "auto":
                import lancedb
                path = config.get("lancedb_path") or getattr(self.settings, "lancedb_path", "data/lancedb")
                db = lancedb.connect(path)
                return {"success": True, "message": "LanceDB 连接成功"}

            elif backend == "qdrant":
                from qdrant_client import QdrantClient
                url = config.get("qdrant_url") or getattr(self.settings, "qdrant_url", "http://localhost:6333")

                # 如果前端传递的是脱敏值（包含星号）或为空，使用已保存的配置
                api_key = config.get("qdrant_api_key", "")
                if self._is_masked_value(api_key):
                    api_key = getattr(self.settings, "qdrant_api_key", "")

                client = QdrantClient(url=url, api_key=api_key if api_key else None)
                client.get_collections()
                return {"success": True, "message": "Qdrant 连接成功"}

        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "未知向量数据库类型"}


# 单例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


__all__ = [
    "ConfigManager",
    "ConfigField",
    "CONFIG_SCHEMA",
    "CONFIG_CATEGORIES",
    "get_config_manager",
]
