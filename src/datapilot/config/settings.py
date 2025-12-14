"""
DataPilot 配置管理模块
使用 Pydantic Settings 管理所有配置
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================
    # 应用配置
    # ============================================
    datapilot_env: Literal["development", "staging", "production"] = "development"
    datapilot_debug: bool = True
    datapilot_log_level: str = "INFO"

    # ============================================
    # DeepSeek LLM 配置
    # ============================================
    deepseek_api_key: str = Field(..., description="DeepSeek API Key")
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = 0.1
    deepseek_max_tokens: int = 4096

    # ============================================
    # Embedding 配置 (SiliconFlow API)
    # ============================================
    embedding_api_key: str = Field(
        default="",
        description="SiliconFlow Embedding API Key (set via ENV)",
    )
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"

    # Rerank 配置 (SiliconFlow API)
    rerank_api_key: str = Field(
        default="",
        description="SiliconFlow Rerank API Key (set via ENV)",
    )
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_model: str = "Qwen/Qwen3-Reranker-8B"

    # ============================================
    # 数据库配置
    # ============================================
    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "datapilot"
    mysql_password: str = ""
    mysql_database: str = "ecommerce"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "datapilot"
    postgres_password: str = ""
    postgres_database: str = "sales"

    # SQLite
    sqlite_database: str = "data/datapilot.db"

    # SQL Server
    sqlserver_host: str = "localhost"
    sqlserver_port: int = 1433
    sqlserver_user: str = "sa"
    sqlserver_password: str = ""
    sqlserver_database: str = "master"
    sqlserver_driver: str = "ODBC Driver 17 for SQL Server"

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "default"

    # DuckDB
    duckdb_database: str = "data/datapilot.duckdb"

    # 默认数据库类型
    default_db_type: Literal["mysql", "postgresql", "sqlite", "sqlserver", "clickhouse", "duckdb"] = "sqlite"

    # ============================================
    # Redis 配置
    # ============================================
    redis_url: str = "redis://localhost:6379"
    redis_password: str = ""

    # ============================================
    # 向量数据库配置
    # ============================================
    # LanceDB (推荐，默认)
    lancedb_path: str = "data/lancedb"
    vector_backend: Literal["lancedb", "qdrant", "auto"] = "auto"  # auto = 优先 LanceDB

    # Qdrant (可选，生产环境大规模场景)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_hnsw_m: int = 16
    qdrant_ef_search: int = 128

    # ============================================
    # 缓存配置
    # ============================================
    cache_similarity_threshold: float = 0.85
    cache_ttl_minutes: int = 30
    cache_eviction: str = "lru"

    # ============================================
    # MCP 配置
    # ============================================
    mcp_timeout_seconds: int = 5
    mcp_retries: int = 2
    mcp_row_limit: int = 1000
    mcp_page_limit: int = 200

    # ============================================
    # E2B 沙箱配置
    # ============================================
    e2b_api_key: str = ""
    sandbox_timeout_seconds: int = 10

    # ============================================
    # Agent 配置
    # ============================================
    # VizExpert 模式: codeact (E2B沙箱执行), echarts (直接生成配置), auto (自动选择)
    viz_expert_mode: Literal["codeact", "echarts", "auto"] = "codeact"
    # 是否启用 AmbiResolver (歧义消解)
    ambi_resolver_enabled: bool = True
    # 是否启用 IntentClassifier (意图分类)
    intent_classifier_enabled: bool = True
    # 是否启用 Supervisor (任务拆解)
    supervisor_enabled: bool = True
    # 是否启用语义缓存
    semantic_cache_enabled: bool = True

    # ============================================
    # DataSniper 数据侦察配置
    # ============================================
    data_sniper_use_graph_pruning: bool = True  # 启用基于外键关系的图剪枝
    data_sniper_use_vector_index: bool = True  # 启用向量索引加速
    data_sniper_vector_index_threshold: int = 50  # 表数量超过此阈值时启用向量索引
    data_sniper_schema_top_k: int = 5  # Rerank 返回的种子表数量
    data_sniper_schema_max_hops: int = 2  # 图剪枝最大跳数
    data_sniper_use_llm_entity_extraction: bool = True  # 使用 LLM 提取实体

    # ============================================
    # LogicArchitect SQL 构建配置
    # ============================================
    logic_architect_use_optimized_modules: bool = True  # 优先加载 DSPy 优化模块
    logic_architect_sql_candidates_count: int = 3  # 生成的 SQL 候选数量
    logic_architect_enable_self_correction: bool = True  # 启用 Self-Correction
    logic_architect_max_correction_rounds: int = 2  # 最大修正轮数
    logic_architect_complexity_threshold: int = 2  # 复杂度判断阈值

    # ============================================
    # AmbiResolver 歧义消解配置
    # ============================================
    ambi_resolver_confidence_threshold: float = 0.6  # 置信度阈值
    ambi_resolver_max_options: int = 5  # 最大选项数

    # ============================================
    # VizExpert 可视化专家配置
    # ============================================
    viz_expert_remove_nulls: bool = True  # 移除空值行
    viz_expert_convert_types: bool = True  # 自动类型转换
    viz_expert_remove_outliers: bool = False  # 移除异常值
    viz_expert_outlier_threshold: float = 3.0  # 异常值阈值 (Z-score)
    viz_expert_codeact_data_threshold: int = 100  # 超过此行数使用 CodeAct 模式

    # ============================================
    # MultiJudge 多 Judge 协调配置
    # ============================================
    multi_judge_enabled: bool = True  # 启用多 Judge 协调
    multi_judge_resolution: Literal["majority", "unanimous", "weighted", "llm_arbiter"] = "weighted"
    multi_judge_rule_weight: float = 1.0  # 规则 Judge 权重
    multi_judge_cost_weight: float = 1.2  # 成本 Judge 权重
    multi_judge_semantic_weight: float = 1.5  # 语义 Judge 权重

    # ============================================
    # Judge 成本熔断配置
    # ============================================
    judge_mysql_rows_examined_limit: int = 1_000_000
    judge_mysql_cost_limit: int = 1_000_000
    judge_postgres_cost_limit: int = 500_000
    judge_allow_full_scan: bool = False
    judge_risk_threshold: float = 0.7  # 风险评分阈值
    judge_max_retries: int = 3  # 最大重试次数

    # ============================================
    # 可观测性配置
    # ============================================
    audit_retention_days: int = 90
    prometheus_port: int = 9090
    jwt_secret_key: str = ""
    admin_api_key: str = ""
    user_api_key: str = ""
    allow_mock_users: bool = False  # 默认禁用内存用户，开发可手动开启

    # ============================================
    # API 配置
    # ============================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = True
    
    # LangGraph/LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "datapilot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    checkpointer_url: str = ""  # e.g., postgres+asyncpg://user:pass@host/dbname

    # ============================================
    # 前端配置
    # ============================================
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ============================================
    # 安全配置
    # ============================================
    # CORS 严格模式 (生产环境建议开启)
    cors_strict_mode: bool = False
    cors_allow_credentials: bool = True
    cors_max_age: int = 600  # 预检请求缓存时间 (秒)

    # HTTPS 强制 (生产环境建议开启)
    https_redirect: bool = False
    hsts_enabled: bool = False
    hsts_max_age: int = 31536000  # 1 年

    # 安全响应头
    security_headers_enabled: bool = True

    # ============================================
    # 计算属性
    # ============================================
    @property
    def mysql_url(self) -> str:
        """MySQL 连接 URL"""
        return f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    @property
    def mysql_sync_url(self) -> str:
        """MySQL 同步连接 URL"""
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    @property
    def postgres_url(self) -> str:
        """PostgreSQL 连接 URL"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def postgres_sync_url(self) -> str:
        """PostgreSQL 同步连接 URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def sqlite_url(self) -> str:
        """SQLite 连接 URL (异步)"""
        return f"sqlite+aiosqlite:///{self.sqlite_database}"

    @property
    def sqlite_sync_url(self) -> str:
        """SQLite 同步连接 URL"""
        return f"sqlite:///{self.sqlite_database}"

    @property
    def sqlserver_url(self) -> str:
        """SQL Server 连接 URL (异步)"""
        driver = self.sqlserver_driver.replace(" ", "+")
        return f"mssql+aioodbc://{self.sqlserver_user}:{self.sqlserver_password}@{self.sqlserver_host}:{self.sqlserver_port}/{self.sqlserver_database}?driver={driver}"

    @property
    def sqlserver_sync_url(self) -> str:
        """SQL Server 同步连接 URL"""
        driver = self.sqlserver_driver.replace(" ", "+")
        return f"mssql+pyodbc://{self.sqlserver_user}:{self.sqlserver_password}@{self.sqlserver_host}:{self.sqlserver_port}/{self.sqlserver_database}?driver={driver}"

    @property
    def clickhouse_url(self) -> str:
        """ClickHouse 连接 URL"""
        return f"clickhouse://{self.clickhouse_user}:{self.clickhouse_password}@{self.clickhouse_host}:{self.clickhouse_port}/{self.clickhouse_database}"

    @property
    def duckdb_url(self) -> str:
        """DuckDB 连接 URL (异步)"""
        return f"duckdb:///{self.duckdb_database}"

    @property
    def duckdb_sync_url(self) -> str:
        """DuckDB 同步连接 URL"""
        return f"duckdb:///{self.duckdb_database}"

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS 允许的源列表"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.datapilot_env == "development"

    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.datapilot_env == "production"


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


def clear_settings_cache() -> None:
    """清除配置缓存，用于配置更新后重新加载"""
    get_settings.cache_clear()


# 导出
__all__ = ["Settings", "get_settings", "clear_settings_cache"]
