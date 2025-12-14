# -*- coding: utf-8 -*-
"""
Observability utilities.

包含:
- Prometheus 指标收集
- 审计日志系统
"""

from .metrics import (
    setup_metrics,
    # HTTP 指标
    REQUEST_COUNT, REQUEST_LATENCY,
    # DB 指标
    DB_QUERY_COUNT, DB_QUERY_LATENCY, DB_QUERY_ROWS,
    # LLM 指标
    LLM_REQUEST_COUNT, LLM_REQUEST_LATENCY, LLM_TOKENS_USED,
    # 沙箱指标
    SANDBOX_EXECUTION_COUNT, SANDBOX_EXECUTION_LATENCY,
    # 错误/熔断指标
    ERRORS_COUNT, COST_REJECTIONS_COUNT,
    # 缓存指标
    CACHE_HITS, CACHE_MISSES, CACHE_SIZE,
    # 连接指标
    ACTIVE_CONNECTIONS,
    # Agent 指标 (P2)
    AGENT_EXECUTION_COUNT, AGENT_EXECUTION_LATENCY, AGENT_RETRIES,
    # 工作流指标 (P2)
    WORKFLOW_COUNT, WORKFLOW_LATENCY, WORKFLOW_AGENT_STEPS,
    # SQL 指标 (P2)
    SQL_GENERATION_COUNT, SQL_CANDIDATES_COUNT, SQL_VALIDATION_COUNT,
    # 歧义指标 (P2)
    AMBIGUITY_DETECTION_COUNT, CLARIFICATION_REQUESTS, HUMAN_HANDOFF_COUNT,
    # 向量指标 (P2)
    VECTOR_SEARCH_COUNT, VECTOR_SEARCH_LATENCY, VECTOR_INDEX_SIZE,
    # Schema 指标 (P2)
    SCHEMA_PRUNING_COUNT, SCHEMA_TABLES_PRUNED,
    # 可视化指标 (P2)
    VISUALIZATION_COUNT, CHART_RECOMMENDATION_COUNT,
    # 跟踪上下文管理器
    track_db_query, track_llm_request, track_sandbox_execution,
    track_agent_execution, track_workflow, track_vector_search,
    # 记录函数
    record_llm_tokens, record_cost_rejection,
    record_cache_hit, record_cache_miss, record_error,
    record_agent_retry, record_sql_generation, record_sql_candidates,
    record_sql_validation, record_ambiguity, record_clarification_request,
    record_human_handoff, record_schema_pruning, record_visualization,
    record_chart_recommendation, record_workflow_steps,
    update_cache_size, update_vector_index_size, record_db_query_rows,
)

from .audit import (
    # 数据类
    AuditEntry,
    # 存储
    AuditStore,
    get_audit_store,
    init_audit_store,
    shutdown_audit_store,
    # 工具
    AuditTimer,
    audit_context,
    # 函数
    log_audit,
)

__all__ = [
    # Metrics
    "setup_metrics",
    # HTTP 指标
    "REQUEST_COUNT", "REQUEST_LATENCY",
    # DB 指标
    "DB_QUERY_COUNT", "DB_QUERY_LATENCY", "DB_QUERY_ROWS",
    # LLM 指标
    "LLM_REQUEST_COUNT", "LLM_REQUEST_LATENCY", "LLM_TOKENS_USED",
    # 沙箱指标
    "SANDBOX_EXECUTION_COUNT", "SANDBOX_EXECUTION_LATENCY",
    # 错误/熔断指标
    "ERRORS_COUNT", "COST_REJECTIONS_COUNT",
    # 缓存指标
    "CACHE_HITS", "CACHE_MISSES", "CACHE_SIZE",
    # 连接指标
    "ACTIVE_CONNECTIONS",
    # Agent 指标 (P2)
    "AGENT_EXECUTION_COUNT", "AGENT_EXECUTION_LATENCY", "AGENT_RETRIES",
    # 工作流指标 (P2)
    "WORKFLOW_COUNT", "WORKFLOW_LATENCY", "WORKFLOW_AGENT_STEPS",
    # SQL 指标 (P2)
    "SQL_GENERATION_COUNT", "SQL_CANDIDATES_COUNT", "SQL_VALIDATION_COUNT",
    # 歧义指标 (P2)
    "AMBIGUITY_DETECTION_COUNT", "CLARIFICATION_REQUESTS", "HUMAN_HANDOFF_COUNT",
    # 向量指标 (P2)
    "VECTOR_SEARCH_COUNT", "VECTOR_SEARCH_LATENCY", "VECTOR_INDEX_SIZE",
    # Schema 指标 (P2)
    "SCHEMA_PRUNING_COUNT", "SCHEMA_TABLES_PRUNED",
    # 可视化指标 (P2)
    "VISUALIZATION_COUNT", "CHART_RECOMMENDATION_COUNT",
    # 跟踪上下文管理器
    "track_db_query", "track_llm_request", "track_sandbox_execution",
    "track_agent_execution", "track_workflow", "track_vector_search",
    # 记录函数
    "record_llm_tokens", "record_cost_rejection",
    "record_cache_hit", "record_cache_miss", "record_error",
    "record_agent_retry", "record_sql_generation", "record_sql_candidates",
    "record_sql_validation", "record_ambiguity", "record_clarification_request",
    "record_human_handoff", "record_schema_pruning", "record_visualization",
    "record_chart_recommendation", "record_workflow_steps",
    "update_cache_size", "update_vector_index_size", "record_db_query_rows",
    # Audit
    "AuditEntry",
    "AuditStore",
    "get_audit_store",
    "init_audit_store",
    "shutdown_audit_store",
    "AuditTimer",
    "audit_context",
    "log_audit",
]
