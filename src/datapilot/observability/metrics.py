# -*- coding: utf-8 -*-
"""
Prometheus Metrics Integration (P2 增强版)
Provides comprehensive metrics for monitoring DataPilot.

README Section 8 要求的指标:
- datapilot_request_total / datapilot_request_latency_seconds (HTTP 请求)
- datapilot_db_query_total / datapilot_db_query_latency_seconds (数据库查询)
- datapilot_llm_request_total / datapilot_llm_request_latency_seconds (LLM 调用)
- datapilot_sandbox_execution_total / datapilot_sandbox_execution_latency_seconds (沙箱执行)
- datapilot_cost_rejections_total (成本熔断)
- datapilot_errors_total (错误计数)

P2 增强指标:
- datapilot_agent_execution_total / datapilot_agent_execution_latency_seconds (Agent 执行)
- datapilot_workflow_total / datapilot_workflow_latency_seconds (完整工作流)
- datapilot_sql_generation_total (SQL 生成统计)
- datapilot_ambiguity_detection_total (歧义检测统计)
- datapilot_vector_search_total (向量搜索统计)
"""

import os
import time
from typing import Callable, Optional
from contextlib import contextmanager
from functools import wraps

from fastapi import FastAPI, Request, Response

# 检查是否禁用指标（测试环境）
_METRICS_DISABLED = os.environ.get("DEEPSQL_DISABLE_METRICS", "").lower() in ("1", "true", "yes")

if not _METRICS_DISABLED:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, Summary, generate_latest, REGISTRY
else:
    # Mock classes for testing
    class _MockMetric:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            pass
        def dec(self, amount=1):
            pass
        def observe(self, amount):
            pass
        def set(self, value):
            pass

    Counter = Histogram = Gauge = Summary = _MockMetric
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None

    def generate_latest():
        return b""


def _get_or_create_counter(name: str, description: str, labelnames: list):
    """获取或创建 Counter，避免重复注册"""
    if _METRICS_DISABLED:
        return _MockMetric()
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    try:
        return Counter(name, description, labelnames=labelnames)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name, _MockMetric())


def _get_or_create_histogram(name: str, description: str, labelnames: list, buckets: tuple):
    """获取或创建 Histogram，避免重复注册"""
    if _METRICS_DISABLED:
        return _MockMetric()
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    try:
        return Histogram(name, description, labelnames=labelnames, buckets=buckets)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name, _MockMetric())


def _get_or_create_gauge(name: str, description: str, labelnames: list):
    """获取或创建 Gauge，避免重复注册"""
    if _METRICS_DISABLED:
        return _MockMetric()
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    try:
        return Gauge(name, description, labelnames=labelnames)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name, _MockMetric())


def _get_or_create_summary(name: str, description: str, labelnames: list):
    """获取或创建 Summary，避免重复注册"""
    if _METRICS_DISABLED:
        return _MockMetric()
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    try:
        return Summary(name, description, labelnames=labelnames)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name, _MockMetric())


# ============================================
# HTTP 请求指标
# ============================================
REQUEST_COUNT = _get_or_create_counter(
    "datapilot_request_total",
    "HTTP request count",
    labelnames=["method", "path", "status"],
)

REQUEST_LATENCY = _get_or_create_histogram(
    "datapilot_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "path", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

# ============================================
# 数据库查询指标
# ============================================
DB_QUERY_COUNT = _get_or_create_counter(
    "datapilot_db_query_total",
    "Database query count",
    labelnames=["db_type", "status"],
)

DB_QUERY_LATENCY = _get_or_create_histogram(
    "datapilot_db_query_latency_seconds",
    "Database query latency in seconds",
    labelnames=["db_type", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

DB_QUERY_ROWS = _get_or_create_histogram(
    "datapilot_db_query_rows",
    "Number of rows returned by queries",
    labelnames=["db_type"],
    buckets=(1, 10, 50, 100, 500, 1000, 5000, 10000),
)

# ============================================
# LLM 调用指标
# ============================================
LLM_REQUEST_COUNT = _get_or_create_counter(
    "datapilot_llm_request_total",
    "LLM request count",
    labelnames=["model", "status"],
)

LLM_REQUEST_LATENCY = _get_or_create_histogram(
    "datapilot_llm_request_latency_seconds",
    "LLM request latency in seconds",
    labelnames=["model", "status"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

LLM_TOKENS_USED = _get_or_create_counter(
    "datapilot_llm_tokens_total",
    "Total tokens used in LLM requests",
    labelnames=["model", "type"],  # type: input/output
)

# ============================================
# 沙箱执行指标
# ============================================
SANDBOX_EXECUTION_COUNT = _get_or_create_counter(
    "datapilot_sandbox_execution_total",
    "Sandbox execution count",
    labelnames=["executor", "status"],
)

SANDBOX_EXECUTION_LATENCY = _get_or_create_histogram(
    "datapilot_sandbox_execution_latency_seconds",
    "Sandbox execution latency in seconds",
    labelnames=["executor", "status"],
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

# ============================================
# 错误和熔断指标
# ============================================
ERRORS_COUNT = _get_or_create_counter(
    "datapilot_errors_total",
    "Total error count",
    labelnames=["error_type"],
)

COST_REJECTIONS_COUNT = _get_or_create_counter(
    "datapilot_cost_rejections_total",
    "Cost circuit breaker rejection count",
    labelnames=["db_type", "reason"],
)

# ============================================
# 缓存指标
# ============================================
CACHE_HITS = _get_or_create_counter(
    "datapilot_cache_hits_total",
    "Cache hit count",
    labelnames=["cache_type"],
)

CACHE_MISSES = _get_or_create_counter(
    "datapilot_cache_misses_total",
    "Cache miss count",
    labelnames=["cache_type"],
)

CACHE_SIZE = _get_or_create_gauge(
    "datapilot_cache_size",
    "Current cache size (entries)",
    labelnames=["cache_type"],
)

# ============================================
# 活跃连接数
# ============================================
ACTIVE_CONNECTIONS = _get_or_create_gauge(
    "datapilot_active_connections",
    "Number of active connections",
    labelnames=["type"],
)

# ============================================
# P2 新增: Agent 执行指标
# ============================================
AGENT_EXECUTION_COUNT = _get_or_create_counter(
    "datapilot_agent_execution_total",
    "Agent execution count",
    labelnames=["agent", "status"],
)

AGENT_EXECUTION_LATENCY = _get_or_create_histogram(
    "datapilot_agent_execution_latency_seconds",
    "Agent execution latency in seconds",
    labelnames=["agent", "status"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
)

AGENT_RETRIES = _get_or_create_counter(
    "datapilot_agent_retries_total",
    "Agent retry count",
    labelnames=["agent"],
)

# ============================================
# P2 新增: 工作流指标
# ============================================
WORKFLOW_COUNT = _get_or_create_counter(
    "datapilot_workflow_total",
    "Complete workflow execution count",
    labelnames=["status", "cache_hit"],
)

WORKFLOW_LATENCY = _get_or_create_histogram(
    "datapilot_workflow_latency_seconds",
    "Complete workflow latency in seconds",
    labelnames=["status", "cache_hit"],
    buckets=(1, 2, 5, 10, 30, 60, 120),
)

WORKFLOW_AGENT_STEPS = _get_or_create_histogram(
    "datapilot_workflow_agent_steps",
    "Number of agent steps in workflow",
    labelnames=["status"],
    buckets=(1, 2, 3, 5, 7, 10, 15),
)

# ============================================
# P2 新增: SQL 生成指标
# ============================================
SQL_GENERATION_COUNT = _get_or_create_counter(
    "datapilot_sql_generation_total",
    "SQL generation count",
    labelnames=["strategy", "status"],  # strategy: direct/decompose/refinement
)

SQL_CANDIDATES_COUNT = _get_or_create_histogram(
    "datapilot_sql_candidates",
    "Number of SQL candidates generated",
    labelnames=["strategy"],
    buckets=(1, 2, 3, 4, 5),
)

SQL_VALIDATION_COUNT = _get_or_create_counter(
    "datapilot_sql_validation_total",
    "SQL validation count",
    labelnames=["status", "issue_type"],  # issue_type: syntax/security/injection/etc
)

# ============================================
# P2 新增: 歧义检测指标
# ============================================
AMBIGUITY_DETECTION_COUNT = _get_or_create_counter(
    "datapilot_ambiguity_detection_total",
    "Ambiguity detection count",
    labelnames=["result"],  # result: ambiguous/clear
)

CLARIFICATION_REQUESTS = _get_or_create_counter(
    "datapilot_clarification_requests_total",
    "User clarification request count",
    labelnames=["type"],  # type: time/entity/scope/etc
)

HUMAN_HANDOFF_COUNT = _get_or_create_counter(
    "datapilot_human_handoff_total",
    "Human handoff count",
    labelnames=["reason"],
)

# ============================================
# P2 新增: 向量搜索指标
# ============================================
VECTOR_SEARCH_COUNT = _get_or_create_counter(
    "datapilot_vector_search_total",
    "Vector search count",
    labelnames=["index_type", "status"],  # index_type: schema/value/cache
)

VECTOR_SEARCH_LATENCY = _get_or_create_histogram(
    "datapilot_vector_search_latency_seconds",
    "Vector search latency in seconds",
    labelnames=["index_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1),
)

VECTOR_INDEX_SIZE = _get_or_create_gauge(
    "datapilot_vector_index_size",
    "Vector index size (entries)",
    labelnames=["index_type"],
)

# ============================================
# P2 新增: Schema 剪枝指标
# ============================================
SCHEMA_PRUNING_COUNT = _get_or_create_counter(
    "datapilot_schema_pruning_total",
    "Schema pruning count",
    labelnames=["method"],  # method: vector/graph/rerank
)

SCHEMA_TABLES_PRUNED = _get_or_create_histogram(
    "datapilot_schema_tables_pruned",
    "Number of tables after pruning",
    labelnames=["method"],
    buckets=(1, 2, 3, 5, 7, 10, 15, 20),
)

# ============================================
# P2 新增: 可视化指标
# ============================================
VISUALIZATION_COUNT = _get_or_create_counter(
    "datapilot_visualization_total",
    "Visualization generation count",
    labelnames=["chart_type", "status"],
)

CHART_RECOMMENDATION_COUNT = _get_or_create_counter(
    "datapilot_chart_recommendation_total",
    "Chart recommendation count",
    labelnames=["recommended_type"],
)


def _normalized_path(path: str) -> str:
    """Normalize path to reduce cardinality (basic static path only)."""
    return path or "/"


# ============================================
# 辅助函数 - 记录指标
# ============================================

@contextmanager
def track_db_query(db_type: str):
    """跟踪数据库查询的上下文管理器"""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        ERRORS_COUNT.labels(error_type="db_error").inc()
        raise
    finally:
        duration = time.perf_counter() - start
        DB_QUERY_COUNT.labels(db_type=db_type, status=status).inc()
        DB_QUERY_LATENCY.labels(db_type=db_type, status=status).observe(duration)


@contextmanager
def track_llm_request(model: str):
    """跟踪 LLM 请求的上下文管理器"""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        ERRORS_COUNT.labels(error_type="llm_error").inc()
        raise
    finally:
        duration = time.perf_counter() - start
        LLM_REQUEST_COUNT.labels(model=model, status=status).inc()
        LLM_REQUEST_LATENCY.labels(model=model, status=status).observe(duration)


@contextmanager
def track_sandbox_execution(executor: str):
    """跟踪沙箱执行的上下文管理器"""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        ERRORS_COUNT.labels(error_type="sandbox_error").inc()
        raise
    finally:
        duration = time.perf_counter() - start
        SANDBOX_EXECUTION_COUNT.labels(executor=executor, status=status).inc()
        SANDBOX_EXECUTION_LATENCY.labels(executor=executor, status=status).observe(duration)


@contextmanager
def track_agent_execution(agent: str):
    """跟踪 Agent 执行的上下文管理器 (P2 新增)"""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        ERRORS_COUNT.labels(error_type=f"agent_{agent}_error").inc()
        raise
    finally:
        duration = time.perf_counter() - start
        AGENT_EXECUTION_COUNT.labels(agent=agent, status=status).inc()
        AGENT_EXECUTION_LATENCY.labels(agent=agent, status=status).observe(duration)


@contextmanager
def track_workflow(cache_hit: bool = False):
    """跟踪完整工作流的上下文管理器 (P2 新增)"""
    start = time.perf_counter()
    status = "success"
    cache_hit_str = "true" if cache_hit else "false"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        WORKFLOW_COUNT.labels(status=status, cache_hit=cache_hit_str).inc()
        WORKFLOW_LATENCY.labels(status=status, cache_hit=cache_hit_str).observe(duration)


@contextmanager
def track_vector_search(index_type: str):
    """跟踪向量搜索的上下文管理器 (P2 新增)"""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        VECTOR_SEARCH_COUNT.labels(index_type=index_type, status=status).inc()
        VECTOR_SEARCH_LATENCY.labels(index_type=index_type).observe(duration)


def record_llm_tokens(model: str, input_tokens: int, output_tokens: int):
    """记录 LLM token 使用量"""
    LLM_TOKENS_USED.labels(model=model, type="input").inc(input_tokens)
    LLM_TOKENS_USED.labels(model=model, type="output").inc(output_tokens)


def record_cost_rejection(db_type: str, reason: str):
    """记录成本熔断"""
    COST_REJECTIONS_COUNT.labels(db_type=db_type, reason=reason).inc()


def record_cache_hit(cache_type: str = "semantic"):
    """记录缓存命中"""
    CACHE_HITS.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str = "semantic"):
    """记录缓存未命中"""
    CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_error(error_type: str):
    """记录错误"""
    ERRORS_COUNT.labels(error_type=error_type).inc()


def record_agent_retry(agent: str):
    """记录 Agent 重试 (P2 新增)"""
    AGENT_RETRIES.labels(agent=agent).inc()


def record_sql_generation(strategy: str, success: bool):
    """记录 SQL 生成 (P2 新增)"""
    status = "success" if success else "error"
    SQL_GENERATION_COUNT.labels(strategy=strategy, status=status).inc()


def record_sql_candidates(strategy: str, count: int):
    """记录 SQL 候选数量 (P2 新增)"""
    SQL_CANDIDATES_COUNT.labels(strategy=strategy).observe(count)


def record_sql_validation(valid: bool, issue_type: str = "none"):
    """记录 SQL 验证结果 (P2 新增)"""
    status = "valid" if valid else "invalid"
    SQL_VALIDATION_COUNT.labels(status=status, issue_type=issue_type).inc()


def record_ambiguity(is_ambiguous: bool):
    """记录歧义检测结果 (P2 新增)"""
    result = "ambiguous" if is_ambiguous else "clear"
    AMBIGUITY_DETECTION_COUNT.labels(result=result).inc()


def record_clarification_request(clarification_type: str):
    """记录澄清请求 (P2 新增)"""
    CLARIFICATION_REQUESTS.labels(type=clarification_type).inc()


def record_human_handoff(reason: str):
    """记录人工介入 (P2 新增)"""
    HUMAN_HANDOFF_COUNT.labels(reason=reason).inc()


def record_schema_pruning(method: str, tables_count: int):
    """记录 Schema 剪枝 (P2 新增)"""
    SCHEMA_PRUNING_COUNT.labels(method=method).inc()
    SCHEMA_TABLES_PRUNED.labels(method=method).observe(tables_count)


def record_visualization(chart_type: str, success: bool):
    """记录可视化生成 (P2 新增)"""
    status = "success" if success else "error"
    VISUALIZATION_COUNT.labels(chart_type=chart_type, status=status).inc()


def record_chart_recommendation(chart_type: str):
    """记录图表推荐 (P2 新增)"""
    CHART_RECOMMENDATION_COUNT.labels(recommended_type=chart_type).inc()


def record_workflow_steps(status: str, step_count: int):
    """记录工作流步骤数 (P2 新增)"""
    WORKFLOW_AGENT_STEPS.labels(status=status).observe(step_count)


def update_cache_size(cache_type: str, size: int):
    """更新缓存大小 (P2 新增)"""
    CACHE_SIZE.labels(cache_type=cache_type).set(size)


def update_vector_index_size(index_type: str, size: int):
    """更新向量索引大小 (P2 新增)"""
    VECTOR_INDEX_SIZE.labels(index_type=index_type).set(size)


def record_db_query_rows(db_type: str, row_count: int):
    """记录查询返回行数 (P2 新增)"""
    DB_QUERY_ROWS.labels(db_type=db_type).observe(row_count)


def setup_metrics(app: FastAPI) -> None:
    """Attach metrics middleware and /metrics endpoint to the app."""

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        path = _normalized_path(request.url.path)
        method = request.method.upper()
        status = response.status_code

        REQUEST_COUNT.labels(method=method, path=path, status=status).inc()
        REQUEST_LATENCY.labels(method=method, path=path, status=status).observe(duration)

        return response

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)


__all__ = [
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
]
