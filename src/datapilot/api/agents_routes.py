# -*- coding: utf-8 -*-
"""
Agents 监控 API 路由

提供 Agent 执行监控、统计和历史查询功能
"""

from typing import Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..observability.audit import get_audit_store

router = APIRouter(prefix="/api/v1/agents", tags=["Agents Monitor"])


# ============================================
# 响应模型
# ============================================

class AgentInfo(BaseModel):
    """Agent 信息"""
    id: str
    name: str
    display_name: str
    description: str
    icon: str


class AgentStatsResponse(BaseModel):
    """Agent 统计响应"""
    success: bool = True
    data: dict = Field(default_factory=dict)


class ActiveQueryInfo(BaseModel):
    """活跃查询信息"""
    trace_id: str
    query: str
    database: str
    user_id: str
    current_agent: str
    start_time: str


class ExecutionDetailResponse(BaseModel):
    """执行详情响应"""
    success: bool = True
    data: Optional[dict] = None


class HistoryResponse(BaseModel):
    """历史记录响应"""
    success: bool = True
    data: list = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20
    has_more: bool = False


class ErrorListResponse(BaseModel):
    """错误列表响应"""
    success: bool = True
    data: list = Field(default_factory=list)
    total: int = 0


# ============================================
# Agent 定义
# ============================================

AGENTS = [
    {
        "id": "cache_check",
        "name": "CacheCheck",
        "display_name": "语义缓存",
        "description": "检查精确匹配和语义相似度缓存，命中则直接返回结果",
        "icon": "ThunderboltOutlined",
        "category": "preprocessing",
    },
    {
        "id": "intent_classifier",
        "name": "IntentClassifier",
        "display_name": "意图分类",
        "description": "使用 LLM 分析查询意图：聚合、查找、对比、趋势、排名、复杂",
        "icon": "AimOutlined",
        "category": "preprocessing",
    },
    {
        "id": "ambi_resolver",
        "name": "AmbiResolver",
        "display_name": "歧义消解",
        "description": "检测时间/指标/范围/粒度/实体歧义，支持 Human-in-the-loop 澄清",
        "icon": "QuestionCircleOutlined",
        "category": "preprocessing",
    },
    {
        "id": "supervisor",
        "name": "Supervisor",
        "display_name": "任务协调器",
        "description": "分析查询复杂度，制定执行计划，路由到对应执行路径",
        "icon": "TeamOutlined",
        "category": "routing",
    },
    {
        "id": "data_sniper",
        "name": "DataSniper",
        "display_name": "数据侦察",
        "description": "Schema 剪枝、值映射、业务词汇匹配、时间/枚举字段识别",
        "icon": "DatabaseOutlined",
        "category": "execution",
    },
    {
        "id": "logic_architect",
        "name": "LogicArchitect",
        "display_name": "SQL 构建",
        "description": "基于 DSPy + MAC-SQL 生成多路 SQL 候选，支持 Self-Correction",
        "icon": "CodeOutlined",
        "category": "execution",
    },
    {
        "id": "judge",
        "name": "Judge",
        "display_name": "裁判校验",
        "description": "SQL 语法校验、安全检查、成本熔断、执行计划分析",
        "icon": "SafetyOutlined",
        "category": "execution",
    },
    {
        "id": "viz_expert",
        "name": "VizExpert",
        "display_name": "可视化专家",
        "description": "生成 ECharts 配置或 Python 代码，支持 E2B 沙箱执行",
        "icon": "BarChartOutlined",
        "category": "output",
    },
]

# 复杂度路由说明 - 包含完整的 Agent 执行路径
# 前置阶段 (所有复杂度共用): CacheCheck -> IntentClassifier -> AmbiResolver -> Supervisor -> DataSniper
# 执行阶段 (根据复杂度不同): LogicArchitect / ParallelExecutor / SupervisorReAct
# 后置阶段 (所有复杂度共用): Judge -> VizExpert
COMPLEXITY_ROUTES = {
    "simple": {
        "name": "简单查询",
        "description": "单表查询、简单聚合 → 直接生成 SQL",
        "path": ["CacheCheck", "IntentClassifier", "AmbiResolver", "Supervisor", "DataSniper", "LogicArchitect", "Judge", "VizExpert"],
        "key_agent": "LogicArchitect",
        "color": "#52c41a",
    },
    "medium": {
        "name": "中等复杂",
        "description": "多维对比、时间序列 → 并行执行子任务",
        "path": ["CacheCheck", "IntentClassifier", "AmbiResolver", "Supervisor", "DataSniper", "ParallelExecutor", "Judge", "VizExpert"],
        "key_agent": "ParallelExecutor",
        "color": "#1890ff",
    },
    "high": {
        "name": "高复杂度",
        "description": "多步依赖、模糊查询 → ReAct 多轮推理",
        "path": ["CacheCheck", "IntentClassifier", "AmbiResolver", "Supervisor", "DataSniper", "SupervisorReAct", "Judge", "VizExpert"],
        "key_agent": "SupervisorReAct",
        "color": "#faad14",
    },
}

# 活跃查询存储 (内存中)
_active_queries: dict[str, dict] = {}


def register_active_query(trace_id: str, query_info: dict):
    """注册活跃查询"""
    _active_queries[trace_id] = {
        **query_info,
        "start_time": datetime.utcnow().isoformat() + "Z",
    }


def update_active_query(trace_id: str, updates: dict):
    """更新活跃查询状态"""
    if trace_id in _active_queries:
        _active_queries[trace_id].update(updates)


def remove_active_query(trace_id: str):
    """移除活跃查询"""
    _active_queries.pop(trace_id, None)


# ============================================
# 路由定义
# ============================================

@router.get("/list")
async def list_agents():
    """
    获取所有 Agent 列表

    返回系统中所有 Agent 的基本信息
    """
    return {
        "success": True,
        "data": AGENTS,
    }


@router.get("/stats")
async def get_agent_stats(
    start_time: Optional[str] = Query(None, description="开始时间 (ISO 格式)"),
    end_time: Optional[str] = Query(None, description="结束时间 (ISO 格式)"),
    tenant_id: Optional[str] = Query(None, description="租户 ID"),
):
    """
    获取 Agent 执行统计

    返回:
    - 总查询数、成功率、平均延迟
    - 缓存命中率
    - 各 Agent 执行统计
    - 每日趋势
    """
    store = get_audit_store()

    # 默认时间范围: 最近 7 天
    if not start_time:
        start_time = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
    if not end_time:
        end_time = datetime.utcnow().isoformat() + "Z"

    try:
        stats = await store.get_statistics(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
        )

        # 计算成功率
        overall = stats.get("overall", {})
        total = overall.get("total_queries", 0)
        success = overall.get("success_count", 0)
        success_rate = (success / total * 100) if total > 0 else 0

        return {
            "success": True,
            "data": {
                "summary": {
                    "total_queries": total,
                    "success_count": success,
                    "error_count": overall.get("error_count", 0),
                    "rejected_count": overall.get("rejected_count", 0),
                    "success_rate": round(success_rate, 2),
                    "cache_hit_rate": round(overall.get("cache_hit_rate", 0) * 100, 2),
                    "avg_duration_ms": round(overall.get("avg_duration_ms", 0), 2),
                    "max_duration_ms": round(overall.get("max_duration_ms", 0), 2),
                },
                "by_database": stats.get("by_database", ),
                "by_status": stats.get("by_status", {}),
                "top_users": stats.get("top_users", []),
                "daily_trend": stats.get("daily_trend", []),
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                },
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.get("/active")
async def get_active_queries():
    """
    获取当前活跃的查询

    返回正在执行中的查询列表
    """
    return {
        "success": True,
        "data": list(_active_queries.values()),
        "count": len(_active_queries),
    }


@router.get("/execution/{trace_id}")
async def get_execution_detail(trace_id: str):
    """
    获取单次执行的详细信息

    Args:
        trace_id: 追踪 ID

    返回:
    - 查询信息
    - Agent 执行路径
    - 各阶段耗时
    - 最终结果
    """
    store = get_audit_store()

    try:
        result = await store.get_by_trace_id(trace_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"执行记录不存在: {trace_id}")

        return {
            "success": True,
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取执行详情失败: {str(e)}")


@router.get("/history")
async def get_execution_history(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="状态过滤"),
    database: Optional[str] = Query(None, description="数据库过滤"),
    user_id: Optional[str] = Query(None, description="用户 ID 过滤"),
    start_time: Optional[str] = Query(None, description="开始时间"),
    end_time: Optional[str] = Query(None, description="结束时间"),
    search: Optional[str] = Query(None, description="搜索关键词"),
):
    """
    获取执行历史 (分页)

    支持多种筛选条件
    """
    store = get_audit_store()

    try:
        # 计算偏移量
        offset = (page - 1) * page_size

        # 查询数据
        results = await store.query(
            user_id=user_id,
            database=database,
            status=status,
            start_time=start_time,
            end_time=end_time,
            search_query=search,
            limit=page_size + 1,  # 多查一条判断是否有更多
            offset=offset,
            order_by="timestamp",
            order_dir="DESC",
        )

        # 判断是否有更多
        has_more = len(results) > page_size
        if has_more:
            results = results[:page_size]

        # 获取总数
        total = await store.count(
            status=status,
            start_time=start_time,
            end_time=end_time,
        )

        return {
            "success": True,
            "data": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@router.get("/metrics")
async def get_performance_metrics(
    start_time: Optional[str] = Query(None, description="开始时间"),
    end_time: Optional[str] = Query(None, description="结束时间"),
):
    """
    获取性能指标

    返回:
    - 延迟分布
    - Agent 性能对比
    - 每日趋势
    - 错误分类
    """
    store = get_audit_store()

    # 默认时间范围: 最近 30 天
    if not start_time:
        start_time = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
    if not end_time:
        end_time = datetime.utcnow().isoformat() + "Z"

    try:
        stats = await store.get_statistics(
            start_time=start_time,
            end_time=end_time,
        )

        # 查询最近的执行记录用于延迟分布
        recent_logs = await store.query(
            start_time=start_time,
            end_time=end_time,
            limit=1000,
            order_by="timestamp",
            order_dir="DESC",
        )

        # 计算延迟分布
        latency_buckets = {
            "0-100ms": 0,
            "100-500ms": 0,
            "500-1000ms": 0,
            "1-3s": 0,
            "3-10s": 0,
            ">10s": 0,
        }

        for log in recent_logs:
            duration = log.get("duration_ms", 0) or 0
            if duration < 100:
                latency_buckets["0-100ms"] += 1
            elif duration < 500:
                latency_buckets["100-500ms"] += 1
            elif duration < 1000:
                latency_buckets["500-1000ms"] += 1
            elif duration < 3000:
                latency_buckets["1-3s"] += 1
            elif duration < 10000:
                latency_buckets["3-10s"] += 1
            else:
                latency_buckets[">10s"] += 1

        # 计算 Agent 执行统计
        agent_stats = {}
        for log in recent_logs:
            agent_path = log.get("agent_path", [])
            for agent in agent_path:
                if agent not in agent_stats:
                    agent_stats[agent] = {"count": 0, "total_duration": 0}
                agent_stats[agent]["count"] += 1

        return {
            "success": True,
            "data": {
                "latency_distribution": latency_buckets,
                "agent_performance": agent_stats,
                "daily_trend": stats.get("daily_trend", []),
                "by_status": stats.get("by_status", {}),
                "by_database": stats.get("by_database", {}),
                "time_range": {
                    "start": start_time,
                    "end": end_time,
                },
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")


@router.get("/errors")
async def get_error_list(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    start_time: Optional[str] = Query(None, description="开始时间"),
    end_time: Optional[str] = Query(None, description="结束时间"),
):
    """
    获取错误列表

    返回执行失败的查询记录
    """
    store = get_audit_store()

    try:
        offset = (page - 1) * page_size

        # 查询错误记录
        results = await store.query(
            status="error",
            start_time=start_time,
            end_time=end_time,
            limit=page_size + 1,
            offset=offset,
            order_by="timestamp",
            order_dir="DESC",
        )

        has_more = len(results) > page_size
        if has_more:
            results = results[:page_size]

        # 获取错误总数
        total = await store.count(
            status="error",
            start_time=start_time,
            end_time=end_time,
        )

        return {
            "success": True,
            "data": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取错误列表失败: {str(e)}")


@router.get("/workflow")
async def get_workflow_definition():
    """
    获取工作流定义

    返回 Agent 工作流的节点和边定义，用于前端绘制流程图
    包含三级复杂度路由：简单、中等、复杂
    """
    nodes = [
        # 输入
        {"id": "start", "type": "input", "label": "用户查询", "position": {"x": 300, "y": 0}, "category": "input"},

        # 预处理阶段
        {"id": "cache_check", "type": "default", "label": "语义缓存", "position": {"x": 300, "y": 70}, "category": "preprocessing", "description": "精确匹配 + 语义相似度检查"},
        {"id": "intent_classifier", "type": "default", "label": "意图分类", "position": {"x": 300, "y": 140}, "category": "preprocessing", "description": "聚合/查找/对比/趋势/排名/复杂"},
        {"id": "ambi_resolver", "type": "default", "label": "歧义消解", "position": {"x": 300, "y": 210}, "category": "preprocessing", "description": "时间/指标/范围/粒度/实体歧义"},

        # 路由阶段
        {"id": "supervisor", "type": "default", "label": "Supervisor", "position": {"x": 300, "y": 280}, "category": "routing", "description": "分析复杂度，制定执行计划"},

        # 数据侦察 (所有路径共用)
        {"id": "data_sniper", "type": "default", "label": "DataSniper", "position": {"x": 300, "y": 350}, "category": "execution", "description": "Schema 剪枝、值映射"},

        # 三条执行路径
        {"id": "logic_architect", "type": "default", "label": "LogicArchitect", "position": {"x": 100, "y": 420}, "category": "simple", "description": "简单查询：直接生成 SQL"},
        {"id": "parallel_executor", "type": "default", "label": "并行执行器", "position": {"x": 300, "y": 420}, "category": "medium", "description": "中等复杂：拆分子任务并行执行"},
        {"id": "supervisor_react", "type": "default", "label": "ReAct 推理", "position": {"x": 500, "y": 420}, "category": "high", "description": "高复杂度：多轮推理循环"},

        # 校验和输出
        {"id": "judge", "type": "default", "label": "Judge", "position": {"x": 300, "y": 490}, "category": "execution", "description": "SQL 校验、成本熔断"},
        {"id": "viz_expert", "type": "default", "label": "VizExpert", "position": {"x": 300, "y": 560}, "category": "output", "description": "ECharts 配置 / Python 代码"},
        {"id": "end", "type": "output", "label": "返回结果", "position": {"x": 300, "y": 630}, "category": "output"},

        # 分支节点
        {"id": "clarify", "type": "default", "label": "用户澄清", "position": {"x": 520, "y": 210}, "category": "branch", "description": "Human-in-the-loop"},
        {"id": "retry", "type": "default", "label": "错误重试", "position": {"x": 520, "y": 490}, "category": "branch", "description": "最多 3 次重试"},
    ]

    edges = [
        # 主流程
        {"id": "e1", "source": "start", "target": "cache_check"},
        {"id": "e2", "source": "cache_check", "target": "intent_classifier", "label": "未命中"},
        {"id": "e3", "source": "cache_check", "target": "viz_expert", "label": "命中", "style": "dashed"},
        {"id": "e4", "source": "intent_classifier", "target": "ambi_resolver"},
        {"id": "e5", "source": "ambi_resolver", "target": "supervisor", "label": "无歧义"},
        {"id": "e6", "source": "ambi_resolver", "target": "clarify", "label": "需澄清"},
        {"id": "e7", "source": "clarify", "target": "ambi_resolver", "label": "用户选择"},
        {"id": "e8", "source": "supervisor", "target": "data_sniper"},

        # 三条复杂度路径
        {"id": "e9", "source": "data_sniper", "target": "logic_architect", "label": "简单", "color": "#52c41a"},
        {"id": "e10", "source": "data_sniper", "target": "parallel_executor", "label": "中等", "color": "#1890ff"},
        {"id": "e11", "source": "data_sniper", "target": "supervisor_react", "label": "复杂", "color": "#faad14"},

        # 汇聚到 Judge
        {"id": "e12", "source": "logic_architect", "target": "judge"},
        {"id": "e13", "source": "parallel_executor", "target": "judge"},
        {"id": "e14", "source": "supervisor_react", "target": "judge"},

        # 校验和输出
        {"id": "e15", "source": "judge", "target": "viz_expert", "label": "通过"},
        {"id": "e16", "source": "judge", "target": "retry", "label": "失败"},
        {"id": "e17", "source": "retry", "target": "logic_architect", "label": "重试 (≤3次)"},
        {"id": "e18", "source": "retry", "target": "supervisor_react", "label": "智能修复"},
        {"id": "e19", "source": "viz_expert", "target": "end"},
    ]

    return {
        "success": True,
        "data": {
            "nodes": nodes,
            "edges": edges,
            "complexity_routes": COMPLEXITY_ROUTES,
        },
    }


@router.get("/complexity-routes")
async def get_complexity_routes():
    """
    获取复杂度路由说明

    返回三级复杂度的详细说明
    """
    return {
        "success": True,
        "data": COMPLEXITY_ROUTES,
    }


# ============================================
# Agent 配置 API
# ============================================

class AgentConfigRequest(BaseModel):
    """Agent 配置请求"""
    # 基础开关
    viz_expert_mode: Optional[str] = Field(None, description="VizExpert 模式: codeact, echarts, auto")
    ambi_resolver_enabled: Optional[bool] = Field(None, description="是否启用歧义消解")
    intent_classifier_enabled: Optional[bool] = Field(None, description="是否启用意图分类")
    supervisor_enabled: Optional[bool] = Field(None, description="是否启用 Supervisor")
    semantic_cache_enabled: Optional[bool] = Field(None, description="是否启用语义缓存")

    # DataSniper 配置
    data_sniper_use_graph_pruning: Optional[bool] = Field(None, description="启用图剪枝")
    data_sniper_use_vector_index: Optional[bool] = Field(None, description="启用向量索引")
    data_sniper_vector_index_threshold: Optional[int] = Field(None, description="向量索引阈值")
    data_sniper_schema_top_k: Optional[int] = Field(None, description="Schema 检索数量")
    data_sniper_schema_max_hops: Optional[int] = Field(None, description="关联跳数")
    data_sniper_use_llm_entity_extraction: Optional[bool] = Field(None, description="使用 LLM 实体提取")

    # Judge 配置
    judge_mysql_rows_examined_limit: Optional[int] = Field(None, description="MySQL 扫描行数限制")
    judge_mysql_cost_limit: Optional[int] = Field(None, description="MySQL 成本限制")
    judge_postgres_cost_limit: Optional[int] = Field(None, description="PostgreSQL 成本限制")
    judge_allow_full_scan: Optional[bool] = Field(None, description="允许全表扫描")
    judge_risk_threshold: Optional[float] = Field(None, description="风险阈值")
    judge_max_retries: Optional[int] = Field(None, description="最大重试次数")

    # LogicArchitect 配置
    logic_architect_use_optimized_modules: Optional[bool] = Field(None, description="使用优化模块")
    logic_architect_sql_candidates_count: Optional[int] = Field(None, description="SQL 候选数量")
    logic_architect_enable_self_correction: Optional[bool] = Field(None, description="启用自动修正")
    logic_architect_max_correction_rounds: Optional[int] = Field(None, description="最大修正轮数")

    # AmbiResolver 配置
    ambi_resolver_confidence_threshold: Optional[float] = Field(None, description="置信度阈值")
    ambi_resolver_max_options: Optional[int] = Field(None, description="最大选项数")

    # VizExpert 配置
    viz_expert_remove_nulls: Optional[bool] = Field(None, description="移除空值")
    viz_expert_remove_outliers: Optional[bool] = Field(None, description="移除异常值")
    viz_expert_outlier_threshold: Optional[float] = Field(None, description="异常值阈值")
    viz_expert_codeact_data_threshold: Optional[int] = Field(None, description="CodeAct 数据阈值")

    # 语义缓存配置
    cache_similarity_threshold: Optional[float] = Field(None, description="相似度阈值")
    cache_ttl_minutes: Optional[int] = Field(None, description="缓存有效期")

    # MultiJudge 配置
    multi_judge_enabled: Optional[bool] = Field(None, description="启用多 Judge")
    multi_judge_resolution: Optional[str] = Field(None, description="冲突解决策略")


@router.get("/config")
async def get_agent_config():
    """
    获取当前 Agent 配置

    返回所有 Agent 的配置状态
    """
    from ..config.settings import get_settings
    settings = get_settings()

    # 检查 E2B 是否可用
    e2b_available = bool(settings.e2b_api_key)

    return {
        "success": True,
        "data": {
            # VizExpert 配置
            "viz_expert": {
                "mode": settings.viz_expert_mode,
                "available_modes": [
                    {"value": "codeact", "label": "CodeAct (E2B 沙箱)", "description": "生成 Python 代码在 E2B 云沙箱执行", "requires_e2b": True},
                    {"value": "echarts", "label": "ECharts 直接生成", "description": "LLM 直接生成 ECharts JSON 配置", "requires_e2b": False},
                    {"value": "auto", "label": "自动选择", "description": "简单数据用 ECharts，复杂数据用 CodeAct", "requires_e2b": True},
                ],
                "e2b_available": e2b_available,
                "e2b_configured": e2b_available,
                "remove_nulls": settings.viz_expert_remove_nulls,
                "remove_outliers": settings.viz_expert_remove_outliers,
                "outlier_threshold": settings.viz_expert_outlier_threshold,
                "codeact_data_threshold": settings.viz_expert_codeact_data_threshold,
            },
            # 歧义消解配置
            "ambi_resolver": {
                "enabled": settings.ambi_resolver_enabled,
                "description": "歧义消解 - 检测并处理查询中的歧义",
                "confidence_threshold": settings.ambi_resolver_confidence_threshold,
                "max_options": settings.ambi_resolver_max_options,
            },
            # 意图分类配置
            "intent_classifier": {
                "enabled": settings.intent_classifier_enabled,
                "description": "意图分类 - 分析查询意图优化路由",
            },
            # Supervisor 配置
            "supervisor": {
                "enabled": settings.supervisor_enabled,
                "description": "任务协调器 - 分析复杂度并拆解任务",
            },
            # 语义缓存配置
            "semantic_cache": {
                "enabled": settings.semantic_cache_enabled,
                "description": "语义缓存 - 缓存相似查询结果",
                "similarity_threshold": settings.cache_similarity_threshold,
                "ttl_minutes": settings.cache_ttl_minutes,
            },
            # DataSniper 配置
            "data_sniper": {
                "use_graph_pruning": settings.data_sniper_use_graph_pruning,
                "use_vector_index": settings.data_sniper_use_vector_index,
                "vector_index_threshold": settings.data_sniper_vector_index_threshold,
                "schema_top_k": settings.data_sniper_schema_top_k,
                "schema_max_hops": settings.data_sniper_schema_max_hops,
                "use_llm_entity_extraction": settings.data_sniper_use_llm_entity_extraction,
                "description": "数据侦察 - Schema 剪枝、值映射、实体提取",
            },
            # Judge 配置
            "judge": {
                "mysql_rows_examined_limit": settings.judge_mysql_rows_examined_limit,
                "mysql_cost_limit": settings.judge_mysql_cost_limit,
                "postgres_cost_limit": settings.judge_postgres_cost_limit,
                "allow_full_scan": settings.judge_allow_full_scan,
                "risk_threshold": settings.judge_risk_threshold,
                "max_retries": settings.judge_max_retries,
                "description": "成本熔断 - SQL 校验、安全检查、执行计划分析",
            },
            # LogicArchitect 配置
            "logic_architect": {
                "use_optimized_modules": settings.logic_architect_use_optimized_modules,
                "sql_candidates_count": settings.logic_architect_sql_candidates_count,
                "enable_self_correction": settings.logic_architect_enable_self_correction,
                "max_correction_rounds": settings.logic_architect_max_correction_rounds,
                "complexity_threshold": settings.logic_architect_complexity_threshold,
                "description": "SQL 构建 - 基于 DSPy + MAC-SQL 生成多路 SQL",
            },
            # MultiJudge 配置
            "multi_judge": {
                "enabled": settings.multi_judge_enabled,
                "resolution": settings.multi_judge_resolution,
                "available_resolutions": [
                    {"value": "majority", "label": "多数决", "description": "超过半数 Judge 通过即可"},
                    {"value": "unanimous", "label": "全票通过", "description": "所有 Judge 必须通过"},
                    {"value": "weighted", "label": "加权投票", "description": "根据 Judge 权重计算"},
                    {"value": "llm_arbiter", "label": "LLM 仲裁", "description": "使用 LLM 进行最终裁决"},
                ],
                "rule_weight": settings.multi_judge_rule_weight,
                "cost_weight": settings.multi_judge_cost_weight,
                "semantic_weight": settings.multi_judge_semantic_weight,
                "description": "多 Judge 协调 - 协调多个 Judge 的评估和投票",
            },
            # 沙箱配置
            "sandbox": {
                "e2b_configured": e2b_available,
                "timeout_seconds": settings.sandbox_timeout_seconds,
            },
        },
    }


@router.put("/config")
async def update_agent_config(config: AgentConfigRequest):
    """
    更新 Agent 配置

    注意: 配置更新会在下次请求时生效
    部分配置需要重启服务才能完全生效
    """
    from ..config.settings import get_settings
    settings = get_settings()

    updated = {}
    warnings = []

    # 更新 VizExpert 模式
    if config.viz_expert_mode is not None:
        if config.viz_expert_mode not in ["codeact", "echarts", "auto"]:
            raise HTTPException(status_code=400, detail=f"无效的 viz_expert_mode: {config.viz_expert_mode}")

        # 检查 E2B 是否可用
        if config.viz_expert_mode in ["codeact", "auto"] and not settings.e2b_api_key:
            warnings.append("E2B API Key 未配置，CodeAct 模式将回退到 ECharts")

        # 动态更新配置 (注意: 这只在当前进程有效)
        settings.viz_expert_mode = config.viz_expert_mode
        updated["viz_expert_mode"] = config.viz_expert_mode

    # 更新基础开关配置
    if config.ambi_resolver_enabled is not None:
        settings.ambi_resolver_enabled = config.ambi_resolver_enabled
        updated["ambi_resolver_enabled"] = config.ambi_resolver_enabled

    if config.intent_classifier_enabled is not None:
        settings.intent_classifier_enabled = config.intent_classifier_enabled
        updated["intent_classifier_enabled"] = config.intent_classifier_enabled

    if config.supervisor_enabled is not None:
        settings.supervisor_enabled = config.supervisor_enabled
        updated["supervisor_enabled"] = config.supervisor_enabled

    if config.semantic_cache_enabled is not None:
        settings.semantic_cache_enabled = config.semantic_cache_enabled
        updated["semantic_cache_enabled"] = config.semantic_cache_enabled

    # 更新 DataSniper 配置
    if config.data_sniper_use_graph_pruning is not None:
        settings.data_sniper_use_graph_pruning = config.data_sniper_use_graph_pruning
        updated["data_sniper_use_graph_pruning"] = config.data_sniper_use_graph_pruning

    if config.data_sniper_use_vector_index is not None:
        settings.data_sniper_use_vector_index = config.data_sniper_use_vector_index
        updated["data_sniper_use_vector_index"] = config.data_sniper_use_vector_index

    if config.data_sniper_vector_index_threshold is not None:
        if not 10 <= config.data_sniper_vector_index_threshold <= 200:
            raise HTTPException(status_code=400, detail="向量索引阈值必须在 10-200 之间")
        settings.data_sniper_vector_index_threshold = config.data_sniper_vector_index_threshold
        updated["data_sniper_vector_index_threshold"] = config.data_sniper_vector_index_threshold

    if config.data_sniper_schema_top_k is not None:
        if not 3 <= config.data_sniper_schema_top_k <= 20:
            raise HTTPException(status_code=400, detail="Schema 检索数量必须在 3-20 之间")
        settings.data_sniper_schema_top_k = config.data_sniper_schema_top_k
        updated["data_sniper_schema_top_k"] = config.data_sniper_schema_top_k

    if config.data_sniper_schema_max_hops is not None:
        if not 1 <= config.data_sniper_schema_max_hops <= 3:
            raise HTTPException(status_code=400, detail="关联跳数必须在 1-3 之间")
        settings.data_sniper_schema_max_hops = config.data_sniper_schema_max_hops
        updated["data_sniper_schema_max_hops"] = config.data_sniper_schema_max_hops

    if config.data_sniper_use_llm_entity_extraction is not None:
        settings.data_sniper_use_llm_entity_extraction = config.data_sniper_use_llm_entity_extraction
        updated["data_sniper_use_llm_entity_extraction"] = config.data_sniper_use_llm_entity_extraction

    # 更新 Judge 配置
    if config.judge_mysql_rows_examined_limit is not None:
        if config.judge_mysql_rows_examined_limit < 10000:
            raise HTTPException(status_code=400, detail="MySQL 扫描行数限制不能小于 10000")
        settings.judge_mysql_rows_examined_limit = config.judge_mysql_rows_examined_limit
        updated["judge_mysql_rows_examined_limit"] = config.judge_mysql_rows_examined_limit

    if config.judge_mysql_cost_limit is not None:
        if config.judge_mysql_cost_limit < 10000:
            raise HTTPException(status_code=400, detail="MySQL 成本限制不能小于 10000")
        settings.judge_mysql_cost_limit = config.judge_mysql_cost_limit
        updated["judge_mysql_cost_limit"] = config.judge_mysql_cost_limit

    if config.judge_postgres_cost_limit is not None:
        if config.judge_postgres_cost_limit < 10000:
            raise HTTPException(status_code=400, detail="PostgreSQL 成本限制不能小于 10000")
        settings.judge_postgres_cost_limit = config.judge_postgres_cost_limit
        updated["judge_postgres_cost_limit"] = config.judge_postgres_cost_limit

    if config.judge_allow_full_scan is not None:
        settings.judge_allow_full_scan = config.judge_allow_full_scan
        updated["judge_allow_full_scan"] = config.judge_allow_full_scan
        if config.judge_allow_full_scan:
            warnings.append("允许全表扫描可能导致性能问题")

    if config.judge_risk_threshold is not None:
        if not 0.5 <= config.judge_risk_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="风险阈值必须在 0.5-1.0 之间")
        settings.judge_risk_threshold = config.judge_risk_threshold
        updated["judge_risk_threshold"] = config.judge_risk_threshold

    if config.judge_max_retries is not None:
        if not 1 <= config.judge_max_retries <= 5:
            raise HTTPException(status_code=400, detail="最大重试次数必须在 1-5 之间")
        settings.judge_max_retries = config.judge_max_retries
        updated["judge_max_retries"] = config.judge_max_retries

    # 更新 LogicArchitect 配置
    if config.logic_architect_use_optimized_modules is not None:
        settings.logic_architect_use_optimized_modules = config.logic_architect_use_optimized_modules
        updated["logic_architect_use_optimized_modules"] = config.logic_architect_use_optimized_modules

    if config.logic_architect_sql_candidates_count is not None:
        if not 1 <= config.logic_architect_sql_candidates_count <= 5:
            raise HTTPException(status_code=400, detail="SQL 候选数量必须在 1-5 之间")
        settings.logic_architect_sql_candidates_count = config.logic_architect_sql_candidates_count
        updated["logic_architect_sql_candidates_count"] = config.logic_architect_sql_candidates_count

    if config.logic_architect_enable_self_correction is not None:
        settings.logic_architect_enable_self_correction = config.logic_architect_enable_self_correction
        updated["logic_architect_enable_self_correction"] = config.logic_architect_enable_self_correction

    if config.logic_architect_max_correction_rounds is not None:
        if not 1 <= config.logic_architect_max_correction_rounds <= 3:
            raise HTTPException(status_code=400, detail="最大修正轮数必须在 1-3 之间")
        settings.logic_architect_max_correction_rounds = config.logic_architect_max_correction_rounds
        updated["logic_architect_max_correction_rounds"] = config.logic_architect_max_correction_rounds

    # 更新 AmbiResolver 配置
    if config.ambi_resolver_confidence_threshold is not None:
        if not 0.3 <= config.ambi_resolver_confidence_threshold <= 0.9:
            raise HTTPException(status_code=400, detail="置信度阈值必须在 0.3-0.9 之间")
        settings.ambi_resolver_confidence_threshold = config.ambi_resolver_confidence_threshold
        updated["ambi_resolver_confidence_threshold"] = config.ambi_resolver_confidence_threshold

    if config.ambi_resolver_max_options is not None:
        if not 3 <= config.ambi_resolver_max_options <= 10:
            raise HTTPException(status_code=400, detail="最大选项数必须在 3-10 之间")
        settings.ambi_resolver_max_options = config.ambi_resolver_max_options
        updated["ambi_resolver_max_options"] = config.ambi_resolver_max_options

    # 更新 VizExpert 配置
    if config.viz_expert_remove_nulls is not None:
        settings.viz_expert_remove_nulls = config.viz_expert_remove_nulls
        updated["viz_expert_remove_nulls"] = config.viz_expert_remove_nulls

    if config.viz_expert_remove_outliers is not None:
        settings.viz_expert_remove_outliers = config.viz_expert_remove_outliers
        updated["viz_expert_remove_outliers"] = config.viz_expert_remove_outliers

    if config.viz_expert_outlier_threshold is not None:
        if not 1.5 <= config.viz_expert_outlier_threshold <= 5.0:
            raise HTTPException(status_code=400, detail="异常值阈值必须在 1.5-5.0 之间")
        settings.viz_expert_outlier_threshold = config.viz_expert_outlier_threshold
        updated["viz_expert_outlier_threshold"] = config.viz_expert_outlier_threshold

    if config.viz_expert_codeact_data_threshold is not None:
        if not 10 <= config.viz_expert_codeact_data_threshold <= 1000:
            raise HTTPException(status_code=400, detail="CodeAct 数据阈值必须在 10-1000 之间")
        settings.viz_expert_codeact_data_threshold = config.viz_expert_codeact_data_threshold
        updated["viz_expert_codeact_data_threshold"] = config.viz_expert_codeact_data_threshold

    # 更新语义缓存配置
    if config.cache_similarity_threshold is not None:
        if not 0.7 <= config.cache_similarity_threshold <= 0.95:
            raise HTTPException(status_code=400, detail="相似度阈值必须在 0.7-0.95 之间")
        settings.cache_similarity_threshold = config.cache_similarity_threshold
        updated["cache_similarity_threshold"] = config.cache_similarity_threshold

    if config.cache_ttl_minutes is not None:
        if not 5 <= config.cache_ttl_minutes <= 1440:
            raise HTTPException(status_code=400, detail="缓存有效期必须在 5-1440 分钟之间")
        settings.cache_ttl_minutes = config.cache_ttl_minutes
        updated["cache_ttl_minutes"] = config.cache_ttl_minutes

    # 更新 MultiJudge 配置
    if config.multi_judge_enabled is not None:
        settings.multi_judge_enabled = config.multi_judge_enabled
        updated["multi_judge_enabled"] = config.multi_judge_enabled

    if config.multi_judge_resolution is not None:
        if config.multi_judge_resolution not in ["majority", "unanimous", "weighted", "llm_arbiter"]:
            raise HTTPException(status_code=400, detail="无效的冲突解决策略")
        settings.multi_judge_resolution = config.multi_judge_resolution
        updated["multi_judge_resolution"] = config.multi_judge_resolution

    return {
        "success": True,
        "message": "配置已更新",
        "updated": updated,
        "warnings": warnings,
        "note": "部分配置可能需要重启服务才能完全生效",
    }


@router.get("/config/e2b-status")
async def get_e2b_status():
    """
    获取 E2B 沙箱状态

    检查 E2B 是否配置正确并可用
    """
    from ..config.settings import get_settings
    settings = get_settings()

    status = {
        "configured": bool(settings.e2b_api_key),
        "api_key_set": bool(settings.e2b_api_key),
        "timeout_seconds": settings.sandbox_timeout_seconds,
        "available": False,
        "error": None,
    }

    if settings.e2b_api_key:
        # 尝试测试 E2B 连接
        try:
            from e2b_code_interpreter import AsyncSandbox
            status["sdk_installed"] = True
            status["available"] = True
        except ImportError as e:
            status["sdk_installed"] = False
            status["error"] = f"E2B SDK 未安装: {str(e)}"

    return {
        "success": True,
        "data": status,
    }


__all__ = [
    "router",
    "register_active_query",
    "update_active_query",
    "remove_active_query",
]
