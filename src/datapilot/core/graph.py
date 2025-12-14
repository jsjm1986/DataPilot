# -*- coding: utf-8 -*-
"""
DataPilot LangGraph Workflow (LangGraph 1.x 优化版)

使用 LangGraph 1.x 新特性:
1. interrupt() - Human-in-the-loop 原生支持
2. Store - 跨会话用户偏好记忆
3. astream_events() - 实时事件流
4. Command - 动态流程控制
5. Semantic Cache - 语义缓存 (P0 新增)
"""

import os
from typing import Literal, Optional, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.store.memory import InMemoryStore

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None

from .state import DataPilotState, ClarifyOption
from .supervisor import Supervisor, supervisor_node, route_after_supervisor
from .parallel_executor import ParallelExecutor, parallel_executor_node
from .react_agent import ReActAgent, react_agent_node
from .supervisor_react import SupervisorReAct, supervisor_react_node
from ..agents import DataSniper, LogicArchitect, Judge, VizExpert, AmbiResolver
from ..agents.intent_classifier import IntentClassifier, classify_intent
from ..db.connector import get_db_manager
from ..config.settings import get_settings
from ..cache.semantic_cache import get_cache, SemanticCache


# ============================================
# 全局 Store (用户偏好记忆)
# ============================================
_store = InMemoryStore()


def get_store() -> InMemoryStore:
    """获取全局 Store 实例"""
    return _store


# ============================================
# 用户偏好管理
# ============================================
class UserPreferences:
    """用户偏好管理器 - 使用 LangGraph Store"""

    @staticmethod
    async def get_preference(user_id: str, key: str) -> Optional[Any]:
        """获取用户偏好"""
        store = get_store()
        namespace = ("user_preferences", user_id)
        try:
            item = store.get(namespace, key)
            return item.value if item else None
        except Exception:
            return None

    @staticmethod
    async def set_preference(user_id: str, key: str, value: Any):
        """设置用户偏好"""
        store = get_store()
        namespace = ("user_preferences", user_id)
        store.put(namespace, key, value)

    @staticmethod
    async def get_query_history(user_id: str, limit: int = 10) -> list[dict]:
        """获取用户查询历史"""
        store = get_store()
        namespace = ("query_history", user_id)
        try:
            items = store.search(namespace)
            return [item.value for item in items][:limit]
        except Exception:
            return []

    @staticmethod
    async def add_query_history(user_id: str, query: str, sql: str):
        """添加查询历史"""
        store = get_store()
        namespace = ("query_history", user_id)
        key = datetime.utcnow().isoformat()
        store.put(namespace, key, {"query": query, "sql": sql, "timestamp": key})


# ============================================
# Agent 执行时间追踪辅助函数
# ============================================

import time

def record_agent_timing(state: DataPilotState, agent_id: str, start_time: float) -> dict:
    """记录 Agent 执行时间到 state.agent_timings"""
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    # 获取现有的 agent_timings 或创建新的
    agent_timings = dict(state.get("agent_timings", {}))
    agent_timings[agent_id] = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_ms": duration_ms,
    }
    return {"agent_timings": agent_timings}


# ============================================
# 节点函数 (使用 LangGraph 1.x 新特性)
# ============================================

async def cache_check_node(state: DataPilotState) -> dict:
    """
    缓存检查节点 - Semantic Cache (P0 新增)

    在工作流入口处检查语义缓存:
    1. 精确匹配 (哈希)
    2. 语义匹配 (向量相似度 >= 0.85)

    如果命中缓存，直接跳转到结果输出
    """
    start_time = time.time()
    query = state["query"]
    database = state.get("database", "default")

    try:
        cache = get_cache()
        cache_hit = await cache.get(query, database, use_semantic=True)

        if cache_hit:
            # 缓存命中! 直接返回结果
            timing = record_agent_timing(state, "cache_check", start_time)
            return {
                "winner_sql": cache_hit.sql,
                "execution_result": {
                    "success": True,
                    "data": cache_hit.data,
                    "row_count": cache_hit.row_count,
                    "columns": list(cache_hit.data[0].keys()) if cache_hit.data else [],
                    "execution_time_ms": 0,
                    "error": None,
                },
                "cache_hit": True,
                "cache_similarity": cache_hit.similarity,
                "current_agent": "cache_check",
                "next_agent": "viz_expert",  # 跳过 SQL 生成，直接可视化
                **timing,
            }
    except Exception as e:
        # 缓存检查失败，继续正常流程
        print(f"Cache check error: {e}")

    # 缓存未命中，继续正常流程
    timing = record_agent_timing(state, "cache_check", start_time)
    return {
        "cache_hit": False,
        "cache_similarity": 0.0,
        "current_agent": "cache_check",
        "next_agent": "intent_classifier",
        **timing,
    }


async def intent_classifier_node(state: DataPilotState) -> dict:
    """
    意图分类节点 - 使用 LLM 分析查询意图以优化后续路由

    在 ambi_resolver 之前执行，为后续 Agent 提供意图信息。
    使用 LLM 进行语义理解，而非硬编码关键词匹配。
    """
    start_time = time.time()
    query = state["query"]

    # 使用 LLM 驱动的 IntentClassifier 分类查询意图
    classifier = IntentClassifier()
    intent_result = await classifier.classify_async(query)

    # SQL 提示已包含在 intent_result 中
    sql_hints = classifier.get_sql_hints(intent_result)

    timing = record_agent_timing(state, "intent_classifier", start_time)
    return {
        "current_agent": "intent_classifier",
        "next_agent": "ambi_resolver",
        "intent_result": {
            "primary_intent": intent_result.primary_intent.value,
            "confidence": intent_result.confidence,
            "secondary_intents": [i.value for i in intent_result.secondary_intents],
            "features": intent_result.features,
            "routing_hint": intent_result.routing_hint,
            "sql_hints": sql_hints,
        },
        **timing,
    }


async def ambi_resolver_node(state: DataPilotState) -> dict:
    """
    歧义消解节点 - 使用 interrupt() 实现 Human-in-the-loop

    当检测到歧义时，使用 interrupt() 暂停工作流等待用户输入
    """
    start_time = time.time()
    query = state["query"]
    database = state.get("database", "default")
    user_id = state.get("user_id", "anonymous")

    print(f"[AmbiResolver] Starting for query: {query}")

    # 检查用户历史偏好
    time_pref = await UserPreferences.get_preference(user_id, "default_time_range")
    print(f"[AmbiResolver] User time preference: {time_pref}")

    resolver = AmbiResolver()
    print(f"[AmbiResolver] Calling detect_ambiguity...")
    # detect_ambiguity 返回元组 (ClarifyOption, AmbiguityResult)
    result = await resolver.detect_ambiguity(query, database)
    print(f"[AmbiResolver] detect_ambiguity returned: {result}")
    clarify_option, ambiguity_result = result if isinstance(result, tuple) else (result, None)

    # ClarifyOption 可能是 dict 或 Pydantic model
    clarify_dict = clarify_option if isinstance(clarify_option, dict) else (
        clarify_option.model_dump() if hasattr(clarify_option, 'model_dump') else
        clarify_option.__dict__ if clarify_option else None
    )

    if clarify_dict and clarify_dict.get("options"):
        # 如果有用户偏好且匹配当前歧义类型，自动应用
        if time_pref and "时间" in clarify_dict.get("question", ""):
            resolved_query = await resolver.resolve_with_selection(query, time_pref)
            timing = record_agent_timing(state, "ambi_resolver", start_time)
            return {
                "query": resolved_query,
                "current_agent": "ambi_resolver",
                "next_agent": "data_sniper",
                "clarify_needed": False,
                **timing,
            }

        # 直接返回澄清请求，让 WebSocket 处理
        # 不使用 interrupt()，因为 WebSocket 需要从 agent_end 事件中检测
        print(f"[AmbiResolver] Returning clarify_needed with options: {clarify_dict['options']}")
        timing = record_agent_timing(state, "ambi_resolver", start_time)
        return {
            "current_agent": "ambi_resolver",
            "next_agent": None,  # 暂停等待用户输入
            "clarify_needed": True,
            "clarify_options": {
                "question": clarify_dict["question"],
                "options": clarify_dict["options"],
            },
            **timing,
        }

    # 无歧义，继续流程
    print(f"[AmbiResolver] No ambiguity detected, continuing to data_sniper")
    timing = record_agent_timing(state, "ambi_resolver", start_time)
    return {
        "current_agent": "ambi_resolver",
        "next_agent": "data_sniper",
        "clarify_needed": False,
        **timing,
    }


async def data_sniper_node(state: DataPilotState) -> dict:
    """Data Sniper 节点 - Schema 剪枝和值映射 (LLM-Native 版)"""
    start_time = time.time()
    database = state.get("database", "default")
    query = state["query"]

    sniper = DataSniper(database)
    result = await sniper.analyze(query)

    timing = record_agent_timing(state, "data_sniper", start_time)
    return {
        "schema_context": result.get("schema", ""),
        "relevant_tables": [t["name"] for t in result.get("relevant_tables", [])],
        "value_mappings": result.get("value_mappings", {}),
        # LLM-Native 上下文 (来自 SchemaIntrospector)
        "time_context": result.get("time_context", ""),
        "enum_context": result.get("enum_context", ""),
        "current_agent": "data_sniper",
        "next_agent": "logic_architect",
        **timing,
    }


async def logic_architect_node(state: DataPilotState) -> dict:
    """Logic Architect 节点 - SQL 生成 (LLM-Native 版)"""
    start_time = time.time()
    architect = LogicArchitect()

    schema = state.get("schema_context", "")
    if not schema:
        db_manager = get_db_manager()
        connector = db_manager.get(state.get("database", "default"))
        schema = await connector.get_schema()

    # 获取 LLM-Native 上下文 (来自 DataSniper 的 SchemaIntrospector)
    time_context = state.get("time_context", "")
    enum_context = state.get("enum_context", "")

    candidates = await architect.generate_sql(
        query=state["query"],
        schema=schema,
        database=state.get("database", "default"),
        value_mappings=state.get("value_mappings"),
        # LLM-Native 上下文
        time_context=time_context,
        enum_context=enum_context,
        schema_context=schema if time_context or enum_context else None,
    )

    # generate_sql 返回的是候选列表，取第一个作为 winner
    winner = candidates[0] if candidates else None
    winner_sql = winner.get("sql") if winner else None

    timing = record_agent_timing(state, "logic_architect", start_time)
    return {
        "candidates": state.get("candidates", []) + candidates,
        "winner_sql": winner_sql,
        "current_agent": "logic_architect",
        "next_agent": "judge",
        **timing,
    }


async def judge_node(state: DataPilotState) -> dict:
    """Judge 节点 - SQL 校验和执行"""
    start_time = time.time()
    database = state.get("database", "default")
    judge = Judge(database)

    sql = state.get("winner_sql")
    if not sql:
        timing = record_agent_timing(state, "judge", start_time)
        return {
            "current_agent": "judge",
            "next_agent": END,
            "last_error": "No SQL to validate",
            **timing,
        }

    # 校验 SQL
    result = await judge.judge(sql)

    if not result.get("approved"):
        retries = state.get("retries", {})
        architect_retries = retries.get("architect", 0)

        if architect_retries < 3:
            retries["architect"] = architect_retries + 1
            timing = record_agent_timing(state, "judge", start_time)
            return {
                "current_agent": "judge",
                "next_agent": "logic_architect",
                "retries": retries,
                "error_context": result.get("reason"),
                "last_error": result.get("reason"),
                **timing,
            }
        else:
            # 超过重试次数，触发 Supervisor ReAct 模式进行智能修复
            # 而不是直接请求人工介入
            timing = record_agent_timing(state, "judge", start_time)
            return {
                "current_agent": "judge",
                "next_agent": "supervisor_react",
                "supervisor_react_trigger": "judge_failed",
                "retries": retries,
                "last_error": result.get("reason"),
                **timing,
            }

    # 执行 SQL
    db_manager = get_db_manager()
    connector = db_manager.get(database)

    try:
        data = await connector.execute_query(sql, limit=100)

        # 保存成功的查询到历史
        user_id = state.get("user_id", "anonymous")
        await UserPreferences.add_query_history(user_id, state["query"], sql)

        # 保存到语义缓存 (P0 新增)
        try:
            cache = get_cache()
            await cache.set(
                query=state["query"],
                sql=sql,
                result=data,
                row_count=len(data),
                database=database,
            )
        except Exception as cache_err:
            print(f"Cache set error: {cache_err}")

        timing = record_agent_timing(state, "judge", start_time)
        return {
            "execution_result": {
                "success": True,
                "data": data,
                "row_count": len(data),
                "columns": list(data[0].keys()) if data else [],
                "execution_time_ms": 0,
                "error": None,
            },
            "current_agent": "judge",
            "next_agent": "viz_expert",
            **timing,
        }
    except Exception as e:
        timing = record_agent_timing(state, "judge", start_time)
        return {
            "execution_result": {
                "success": False,
                "data": [],
                "row_count": 0,
                "columns": [],
                "execution_time_ms": 0,
                "error": str(e),
            },
            "current_agent": "judge",
            "next_agent": END,
            "last_error": str(e),
            **timing,
        }


async def viz_expert_node(state: DataPilotState) -> dict:
    """Viz Expert 节点 - 图表推荐"""
    start_time = time.time()
    
    # 从 state 获取可视化模式
    viz_mode = state.get("viz_mode", "echarts")
    viz = VizExpert(mode=viz_mode)

    exec_result = state.get("execution_result", {})
    data = exec_result.get("data", [])

    if not data:
        timing = record_agent_timing(state, "viz_expert", start_time)
        return {
            "current_agent": "viz_expert",
            "next_agent": END,
            **timing,
        }

    # analyze 是异步方法，需要 await
    # force_mode 确保使用用户选择的模式
    analysis = await viz.analyze(data, state["query"], force_mode=viz_mode)

    timing = record_agent_timing(state, "viz_expert", start_time)
    return {
        "chart_config": {
            "chart_type": analysis.get("chart_type"),
            "title": state["query"][:50],
            "config": analysis.get("echarts_config"),  # 修复: 使用 echarts_config
        },
        "python_code": analysis.get("python_code"),
        "sandbox_result": analysis.get("sandbox_result"),
        "current_agent": "viz_expert",
        "next_agent": END,
        **timing,
    }


# ============================================
# 路由函数
# ============================================

def route_after_intent(state: DataPilotState) -> Literal["ambi_resolver", "data_sniper"]:
    """意图分类后的路由"""
    # 根据意图结果决定是否需要歧义消解
    intent_result = state.get("intent_result", {})
    primary_intent = intent_result.get("primary_intent", "unknown")

    # 简单查找类查询可以跳过歧义消解
    if primary_intent == "lookup" and intent_result.get("confidence", 0) > 0.8:
        return "data_sniper"

    return "ambi_resolver"


def route_after_ambi(state: DataPilotState) -> Literal["data_sniper", "supervisor", "__end__"]:
    """歧义消解后的路由"""
    if state.get("clarify_needed"):
        return "__end__"
    # 默认返回 supervisor，如果没有启用则由 build_graph 处理
    return "supervisor"


def route_after_sniper(state: DataPilotState) -> Literal["logic_architect", "__end__"]:
    """Data Sniper 后的路由"""
    return state.get("next_agent", "logic_architect")


def route_after_architect(state: DataPilotState) -> Literal["judge", "__end__"]:
    """Logic Architect 后的路由"""
    return state.get("next_agent", "judge")


def route_after_judge(state: DataPilotState) -> Literal["logic_architect", "viz_expert", "supervisor_react", "__end__"]:
    """Judge 后的路由"""
    next_agent = state.get("next_agent")

    if next_agent == END or next_agent == "__end__":
        return "__end__"

    if state.get("human_handoff"):
        return "__end__"

    if next_agent == "logic_architect":
        return "logic_architect"

    # 新增: 失败后触发 Supervisor ReAct
    if next_agent == "supervisor_react":
        return "supervisor_react"

    return "viz_expert"


def route_after_viz(state: DataPilotState) -> Literal["__end__"]:
    """Viz Expert 后的路由"""
    return "__end__"


def route_after_cache_check(state: DataPilotState) -> Literal["viz_expert", "intent_classifier"]:
    """缓存检查后的路由"""
    if state.get("cache_hit"):
        return "viz_expert"  # 缓存命中，直接可视化
    return "intent_classifier"  # 缓存未命中，继续正常流程


# ============================================
# 构建工作流图
# ============================================

def build_graph(
    include_ambi_resolver: bool = True,
    include_intent_classifier: bool = True,
    include_supervisor: bool = True,
    include_react_agent: bool = True,
    include_semantic_cache: bool = True,
) -> StateGraph:
    """
    构建 DataPilot 工作流图 (LLM-Native 版 + ReAct Agent + Semantic Cache)

    Args:
        include_ambi_resolver: 是否包含歧义消解节点
        include_intent_classifier: 是否包含意图分类节点
        include_supervisor: 是否包含 Supervisor 任务拆解
        include_react_agent: 是否包含 ReAct Agent (复杂推理)
        include_semantic_cache: 是否包含语义缓存 (P0 新增)

    Returns:
        StateGraph 实例

    工作流结构 (含缓存):
        CacheCheck → [命中] → Viz
        CacheCheck → [未命中] → Intent → Ambi → Supervisor → ...

    简单查询: Intent -> Ambi -> Supervisor -> DataSniper -> Logic -> Judge -> Viz
    中等复杂: Intent -> Ambi -> Supervisor -> ParallelExecutor -> Viz
    高度复杂: Intent -> Ambi -> Supervisor -> ReActAgent -> Viz
    """
    workflow = StateGraph(DataPilotState)

    # ============================================
    # 添加所有节点
    # ============================================
    # 缓存检查节点 (P0 新增)
    if include_semantic_cache:
        workflow.add_node("cache_check", cache_check_node)

    if include_intent_classifier:
        workflow.add_node("intent_classifier", intent_classifier_node)
    if include_ambi_resolver:
        workflow.add_node("ambi_resolver", ambi_resolver_node)
    if include_supervisor:
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("parallel_executor", parallel_executor_node)
    if include_react_agent:
        workflow.add_node("react_agent", react_agent_node)

    # 新增: Supervisor ReAct 节点 (用于失败回退和复杂查询)
    workflow.add_node("supervisor_react", supervisor_react_node)

    workflow.add_node("data_sniper", data_sniper_node)
    workflow.add_node("logic_architect", logic_architect_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("viz_expert", viz_expert_node)

    # ============================================
    # 设置入口点和边
    # ============================================
    # 缓存检查作为入口点 (P0 新增)
    if include_semantic_cache:
        workflow.set_entry_point("cache_check")
        if include_intent_classifier:
            workflow.add_conditional_edges("cache_check", route_after_cache_check, {
                "viz_expert": "viz_expert",
                "intent_classifier": "intent_classifier",
            })
            # 添加 intent_classifier 后续的边
            if include_ambi_resolver:
                workflow.add_conditional_edges("intent_classifier", route_after_intent, {
                    "ambi_resolver": "ambi_resolver",
                    "data_sniper": "data_sniper",
                })
                if include_supervisor:
                    workflow.add_conditional_edges("ambi_resolver", route_after_ambi, {
                        "supervisor": "supervisor",
                        "__end__": END,
                    })
                else:
                    workflow.add_conditional_edges("ambi_resolver", route_after_ambi, {
                        "data_sniper": "data_sniper",
                        "__end__": END,
                    })
            else:
                if include_supervisor:
                    workflow.add_edge("intent_classifier", "supervisor")
                else:
                    workflow.add_edge("intent_classifier", "data_sniper")
        else:
            workflow.add_conditional_edges("cache_check", route_after_cache_check, {
                "viz_expert": "viz_expert",
                "intent_classifier": "data_sniper",  # 没有 intent_classifier 则跳到 data_sniper
            })
    elif include_intent_classifier:
        workflow.set_entry_point("intent_classifier")
        if include_ambi_resolver:
            workflow.add_conditional_edges("intent_classifier", route_after_intent, {
                "ambi_resolver": "ambi_resolver",
                "data_sniper": "data_sniper",
            })
            if include_supervisor:
                workflow.add_conditional_edges("ambi_resolver", route_after_ambi, {
                    "supervisor": "supervisor",
                    "__end__": END,
                })
            else:
                workflow.add_conditional_edges("ambi_resolver", route_after_ambi, {
                    "data_sniper": "data_sniper",
                    "__end__": END,
                })
        else:
            if include_supervisor:
                workflow.add_edge("intent_classifier", "supervisor")
            else:
                workflow.add_edge("intent_classifier", "data_sniper")
    elif include_ambi_resolver:
        workflow.set_entry_point("ambi_resolver")
        if include_supervisor:
            workflow.add_conditional_edges("ambi_resolver", route_after_ambi, {
                "supervisor": "supervisor",
                "__end__": END,
            })
        else:
            workflow.add_conditional_edges("ambi_resolver", route_after_ambi, {
                "data_sniper": "data_sniper",
                "__end__": END,
            })
    elif include_supervisor:
        workflow.set_entry_point("supervisor")
    else:
        workflow.set_entry_point("data_sniper")

    # ============================================
    # Supervisor 路由 (核心: 简单/中等/复杂查询分流)
    # ============================================
    if include_supervisor:
        if include_react_agent:
            # 三级路由: 简单 -> 串行, 中等 -> 并行, 复杂 -> Supervisor ReAct
            workflow.add_conditional_edges("supervisor", _route_after_supervisor_with_react, {
                "data_sniper": "data_sniper",           # 简单查询 -> 串行流程
                "parallel_executor": "parallel_executor", # 中等复杂 -> 并行执行
                "supervisor_react": "supervisor_react",   # 高复杂度 -> Supervisor ReAct
            })

            # ReAct Agent 完成后 -> 可视化
            workflow.add_conditional_edges("react_agent", _route_after_react, {
                "viz_expert": "viz_expert",
                "__end__": END,
            })
        else:
            workflow.add_conditional_edges("supervisor", _route_after_supervisor, {
                "data_sniper": "data_sniper",
                "parallel_executor": "parallel_executor",
            })

        # 并行执行器完成后 -> 可视化
        workflow.add_conditional_edges("parallel_executor", _route_after_parallel, {
            "viz_expert": "viz_expert",
            "__end__": END,
        })

    # ============================================
    # 串行流程边
    # ============================================
    workflow.add_conditional_edges("data_sniper", route_after_sniper, {
        "logic_architect": "logic_architect",
        "__end__": END,
    })

    workflow.add_conditional_edges("logic_architect", route_after_architect, {
        "judge": "judge",
        "__end__": END,
    })

    workflow.add_conditional_edges("judge", route_after_judge, {
        "logic_architect": "logic_architect",
        "viz_expert": "viz_expert",
        "supervisor_react": "supervisor_react",  # 新增: 失败回退到 Supervisor ReAct
        "__end__": END,
    })

    # 新增: Supervisor ReAct 后的路由
    workflow.add_conditional_edges("supervisor_react", _route_after_supervisor_react, {
        "viz_expert": "viz_expert",
        "__end__": END,
    })

    workflow.add_conditional_edges("viz_expert", route_after_viz, {
        "__end__": END,
    })

    return workflow


def _route_after_supervisor(state: DataPilotState) -> Literal["data_sniper", "parallel_executor"]:
    """Supervisor 后的路由 - 根据任务复杂度分流 (双路)"""
    task_plan = state.get("task_plan")

    if task_plan and task_plan.get("needs_decomposition"):
        # 复杂查询，进入并行执行
        return "parallel_executor"

    # 简单查询，继续串行流程
    return "data_sniper"


def _route_after_supervisor_with_react(state: DataPilotState) -> Literal["data_sniper", "parallel_executor", "supervisor_react"]:
    """Supervisor 后的路由 - 三级分流 (简单/中等/复杂)

    改进: 复杂查询使用 Supervisor ReAct 而不是独立的 ReAct Agent
    """
    task_plan = state.get("task_plan")

    if not task_plan:
        return "data_sniper"

    # 高复杂度 -> Supervisor ReAct (替代独立 ReAct Agent)
    # 判断条件:
    # 1. 任务计划明确标记为高复杂度
    # 2. 或者需要多轮推理 (多步依赖)
    complexity = task_plan.get("complexity", "simple")
    needs_reasoning = task_plan.get("needs_reasoning", False)

    if complexity == "high" or needs_reasoning:
        return "supervisor_react"

    # 中等复杂度 -> 并行执行
    if task_plan.get("needs_decomposition"):
        return "parallel_executor"

    # 简单查询 -> 串行流程
    return "data_sniper"


def _route_after_supervisor_react(state: DataPilotState) -> Literal["viz_expert", "__end__"]:
    """Supervisor ReAct 后的路由"""
    execution_result = state.get("execution_result")

    if execution_result and execution_result.get("success"):
        return "viz_expert"

    return "__end__"


def _route_after_react(state: DataPilotState) -> Literal["viz_expert", "__end__"]:
    """ReAct Agent 后的路由"""
    execution_result = state.get("execution_result")

    if execution_result and execution_result.get("success"):
        return "viz_expert"

    return "__end__"


def _route_after_parallel(state: DataPilotState) -> Literal["viz_expert", "__end__"]:
    """并行执行器后的路由"""
    execution_result = state.get("execution_result")

    if execution_result and execution_result.get("success"):
        return "viz_expert"

    return "__end__"


# ============================================
# 配置和编译
# ============================================

def _setup_langsmith():
    """配置 LangSmith 追踪"""
    settings = get_settings()
    langsmith_key = getattr(settings, 'langsmith_api_key', None)
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = getattr(settings, 'langsmith_endpoint', 'https://api.smith.langchain.com')
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = getattr(settings, 'langsmith_project', 'datapilot')


def _get_checkpointer():
    """获取检查点存储器"""
    settings = get_settings()
    checkpointer_url = getattr(settings, 'checkpointer_url', None)

    # 优先使用 SQLite (持久化)
    if SqliteSaver:
        try:
            return SqliteSaver.from_conn_string("data/checkpoints.db")
        except Exception:
            pass

    # 回退到内存
    return MemorySaver()


def get_compiled_graph(
    include_ambi_resolver: bool = True,
    include_semantic_cache: bool = True,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
):
    """
    获取编译后的工作流

    Args:
        include_ambi_resolver: 是否包含歧义消解
        include_semantic_cache: 是否包含语义缓存 (P0 新增)
        interrupt_before: 在这些节点前中断
        interrupt_after: 在这些节点后中断

    Returns:
        编译后的 CompiledGraph
    """
    _setup_langsmith()
    workflow = build_graph(
        include_ambi_resolver=include_ambi_resolver,
        include_semantic_cache=include_semantic_cache,
    )
    checkpointer = _get_checkpointer()
    store = get_store()

    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )


# ============================================
# 流式执行辅助函数
# ============================================

def _safe_serialize(obj):
    """安全序列化对象，处理不可序列化的类型"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    # 处理 LangGraph Interrupt 对象
    if hasattr(obj, 'value'):
        return _safe_serialize(obj.value)
    if hasattr(obj, '__dict__'):
        return _safe_serialize(obj.__dict__)
    # 其他类型转为字符串
    return str(obj)


async def run_with_streaming(
    query: str,
    database: str = "default",
    user_id: str = "anonymous",
    tenant_id: str = "default",
    thread_id: Optional[str] = None,
    skip_ambi_resolver: bool = False,
):
    """
    使用 astream_events() 流式执行查询

    Args:
        skip_ambi_resolver: 是否跳过歧义检测（用于澄清后的查询）

    Yields:
        事件字典，包含进度更新和最终结果
    """
    from .state import create_initial_state

    graph = get_compiled_graph(include_ambi_resolver=not skip_ambi_resolver)
    initial_state = create_initial_state(
        query=query,
        user_id=user_id,
        tenant_id=tenant_id,
        database=database,
    )

    config = {"configurable": {"thread_id": thread_id or initial_state["trace_id"]}}

    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        event_type = event.get("event")

        if event_type == "on_chain_start":
            yield {
                "type": "agent_start",
                "agent": event.get("name"),
                "timestamp": datetime.utcnow().isoformat(),
            }

        elif event_type == "on_chain_end":
            # 安全序列化输出，处理 Interrupt 等不可序列化对象
            raw_output = event.get("data", {}).get("output")
            safe_output = _safe_serialize(raw_output)
            yield {
                "type": "agent_end",
                "agent": event.get("name"),
                "output": safe_output,
                "timestamp": datetime.utcnow().isoformat(),
            }

        elif event_type == "on_chain_stream":
            # 安全序列化流数据
            raw_data = event.get("data")
            safe_data = _safe_serialize(raw_data)
            yield {
                "type": "stream",
                "data": safe_data,
                "timestamp": datetime.utcnow().isoformat(),
            }


async def resume_after_interrupt(
    thread_id: str,
    user_input: Any,
):
    """
    在 interrupt 后恢复执行

    Args:
        thread_id: 线程 ID
        user_input: 用户输入

    Returns:
        最终状态
    """
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # 使用 Command 恢复执行
    result = await graph.ainvoke(Command(resume=user_input), config=config)
    return result


__all__ = [
    "build_graph",
    "get_compiled_graph",
    "get_store",
    "UserPreferences",
    "run_with_streaming",
    "resume_after_interrupt",
    "cache_check_node",
    "intent_classifier_node",
    "ambi_resolver_node",
    "data_sniper_node",
    "logic_architect_node",
    "judge_node",
    "viz_expert_node",
]
