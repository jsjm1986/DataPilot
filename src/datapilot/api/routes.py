"""
DataPilot API 路由
FastAPI 路由定义

使用 LangGraph 工作流执行查询，符合 README 设计规范
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from langgraph.types import Command

from ..agents import LogicArchitect, DataSniper, Judge, VizExpert, AmbiResolver
from ..core.state import QueryRequest, QueryResponse, ClarifyRequest, create_initial_state, AgentTrace, PipelineTrace
from ..core.graph import get_compiled_graph
from ..db.connector import get_db_manager
from ..cache import get_cache
from ..observability.audit import log_audit, AuditTimer
from datetime import datetime
import time

# 创建路由器

# Agent 元数据定义 (完整版 - 覆盖所有复杂度路径)
AGENT_METADATA = {
    # 预处理阶段
    "cache_check": {
        "display_name": "语义缓存",
        "description": "检查精确匹配和语义相似度缓存，命中则直接返回结果",
    },
    "intent_classifier": {
        "display_name": "意图分类",
        "description": "使用 LLM 分析查询意图：聚合、查找、对比、趋势、排名、复杂",
    },
    "ambi_resolver": {
        "display_name": "歧义消解器",
        "description": "检测用户查询中的歧义，必要时请求澄清",
    },
    # 路由阶段
    "supervisor": {
        "display_name": "任务协调器",
        "description": "分析查询复杂度，制定执行计划，路由到对应执行路径",
    },
    # 执行阶段 - 简单查询
    "data_sniper": {
        "display_name": "数据狙击手",
        "description": "分析 Schema，识别相关表和字段，进行值映射",
    },
    "logic_architect": {
        "display_name": "逻辑架构师",
        "description": "根据 Schema 上下文生成 SQL 候选",
    },
    # 执行阶段 - 中等复杂度
    "parallel_executor": {
        "display_name": "并行执行器",
        "description": "拆分子任务并行执行，适用于多维对比、时间序列查询",
    },
    # 执行阶段 - 高复杂度
    "supervisor_react": {
        "display_name": "ReAct 推理",
        "description": "多轮推理循环，适用于多步依赖、模糊查询",
    },
    "react_agent": {
        "display_name": "ReAct Agent",
        "description": "独立的 ReAct 推理 Agent",
    },
    # 校验和输出
    "judge": {
        "display_name": "裁判官",
        "description": "校验 SQL 语法和语义，执行查询并验证结果",
    },
    "viz_expert": {
        "display_name": "可视化专家",
        "description": "根据查询结果生成图表配置和数据洞察",
    },
}
router = APIRouter(prefix="/api/v1", tags=["DataPilot"])

# 简单的澄清会话暂存（内存级），生产应改为持久化/Redis
_clarify_sessions: dict[str, QueryRequest] = {}
# 存储待处理的 interrupt 状态（用于 LangGraph Human-in-the-loop）
_pending_interrupts: dict[str, dict] = {}


def _create_agent_trace(agent: str) -> AgentTrace:
    """创建 Agent 追踪对象"""
    metadata = AGENT_METADATA.get(agent, {})
    return AgentTrace(
        agent=agent,
        display_name=metadata.get("display_name", agent),
        description=metadata.get("description", ""),
        status="pending",
        logs=[],
        llm_calls=[],
        steps=[],
    )


def _start_agent(trace: AgentTrace, input_data: dict = None) -> AgentTrace:
    """开始 Agent 执行"""
    trace.status = "running"
    trace.start_time = datetime.utcnow().isoformat()
    trace.input_data = input_data
    trace.logs.append({
        "timestamp": trace.start_time,
        "level": "info",
        "message": f"开始执行 {trace.display_name or trace.agent}"
    })
    # 记录开始步骤
    trace.steps.append({
        "step": "start",
        "timestamp": trace.start_time,
        "description": f"开始执行 {trace.display_name}",
        "input": input_data,
    })
    return trace


def _apply_real_timing(trace: AgentTrace, agent_timings: dict, agent_id: str) -> None:
    """应用真实的 Agent 执行时间 (从 LangGraph 节点记录)"""
    timing = agent_timings.get(agent_id)
    if timing:
        trace.start_time = timing.get("start_time")
        trace.end_time = timing.get("end_time")
        trace.duration_ms = timing.get("duration_ms", 0)


def _end_agent(
    trace: AgentTrace,
    output_data: dict = None,
    error: str = None,
    steps: list = None,
    llm_calls: list = None,
    agent_timings: dict = None,
    agent_id: str = None,
) -> AgentTrace:
    """结束 Agent 执行"""
    # 优先使用真实的执行时间
    if agent_timings and agent_id and agent_id in agent_timings:
        _apply_real_timing(trace, agent_timings, agent_id)
    else:
        trace.end_time = datetime.utcnow().isoformat()
        if trace.start_time:
            start = datetime.fromisoformat(trace.start_time)
            end = datetime.fromisoformat(trace.end_time)
            trace.duration_ms = (end - start).total_seconds() * 1000

    # 添加额外的步骤记录
    if steps:
        trace.steps.extend(steps)

    # 添加 LLM 调用记录
    if llm_calls:
        trace.llm_calls.extend(llm_calls)

    if error:
        trace.status = "error"
        trace.error = error
        trace.logs.append({
            "timestamp": trace.end_time,
            "level": "error",
            "message": f"{trace.display_name or trace.agent} 执行失败: {error}"
        })
        trace.steps.append({
            "step": "error",
            "timestamp": trace.end_time,
            "description": f"执行失败: {error}",
        })
    else:
        trace.status = "success"
        trace.output_data = output_data
        trace.logs.append({
            "timestamp": trace.end_time,
            "level": "info",
            "message": f"{trace.display_name or trace.agent} 执行完成",
            "duration_ms": trace.duration_ms
        })
        trace.steps.append({
            "step": "complete",
            "timestamp": trace.end_time,
            "description": f"执行完成，耗时 {trace.duration_ms:.0f}ms",
            "output": output_data,
        })
    return trace


def _get_executed_agents(pipeline_trace: PipelineTrace) -> list[str]:
    """从 pipeline_trace 中提取已执行的 Agent 列表 (支持所有复杂度路径)"""
    executed = []
    # 完整的 Agent 执行顺序 (按工作流顺序)
    agent_order = [
        "cache_check", "intent_classifier", "ambi_resolver", "supervisor",
        "data_sniper", "logic_architect",  # 简单查询路径
        "parallel_executor",  # 中等复杂度路径
        "supervisor_react",  # 高复杂度路径
        "judge", "viz_expert"
    ]
    agent_names = {
        "cache_check": "CacheCheck",
        "intent_classifier": "IntentClassifier",
        "ambi_resolver": "AmbiResolver",
        "supervisor": "Supervisor",
        "data_sniper": "DataSniper",
        "logic_architect": "LogicArchitect",
        "parallel_executor": "ParallelExecutor",
        "supervisor_react": "SupervisorReAct",
        "judge": "Judge",
        "viz_expert": "VizExpert",
    }
    for agent_id in agent_order:
        agent_trace = pipeline_trace.agents.get(agent_id)
        if agent_trace and agent_trace.status in ("success", "error", "running"):
            executed.append(agent_names.get(agent_id, agent_id))
    return executed


def _get_agent_details(pipeline_trace: PipelineTrace) -> dict:
    """从 pipeline_trace 中提取 Agent 详细执行信息"""
    details = {}
    for agent_id, agent_trace in pipeline_trace.agents.items():
        if agent_trace.status in ("success", "error", "running"):
            details[agent_id] = {
                "display_name": agent_trace.display_name,
                "description": agent_trace.description,
                "status": agent_trace.status,
                "start_time": agent_trace.start_time,
                "end_time": agent_trace.end_time,
                "duration_ms": agent_trace.duration_ms,
                "input_data": agent_trace.input_data,
                "output_data": agent_trace.output_data,
                "error": agent_trace.error,
                "steps": agent_trace.steps,
                "llm_calls": [call.model_dump() for call in agent_trace.llm_calls] if agent_trace.llm_calls else [],
                "logs": agent_trace.logs,
            }
    return details


async def _run_pipeline(
    request: QueryRequest,
    allow_clarify: bool = True,
    original_query: str = None,  # 原始查询（用于澄清后的审计日志）
    clarification_info: dict = None,  # 澄清信息
) -> QueryResponse:
    """
    主查询流水线 - 使用 LangGraph 工作流执行

    流程: Ambi-Resolver -> Data Sniper -> Logic Architect -> Judge -> Viz Expert

    符合 README 设计规范，使用 LangGraph Supervisor 多智能体编排架构
    """
    # 创建初始状态
    initial_state = create_initial_state(
        query=request.query,
        user_id=request.user_id,
        tenant_id=request.tenant_id,
        database=request.database,
        viz_mode=request.viz_mode,
    )

    session_id = request.session_id or initial_state["trace_id"]
    trace_id = initial_state["trace_id"]

    # 审计日志使用的查询：优先使用原始查询（澄清场景），否则使用当前查询
    audit_query = original_query or request.query

    # 初始化追踪数据 (完整版 - 覆盖所有复杂度路径的 Agent)
    pipeline_start = datetime.utcnow()
    pipeline_trace = PipelineTrace(
        trace_id=trace_id,
        start_time=pipeline_start.isoformat(),
        agents={
            # 预处理阶段
            "cache_check": _create_agent_trace("cache_check"),
            "intent_classifier": _create_agent_trace("intent_classifier"),
            "ambi_resolver": _create_agent_trace("ambi_resolver"),
            # 路由阶段
            "supervisor": _create_agent_trace("supervisor"),
            # 执行阶段 - 简单查询
            "data_sniper": _create_agent_trace("data_sniper"),
            "logic_architect": _create_agent_trace("logic_architect"),
            # 执行阶段 - 中等复杂度
            "parallel_executor": _create_agent_trace("parallel_executor"),
            # 执行阶段 - 高复杂度
            "supervisor_react": _create_agent_trace("supervisor_react"),
            # 校验和输出
            "judge": _create_agent_trace("judge"),
            "viz_expert": _create_agent_trace("viz_expert"),
        },
        logs=[]
    )

    pipeline_trace.logs.append({
        "timestamp": pipeline_start.isoformat(),
        "level": "info",
        "message": f"开始处理查询 (LangGraph): {request.query[:50]}..."
    })

    with AuditTimer() as timer:
        # 检查缓存
        cache = get_cache()
        cached = await cache.get(request.query, request.database or "default")
        if cached:
            # 缓存命中，跳过所有 Agent
            for agent_name in ["ambi_resolver", "data_sniper", "logic_architect", "judge", "viz_expert"]:
                pipeline_trace.agents[agent_name].status = "skipped"

            pipeline_trace.end_time = datetime.utcnow().isoformat()
            pipeline_trace.total_duration_ms = (datetime.utcnow() - pipeline_start).total_seconds() * 1000
            pipeline_trace.logs.append({
                "timestamp": pipeline_trace.end_time,
                "level": "info",
                "message": "缓存命中，直接返回结果"
            })

            # 缓存命中时也生成图表配置
            chart_config = None
            if cached.data:
                try:
                    viz = VizExpert(mode=request.viz_mode)
                    analysis = await viz.analyze(cached.data, request.query, force_mode=request.viz_mode)
                    chart_config = {
                        "chart_type": analysis.get("chart_type"),
                        "title": request.query[:50],
                        "config": analysis.get("echarts_config"),
                    }
                except Exception as viz_err:
                    print(f"VizExpert error on cache hit: {viz_err}")

            log_audit(
                user_id=request.user_id,
                tenant_id=request.tenant_id,
                trace_id=trace_id,
                session_id=session_id,
                database=request.database,
                query=audit_query,
                sql=cached.sql,
                status="success",
                row_count=cached.row_count,
                duration_ms=pipeline_trace.total_duration_ms,
                agent_path=["CacheHit"],
                cache_hit=True,
                extra={"cached": True, **(clarification_info or {})},
            )
            return QueryResponse(
                session_id=session_id,
                trace_id=trace_id,
                sql=cached.sql,
                data=cached.data,
                row_count=cached.row_count,
                chart_config=chart_config,
                status="success",
                insight=f"(cached, hits: {cached.hit_count})",
                trace=pipeline_trace,
            )

        try:
            # 获取编译后的 LangGraph 工作流
            graph = get_compiled_graph(include_ambi_resolver=allow_clarify)
            config = {"configurable": {"thread_id": trace_id}}

            # 使用 ainvoke 执行工作流（比 astream_events 更可靠地处理 interrupt）
            # ainvoke 会在遇到 interrupt 时返回当前状态
            final_state = await graph.ainvoke(initial_state, config=config)

            # 获取状态快照来检查是否被中断
            state_snapshot = await graph.aget_state(config)

            # 检查是否有待处理的中断（interrupt）
            # 当 next 不为空时，说明工作流被中断了
            if state_snapshot.next:
                # 工作流被中断，检查中断数据
                # 从 tasks 中获取中断信息
                interrupt_data = None
                if hasattr(state_snapshot, 'tasks') and state_snapshot.tasks:
                    for task in state_snapshot.tasks:
                        if hasattr(task, 'interrupts') and task.interrupts:
                            interrupt_data = task.interrupts[0].value if task.interrupts else None
                            break

                if interrupt_data:
                    interrupt_type = interrupt_data.get("type", "")

                    if interrupt_type == "clarify_needed":
                        # 需要澄清
                        _start_agent(pipeline_trace.agents["ambi_resolver"], {"query": request.query})
                        _end_agent(
                            pipeline_trace.agents["ambi_resolver"],
                            {"result": "clarify_needed", "options": interrupt_data.get("options", [])}
                        )

                        # 保存会话状态用于后续恢复
                        _clarify_sessions[session_id] = request
                        _pending_interrupts[session_id] = {
                            "thread_id": trace_id,
                            "config": config,
                        }

                        pipeline_trace.end_time = datetime.utcnow().isoformat()
                        pipeline_trace.total_duration_ms = (datetime.utcnow() - pipeline_start).total_seconds() * 1000

                        clarify_options = {
                            "question": interrupt_data.get("question", "请选择一个选项"),
                            "options": interrupt_data.get("options", []),
                        }

                        log_audit(
                            user_id=request.user_id,
                            tenant_id=request.tenant_id,
                            trace_id=trace_id,
                            session_id=session_id,
                            database=request.database,
                            query=audit_query,
                            sql=None,
                            status="clarify_needed",
                            duration_ms=pipeline_trace.total_duration_ms,
                            agent_path=_get_executed_agents(pipeline_trace),
                            agent_details=_get_agent_details(pipeline_trace),
                        )
                        return QueryResponse(
                            session_id=session_id,
                            trace_id=trace_id,
                            status="clarify_needed",
                            clarify_options=clarify_options,
                            trace=pipeline_trace,
                        )

                    elif interrupt_type == "human_handoff":
                        # 需要人工介入
                        pipeline_trace.end_time = datetime.utcnow().isoformat()
                        pipeline_trace.total_duration_ms = (datetime.utcnow() - pipeline_start).total_seconds() * 1000

                        log_audit(
                            user_id=request.user_id,
                            tenant_id=request.tenant_id,
                            trace_id=trace_id,
                            session_id=session_id,
                            database=request.database,
                            query=audit_query,
                            sql=None,
                            status="human_handoff",
                            duration_ms=pipeline_trace.total_duration_ms,
                            agent_path=_get_executed_agents(pipeline_trace),
                            agent_details=_get_agent_details(pipeline_trace),
                        )
                        return QueryResponse(
                            session_id=session_id,
                            trace_id=trace_id,
                            status="human_handoff",
                            error_message=interrupt_data.get("reason", "需要人工介入"),
                            trace=pipeline_trace,
                        )

            # 工作流正常完成，更新 Agent 追踪状态
            # 根据 final_state 中的 current_agent 和其他字段更新追踪
            if final_state:
                # 获取真实的 Agent 执行时间 (从 LangGraph 节点记录)
                agent_timings = final_state.get("agent_timings", {})

                # 完整的 Agent 执行顺序 (支持所有复杂度路径)
                all_agents = [
                    "cache_check", "intent_classifier", "ambi_resolver", "supervisor",
                    "data_sniper", "logic_architect",  # 简单查询路径
                    "parallel_executor",  # 中等复杂度路径
                    "supervisor_react",  # 高复杂度路径
                    "judge", "viz_expert"
                ]
                current_agent = final_state.get("current_agent", "")
                task_plan = final_state.get("task_plan", {})

                # 根据任务计划判断执行路径
                complexity = task_plan.get("complexity", "simple") if task_plan else "simple"
                needs_decomposition = task_plan.get("needs_decomposition", False) if task_plan else False

                # ============================================
                # 预处理阶段 Agent 追踪
                # ============================================

                # cache_check
                cache_trace = pipeline_trace.agents["cache_check"]
                if final_state.get("cache_hit"):
                    _start_agent(cache_trace, {"query": request.query})
                    _end_agent(cache_trace, {
                        "cache_hit": True,
                        "similarity": final_state.get("cache_similarity", 1.0),
                    }, steps=[{"step": "cache_lookup", "description": "语义缓存查找", "result": "命中缓存"}],
                    agent_timings=agent_timings, agent_id="cache_check")
                else:
                    _start_agent(cache_trace, {"query": request.query})
                    _end_agent(cache_trace, {"cache_hit": False}, steps=[{"step": "cache_lookup", "description": "语义缓存查找", "result": "未命中"}],
                    agent_timings=agent_timings, agent_id="cache_check")

                # intent_classifier
                intent_trace = pipeline_trace.agents["intent_classifier"]
                intent_result = final_state.get("intent_result", {})
                if intent_result:
                    _start_agent(intent_trace, {"query": request.query})
                    _end_agent(intent_trace, {
                        "primary_intent": intent_result.get("primary_intent"),
                        "confidence": intent_result.get("confidence"),
                    }, steps=[
                        {"step": "classify", "description": "意图分类", "result": intent_result.get("primary_intent", "unknown")},
                        {"step": "routing_hint", "description": "路由建议", "result": intent_result.get("routing_hint", "")},
                    ], agent_timings=agent_timings, agent_id="intent_classifier")
                else:
                    intent_trace.status = "skipped"

                # supervisor
                supervisor_trace = pipeline_trace.agents["supervisor"]
                if task_plan:
                    _start_agent(supervisor_trace, {"query": request.query, "intent": intent_result.get("primary_intent")})
                    _end_agent(supervisor_trace, {
                        "complexity": complexity,
                        "needs_decomposition": needs_decomposition,
                        "subtask_count": len(task_plan.get("subtasks", [])),
                    }, steps=[
                        {"step": "analyze_complexity", "description": "分析查询复杂度", "result": complexity},
                        {"step": "create_plan", "description": "制定执行计划", "result": f"需要拆解: {needs_decomposition}"},
                    ], agent_timings=agent_timings, agent_id="supervisor")
                else:
                    supervisor_trace.status = "skipped"

                # ============================================
                # 中等复杂度路径: parallel_executor
                # ============================================
                parallel_trace = pipeline_trace.agents["parallel_executor"]
                if needs_decomposition and complexity != "high":
                    subtask_results = final_state.get("subtask_results", {})
                    _start_agent(parallel_trace, {
                        "subtask_count": len(task_plan.get("subtasks", [])),
                    })
                    steps = [{"step": "decompose", "description": "任务拆解", "result": f"拆分为 {len(task_plan.get('subtasks', []))} 个子任务"}]
                    for task_id, result in subtask_results.items():
                        steps.append({
                            "step": f"subtask_{task_id}",
                            "description": f"执行子任务 {task_id}",
                            "result": result,
                        })
                    steps.append({"step": "aggregate", "description": "结果聚合", "result": task_plan.get("aggregation_strategy", "merge")})
                    _end_agent(parallel_trace, {
                        "subtask_count": len(subtask_results),
                        "aggregation_strategy": task_plan.get("aggregation_strategy"),
                    }, steps=steps, agent_timings=agent_timings, agent_id="parallel_executor")
                else:
                    parallel_trace.status = "skipped"

                # ============================================
                # 高复杂度路径: supervisor_react
                # ============================================
                react_trace = pipeline_trace.agents["supervisor_react"]
                if complexity == "high" or final_state.get("supervisor_react_trigger"):
                    _start_agent(react_trace, {
                        "query": request.query,
                        "trigger": final_state.get("supervisor_react_trigger", "high_complexity"),
                    })
                    # 从 final_state 获取 ReAct 推理步骤
                    react_steps = final_state.get("react_steps", [])
                    steps = [{"step": "init", "description": "初始化 ReAct 推理", "result": "开始多轮推理"}]
                    for i, rs in enumerate(react_steps[:5]):  # 最多记录5轮
                        steps.append({
                            "step": f"thought_{i+1}",
                            "description": f"推理轮次 {i+1}",
                            "result": rs,
                        })
                    if final_state.get("execution_result", {}).get("success"):
                        _end_agent(react_trace, {
                            "reasoning_rounds": len(react_steps),
                            "final_sql": final_state.get("winner_sql", "")[:200],
                        }, steps=steps, agent_timings=agent_timings, agent_id="supervisor_react")
                    else:
                        _end_agent(react_trace, None, final_state.get("last_error", "推理失败"), steps=steps,
                                   agent_timings=agent_timings, agent_id="supervisor_react")
                else:
                    react_trace.status = "skipped"

                # ============================================
                # 简单查询路径 Agent 追踪 (data_sniper, logic_architect, judge, viz_expert, ambi_resolver)
                # ============================================
                simple_path_agents = ["ambi_resolver", "data_sniper", "logic_architect", "judge", "viz_expert"]

                for agent_name in simple_path_agents:
                    agent_trace = pipeline_trace.agents[agent_name]
                    if not allow_clarify and agent_name == "ambi_resolver":
                        agent_trace.status = "skipped"
                        continue

                    # 根据状态判断 Agent 是否执行，记录详细步骤
                    # DataSniper: 只要有 winner_sql，说明 DataSniper 执行过了（即使 schema_context 为空）
                    if agent_name == "data_sniper" and final_state.get("winner_sql"):
                        _start_agent(agent_trace, {"query": request.query, "database": request.database})
                        # 记录详细步骤
                        relevant_tables = final_state.get("relevant_tables", [])
                        value_mappings = final_state.get("value_mappings", {})
                        steps = [
                            {"step": "schema_retrieval", "description": "获取数据库 Schema", "result": f"获取到 {len(relevant_tables)} 张相关表" if relevant_tables else "Schema 分析完成"},
                            {"step": "value_mapping", "description": "值映射分析", "result": f"映射 {len(value_mappings)} 个值" if value_mappings else "无需值映射"},
                        ]
                        if final_state.get("time_context"):
                            steps.append({"step": "time_context", "description": "时间字段分析", "result": final_state.get("time_context", "")[:200]})
                        if final_state.get("enum_context"):
                            steps.append({"step": "enum_context", "description": "枚举字段分析", "result": final_state.get("enum_context", "")[:200]})
                        _end_agent(agent_trace, {
                            "tables": relevant_tables,
                            "schema_length": len(final_state.get("schema_context", "")),
                            "value_mappings_count": len(value_mappings),
                        }, steps=steps, agent_timings=agent_timings, agent_id="data_sniper")

                    elif agent_name == "logic_architect" and final_state.get("winner_sql"):
                        candidates = final_state.get("candidates", [])
                        _start_agent(agent_trace, {
                            "schema_context_length": len(final_state.get("schema_context", "")),
                            "tables": final_state.get("relevant_tables", []),
                        })
                        # 记录 SQL 生成步骤
                        steps = [
                            {"step": "analyze_query", "description": "分析用户查询意图", "result": request.query},
                            {"step": "generate_candidates", "description": "生成 SQL 候选", "result": f"生成 {len(candidates)} 个候选 SQL"},
                        ]
                        for i, candidate in enumerate(candidates[:3]):  # 最多记录3个候选
                            steps.append({
                                "step": f"candidate_{i+1}",
                                "description": f"候选 SQL {i+1}",
                                "result": {
                                    "sql": candidate.get("sql", "")[:500],
                                    "confidence": candidate.get("confidence", 0),
                                    "strategy": candidate.get("strategy", "unknown"),
                                }
                            })
                        steps.append({"step": "select_winner", "description": "选择最佳 SQL", "result": final_state.get("winner_sql", "")[:500]})
                        _end_agent(agent_trace, {
                            "sql": final_state.get("winner_sql", ""),
                            "candidates_count": len(candidates),
                        }, steps=steps, agent_timings=agent_timings, agent_id="logic_architect")

                    elif agent_name == "judge" and final_state.get("execution_result"):
                        exec_result = final_state.get("execution_result", {})
                        _start_agent(agent_trace, {"sql": final_state.get("winner_sql", "")})
                        # 记录校验和执行步骤
                        steps = [
                            {"step": "syntax_check", "description": "SQL 语法校验", "result": "通过"},
                            {"step": "semantic_check", "description": "SQL 语义校验", "result": "通过"},
                        ]
                        if final_state.get("execution_plan"):
                            steps.append({"step": "explain_plan", "description": "执行计划分析", "result": final_state.get("execution_plan", "")[:500]})
                        steps.append({
                            "step": "execute_sql",
                            "description": "执行 SQL 查询",
                            "result": {
                                "success": exec_result.get("success", False),
                                "row_count": exec_result.get("row_count", 0),
                                "execution_time_ms": exec_result.get("execution_time_ms", 0),
                                "columns": exec_result.get("columns", []),
                            }
                        })
                        if exec_result.get("success"):
                            _end_agent(agent_trace, {
                                "row_count": exec_result.get("row_count", 0),
                                "columns": exec_result.get("columns", []),
                                "execution_time_ms": exec_result.get("execution_time_ms", 0),
                            }, steps=steps, agent_timings=agent_timings, agent_id="judge")
                        else:
                            _end_agent(agent_trace, None, exec_result.get("error", "执行失败"), steps=steps,
                                       agent_timings=agent_timings, agent_id="judge")

                    elif agent_name == "viz_expert" and final_state.get("chart_config"):
                        chart_config = final_state.get("chart_config", {})
                        exec_result = final_state.get("execution_result", {})
                        _start_agent(agent_trace, {
                            "data_count": exec_result.get("row_count", 0),
                            "columns": exec_result.get("columns", []),
                        })
                        # 记录可视化步骤
                        steps = [
                            {"step": "analyze_data", "description": "分析数据特征", "result": f"数据行数: {exec_result.get('row_count', 0)}, 列: {exec_result.get('columns', [])}"},
                            {"step": "select_chart_type", "description": "选择图表类型", "result": chart_config.get("chart_type", "unknown")},
                            {"step": "generate_config", "description": "生成图表配置", "result": f"标题: {chart_config.get('title', '')}"},
                        ]
                        if final_state.get("insight_text"):
                            steps.append({"step": "generate_insight", "description": "生成数据洞察", "result": final_state.get("insight_text", "")[:300]})
                        _end_agent(agent_trace, {
                            "chart_type": chart_config.get("chart_type"),
                            "title": chart_config.get("title"),
                            "has_insight": bool(final_state.get("insight_text")),
                        }, steps=steps, agent_timings=agent_timings, agent_id="viz_expert")

                    elif agent_name == "ambi_resolver" and allow_clarify:
                        _start_agent(agent_trace, {"query": request.query})
                        steps = [
                            {"step": "detect_ambiguity", "description": "检测查询歧义", "result": "无歧义，继续执行"},
                        ]
                        _end_agent(agent_trace, {"resolved": True, "clarify_needed": False}, steps=steps,
                                   agent_timings=agent_timings, agent_id="ambi_resolver")

            # 从最终状态提取结果
            if final_state:
                sql = final_state.get("winner_sql")
                exec_result = final_state.get("execution_result") or {}
                data = exec_result.get("data", []) if exec_result.get("success") else []
                row_count = exec_result.get("row_count", 0) if exec_result else 0
                chart_config = final_state.get("chart_config")

                # 检查是否有错误
                if exec_result and not exec_result.get("success"):
                    error_msg = exec_result.get("error", "执行失败")
                    pipeline_trace.end_time = datetime.utcnow().isoformat()
                    pipeline_trace.total_duration_ms = (datetime.utcnow() - pipeline_start).total_seconds() * 1000
                    pipeline_trace.logs.append({
                        "timestamp": pipeline_trace.end_time,
                        "level": "error",
                        "message": f"SQL 执行失败: {error_msg}"
                    })

                    log_audit(
                        user_id=request.user_id,
                        tenant_id=request.tenant_id,
                        trace_id=trace_id,
                        session_id=session_id,
                        database=request.database,
                        query=audit_query,
                        sql=sql,
                        status="error",
                        error=error_msg,
                        duration_ms=pipeline_trace.total_duration_ms,
                        agent_path=_get_executed_agents(pipeline_trace),
                        agent_details=_get_agent_details(pipeline_trace),
                        extra=clarification_info,
                    )
                    return QueryResponse(
                        session_id=session_id,
                        trace_id=trace_id,
                        sql=sql,
                        status="error",
                        error_message=error_msg,
                        trace=pipeline_trace,
                    )

                # 写入缓存
                if sql and data:
                    cache.set(request.query, sql, data, row_count)

                # 成功完成
                pipeline_trace.end_time = datetime.utcnow().isoformat()
                pipeline_trace.total_duration_ms = (datetime.utcnow() - pipeline_start).total_seconds() * 1000
                pipeline_trace.logs.append({
                    "timestamp": pipeline_trace.end_time,
                    "level": "info",
                    "message": f"查询完成，返回 {row_count} 条数据"
                })

                log_audit(
                    user_id=request.user_id,
                    tenant_id=request.tenant_id,
                    trace_id=trace_id,
                    session_id=session_id,
                    database=request.database,
                    query=audit_query,
                    sql=sql,
                    status="success",
                    row_count=row_count,
                    duration_ms=pipeline_trace.total_duration_ms,
                    agent_path=_get_executed_agents(pipeline_trace),
                    agent_details=_get_agent_details(pipeline_trace),
                    extra=clarification_info,
                )

                return QueryResponse(
                    session_id=session_id,
                    trace_id=trace_id,
                    sql=sql,
                    data=data,
                    row_count=row_count,
                    chart_config=chart_config,
                    status="success",
                    trace=pipeline_trace,
                )

            # 没有最终状态，返回错误
            pipeline_trace.end_time = datetime.utcnow().isoformat()
            pipeline_trace.total_duration_ms = (datetime.utcnow() - pipeline_start).total_seconds() * 1000

            return QueryResponse(
                session_id=session_id,
                trace_id=trace_id,
                status="error",
                error_message="工作流执行完成但没有结果",
                trace=pipeline_trace,
            )

        except Exception as e:
            # 处理异常 - 打印完整堆栈
            import traceback
            error_traceback = traceback.format_exc()
            print(f"LangGraph workflow error:\n{error_traceback}")

            pipeline_trace.end_time = datetime.utcnow().isoformat()
            pipeline_trace.total_duration_ms = (datetime.utcnow() - pipeline_start).total_seconds() * 1000
            pipeline_trace.logs.append({
                "timestamp": pipeline_trace.end_time,
                "level": "error",
                "message": f"工作流执行失败: {str(e)}\n{error_traceback[:500]}"
            })

            log_audit(
                user_id=request.user_id,
                tenant_id=request.tenant_id,
                trace_id=trace_id,
                session_id=session_id,
                database=request.database,
                query=audit_query,
                sql=None,
                status="error",
                error=str(e),
                duration_ms=pipeline_trace.total_duration_ms,
                agent_path=_get_executed_agents(pipeline_trace),
                agent_details=_get_agent_details(pipeline_trace),
                extra=clarification_info,
            )
            return QueryResponse(
                session_id=session_id,
                trace_id=trace_id,
                status="error",
                error_message=f"工作流执行失败: {str(e)}",
                trace=pipeline_trace,
            )


def _safe_output_data(output: Any) -> dict:
    """安全地提取输出数据，移除大数据字段"""
    if output is None:
        return {}
    if not isinstance(output, dict):
        return {"raw": str(output)[:200]}

    result = {}
    try:
        for k, v in output.items():
            if k in ("data", "execution_result") and isinstance(v, (list, dict)):
                if isinstance(v, list):
                    result[k] = f"[{len(v)} items]"
                elif isinstance(v, dict) and "data" in v:
                    result[k] = {**{kk: vv for kk, vv in v.items() if kk != "data"}, "data": f"[{len(v.get('data', []))} items]"}
                else:
                    result[k] = v
            elif isinstance(v, str) and len(v) > 500:
                result[k] = v[:500] + "..."
            else:
                result[k] = v
    except Exception:
        return {"raw": str(output)[:200]}
    return result


# ============================================
# 请求/响应模型
# ============================================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "0.1.0"


class SchemaResponse(BaseModel):
    """Schema 响应"""
    database: str
    tables: list[dict]
    ddl: str


class ExecuteRequest(BaseModel):
    """SQL 执行请求"""
    sql: str = Field(..., description="要执行的 SQL")
    database: str = Field(default="default", description="目标数据库")
    limit: int = Field(default=100, ge=1, le=1000, description="结果行数限制")


class ExecuteResponse(BaseModel):
    """SQL 执行响应"""
    success: bool
    data: Optional[list[dict]] = None
    row_count: int = 0
    columns: list[str] = []
    error: Optional[str] = None


# ============================================
# 路由定义
# ============================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse()


@router.get("/databases")
async def list_databases():
    """获取可用数据库列表"""
    return {
        "databases": [
            {"name": "default", "type": "sqlite", "description": "默认数据库 (SQLite)"},
        ]
    }


@router.get("/schema/{database}", response_model=SchemaResponse)
async def get_schema(database: str):
    """
    获取数据库 Schema

    Args:
        database: 数据库名称 (ecommerce/sales)
    """
    try:
        db_manager = get_db_manager()
        connector = db_manager.get(database)

        tables = await connector.get_tables()
        ddl = await connector.get_schema()

        return SchemaResponse(
            database=database,
            tables=tables,
            ddl=ddl,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 Schema 失败: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    自然语言查询接口 - 完整 Agent 链路 + 语义缓存

    流程: Cache -> DataSniper -> LogicArchitect -> Judge -> VizExpert
    """
    try:
        return await _run_pipeline(request, allow_clarify=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/clarify", response_model=QueryResponse)
async def clarify(request: ClarifyRequest):
    """
    澄清接口：用户选择澄清选项后继续执行查询
    """
    try:
        original = _clarify_sessions.pop(request.session_id, None)
        if not original:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        resolver = AmbiResolver()
        resolved_query = await resolver.resolve_with_selection(original.query, request.selected_option)

        clarified_request = QueryRequest(
            query=resolved_query,
            user_id=original.user_id,
            tenant_id=original.tenant_id,
            database=original.database,
            session_id=request.session_id,
        )

        # 构建澄清信息，用于审计日志
        clarification_info = {
            "clarified": True,
            "selected_option": request.selected_option,
            "resolved_query": resolved_query,
        }

        # 继续流水线，避免再次澄清循环
        # 传递原始查询用于审计日志，确保日志记录的是用户原始问题
        return await _run_pipeline(
            clarified_request,
            allow_clarify=False,
            original_query=original.query,
            clarification_info=clarification_info,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clarification failed: {str(e)}")

@router.post("/execute", response_model=ExecuteResponse)
async def execute_sql(request: ExecuteRequest):
    """
    直接执行 SQL 查询

    用于调试和测试
    """
    try:
        db_manager = get_db_manager()
        connector = db_manager.get(request.database)

        # 安全检查：只允许 SELECT
        sql_lower = request.sql.lower().strip()
        if not sql_lower.startswith("select"):
            raise HTTPException(
                status_code=400,
                detail="只允许执行 SELECT 查询"
            )

        # 执行查询
        data = await connector.execute_query(request.sql, limit=request.limit)

        columns = list(data[0].keys()) if data else []

        return ExecuteResponse(
            success=True,
            data=data,
            row_count=len(data),
            columns=columns,
        )

    except HTTPException:
        raise
    except Exception as e:
        return ExecuteResponse(
            success=False,
            error=str(e),
        )


@router.get("/tables/{database}")
async def list_tables(database: str):
    """获取数据库表列表"""
    try:
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        tables = await connector.get_tables()

        return {
            "database": database,
            "tables": [
                {
                    "name": t["name"],
                    "comment": t.get("comment", ""),
                    "column_count": len(t.get("columns", [])),
                }
                for t in tables
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/table/{database}/{table_name}")
async def get_table_info(database: str, table_name: str):
    """获取表详细信息"""
    try:
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        tables = await connector.get_tables()

        for table in tables:
            if table["name"] == table_name:
                return table

        raise HTTPException(status_code=404, detail=f"表不存在: {table_name}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 导出
__all__ = ["router"]
