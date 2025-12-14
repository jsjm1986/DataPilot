# -*- coding: utf-8 -*-
"""
Supervisor Agent - 中央协调器

实现真正的多 Agent 协作：
1. 任务拆解 (Task Decomposition)
2. 动态路由 (Dynamic Routing)
3. 并行执行 (Parallel Execution)
4. 结果聚合 (Result Aggregation)

架构模式: Supervisor Pattern (LangGraph 推荐)
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.types import Command

from ..llm.deepseek import get_deepseek_client


class TaskType(Enum):
    """任务类型"""
    SIMPLE = "simple"           # 简单查询，直接执行
    COMPLEX = "complex"         # 复杂查询，需要拆解
    MULTI_TABLE = "multi_table" # 多表关联
    AGGREGATION = "aggregation" # 聚合分析
    COMPARISON = "comparison"   # 对比分析
    TREND = "trend"             # 趋势分析


class AgentRole(Enum):
    """Agent 角色"""
    DATA_SNIPER = "data_sniper"
    LOGIC_ARCHITECT = "logic_architect"
    JUDGE = "judge"
    VIZ_EXPERT = "viz_expert"
    AGGREGATOR = "aggregator"  # 结果聚合器


@dataclass
class SubTask:
    """子任务"""
    id: str
    parent_id: Optional[str]
    query: str
    task_type: TaskType
    assigned_agent: AgentRole
    dependencies: list[str] = field(default_factory=list)  # 依赖的子任务 ID
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class ExecutionPlan:
    """执行计划"""
    original_query: str
    task_type: TaskType
    subtasks: list[SubTask]
    execution_order: list[list[str]]  # 分层执行顺序，同层可并行
    aggregation_strategy: str  # union, join, merge, none


# Supervisor 的 System Prompt
SUPERVISOR_PROMPT = """你是一个智能任务协调器，负责分析用户查询并制定执行计划。

## 任务
分析用户的自然语言查询，判断复杂度和执行策略。

## 复杂度分级

### 简单 (simple) - 串行执行
- 单表查询
- 简单聚合 (单一 COUNT/SUM/AVG)
- 简单过滤条件
- 直接排序

### 中等 (medium) - 并行执行
- 多维度对比: "A 和 B 的销售额对比" → 拆成 2 个子查询
- 时间序列对比: "本月 vs 上月" → 拆成 2 个子查询
- 多表关联但结构清晰

### 高复杂 (high) - ReAct 推理
- 多步骤依赖: "找出销量最高的产品，然后分析其趋势"
- 需要中间结果: "销售额超过平均值的产品"
- 模糊/不确定查询: "哪个地区客户最活跃" (需要定义"活跃")
- 需要推理判断的查询

## 输出格式
```json
{
    "task_type": "simple|complex|multi_table|aggregation|comparison|trend",
    "complexity": "simple|medium|high",
    "needs_decomposition": true/false,
    "needs_reasoning": true/false,
    "reasoning": "判断理由",
    "subtasks": [
        {
            "id": "task_1",
            "query": "子查询描述",
            "task_type": "类型",
            "dependencies": []
        }
    ],
    "execution_order": [["task_1"]],
    "aggregation_strategy": "union|join|merge|none"
}
```

## 判断 needs_reasoning = true 的情况
1. 查询中有模糊概念需要定义 (如"最活跃"、"最好"、"最近")
2. 需要根据中间结果做进一步决策
3. 可能需要多轮尝试/修正
4. 查询意图不明确，需要推理

## 注意
- 不要过度拆解简单查询
- 高复杂度查询应该 needs_reasoning = true
- 中间结果依赖链应该标记为 high 复杂度
"""


class Supervisor:
    """
    Supervisor Agent - 中央协调器

    职责:
    1. 分析查询复杂度
    2. 拆解复杂任务
    3. 协调 Agent 执行
    4. 聚合结果
    """

    def __init__(self):
        self.llm = get_deepseek_client()

    async def analyze_and_plan(self, query: str, schema: str = "") -> ExecutionPlan:
        """
        分析查询并制定执行计划

        Args:
            query: 用户查询
            schema: 数据库 Schema (可选，用于更准确的判断)

        Returns:
            执行计划
        """
        messages = [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": f"用户查询: {query}\n\nSchema 信息:\n{schema[:2000] if schema else '未提供'}"},
        ]

        try:
            response = await self.llm.chat(messages)
            return self._parse_plan(response, query)
        except Exception as e:
            # 解析失败，返回简单计划
            return ExecutionPlan(
                original_query=query,
                task_type=TaskType.SIMPLE,
                subtasks=[
                    SubTask(
                        id="main",
                        parent_id=None,
                        query=query,
                        task_type=TaskType.SIMPLE,
                        assigned_agent=AgentRole.LOGIC_ARCHITECT,
                    )
                ],
                execution_order=[["main"]],
                aggregation_strategy="none",
            )

    def _parse_plan(self, response: str, original_query: str) -> ExecutionPlan:
        """解析 LLM 响应"""
        import json
        import re

        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return self._simple_plan(original_query)

        try:
            data = json.loads(json_match.group())

            task_type_str = data.get("task_type", "simple").lower()
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.SIMPLE

            subtasks = []
            for st in data.get("subtasks", []):
                st_type_str = st.get("task_type", "simple").lower()
                try:
                    st_type = TaskType(st_type_str)
                except ValueError:
                    st_type = TaskType.SIMPLE

                subtasks.append(SubTask(
                    id=st.get("id", f"task_{len(subtasks)}"),
                    parent_id=None,
                    query=st.get("query", original_query),
                    task_type=st_type,
                    assigned_agent=AgentRole.LOGIC_ARCHITECT,
                    dependencies=st.get("dependencies", []),
                ))

            if not subtasks:
                return self._simple_plan(original_query)

            return ExecutionPlan(
                original_query=original_query,
                task_type=task_type,
                subtasks=subtasks,
                execution_order=data.get("execution_order", [[st.id for st in subtasks]]),
                aggregation_strategy=data.get("aggregation_strategy", "none"),
            )

        except (json.JSONDecodeError, KeyError, TypeError):
            return self._simple_plan(original_query)

    def _simple_plan(self, query: str) -> ExecutionPlan:
        """创建简单执行计划"""
        return ExecutionPlan(
            original_query=query,
            task_type=TaskType.SIMPLE,
            subtasks=[
                SubTask(
                    id="main",
                    parent_id=None,
                    query=query,
                    task_type=TaskType.SIMPLE,
                    assigned_agent=AgentRole.LOGIC_ARCHITECT,
                )
            ],
            execution_order=[["main"]],
            aggregation_strategy="none",
        )

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        schema: str,
        database: str,
        execute_subtask_fn,
    ) -> dict:
        """
        执行计划

        Args:
            plan: 执行计划
            schema: Schema 上下文
            database: 数据库名称
            execute_subtask_fn: 子任务执行函数

        Returns:
            聚合后的结果
        """
        results = {}

        # 按层执行，同层并行
        for layer in plan.execution_order:
            # 检查依赖是否满足
            tasks_to_run = []
            for task_id in layer:
                task = next((t for t in plan.subtasks if t.id == task_id), None)
                if task:
                    # 检查依赖
                    deps_satisfied = all(
                        dep in results and results[dep].get("success")
                        for dep in task.dependencies
                    )
                    if deps_satisfied:
                        tasks_to_run.append(task)

            # 并行执行同层任务
            if tasks_to_run:
                layer_results = await asyncio.gather(*[
                    execute_subtask_fn(
                        task=task,
                        schema=schema,
                        database=database,
                        context={dep: results.get(dep) for dep in task.dependencies},
                    )
                    for task in tasks_to_run
                ], return_exceptions=True)

                # 收集结果
                for task, result in zip(tasks_to_run, layer_results):
                    if isinstance(result, Exception):
                        results[task.id] = {
                            "success": False,
                            "error": str(result),
                        }
                        task.status = "failed"
                        task.error = str(result)
                    else:
                        results[task.id] = result
                        task.status = "completed"
                        task.result = result
                    task.completed_at = datetime.utcnow().isoformat()

        # 聚合结果
        return await self._aggregate_results(plan, results)

    async def _aggregate_results(
        self,
        plan: ExecutionPlan,
        results: dict[str, dict],
    ) -> dict:
        """聚合子任务结果"""
        if plan.aggregation_strategy == "none" or len(plan.subtasks) == 1:
            # 单任务，直接返回
            main_result = results.get("main") or list(results.values())[0]
            return main_result

        strategy = plan.aggregation_strategy

        if strategy == "union":
            # 合并数据 (UNION)
            all_data = []
            for task_id, result in results.items():
                if result.get("success") and result.get("data"):
                    all_data.extend(result["data"])
            return {
                "success": True,
                "data": all_data,
                "row_count": len(all_data),
                "aggregation": "union",
            }

        elif strategy == "join":
            # 关联数据 (需要 LLM 帮助)
            return await self._llm_aggregate(plan, results, "join")

        elif strategy == "merge":
            # 合并结果 (保留所有字段)
            merged = {}
            for task_id, result in results.items():
                if result.get("success"):
                    merged[task_id] = result
            return {
                "success": True,
                "merged_results": merged,
                "aggregation": "merge",
            }

        else:
            # 默认返回所有结果
            return {
                "success": True,
                "results": results,
                "aggregation": "default",
            }

    async def _llm_aggregate(
        self,
        plan: ExecutionPlan,
        results: dict,
        strategy: str,
    ) -> dict:
        """使用 LLM 聚合复杂结果"""
        import json

        prompt = f"""请将以下子查询结果聚合成最终答案。

原始查询: {plan.original_query}
聚合策略: {strategy}

子查询结果:
{json.dumps(results, ensure_ascii=False, indent=2, default=str)[:3000]}

请返回聚合后的结果，格式:
```json
{{
    "success": true,
    "data": [...],
    "summary": "结果摘要"
}}
```
"""
        messages = [
            {"role": "system", "content": "你是一个数据聚合专家，负责将多个子查询结果合并成最终答案。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.llm.chat(messages)
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        # 回退到简单合并
        return {
            "success": True,
            "results": results,
            "aggregation": "fallback",
        }


# ============================================
# 集成到 LangGraph 工作流
# ============================================

async def supervisor_node(state: dict) -> dict:
    """
    Supervisor 节点 - 分析查询并制定执行计划

    在 intent_classifier 之后、data_sniper 之前执行
    返回 task_plan 包含:
    - task_type: 任务类型
    - complexity: 复杂度 (simple/medium/high)
    - needs_decomposition: 是否需要拆解
    - needs_reasoning: 是否需要多轮推理 (用于路由到 ReActAgent)
    """
    import time
    start_time = time.time()
    query = state["query"]
    schema = state.get("schema_context", "")

    supervisor = Supervisor()
    plan = await supervisor.analyze_and_plan(query, schema)

    # 判断复杂度
    subtask_count = len(plan.subtasks)
    has_dependencies = any(st.dependencies for st in plan.subtasks)

    # 自动推断 complexity 和 needs_reasoning
    if subtask_count == 1 and not has_dependencies:
        complexity = "simple"
        needs_reasoning = False
    elif has_dependencies:
        # 有依赖链的任务需要推理
        complexity = "high"
        needs_reasoning = True
    elif subtask_count > 1:
        complexity = "medium"
        needs_reasoning = False
    else:
        complexity = "simple"
        needs_reasoning = False

    # 记录执行时间
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    agent_timings = dict(state.get("agent_timings", {}))
    agent_timings["supervisor"] = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_ms": duration_ms,
    }

    return {
        "task_plan": {
            "task_type": plan.task_type.value,
            "complexity": complexity,
            "needs_decomposition": subtask_count > 1,
            "needs_reasoning": needs_reasoning,
            "subtasks": [
                {
                    "id": st.id,
                    "query": st.query,
                    "task_type": st.task_type.value,
                    "dependencies": st.dependencies,
                }
                for st in plan.subtasks
            ],
            "execution_order": plan.execution_order,
            "aggregation_strategy": plan.aggregation_strategy,
        },
        "current_agent": "supervisor",
        "next_agent": "data_sniper",
        "agent_timings": agent_timings,
    }


def route_after_supervisor(state: dict) -> str:
    """Supervisor 后的路由"""
    plan = state.get("execution_plan", {})

    if plan.get("needs_decomposition"):
        # 复杂查询，进入并行执行模式
        return "parallel_executor"
    else:
        # 简单查询，继续串行流程
        return "data_sniper"


__all__ = [
    "Supervisor",
    "ExecutionPlan",
    "SubTask",
    "TaskType",
    "AgentRole",
    "supervisor_node",
    "route_after_supervisor",
]
