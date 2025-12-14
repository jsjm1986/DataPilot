# -*- coding: utf-8 -*-
"""
并行执行器 - 支持子任务并行执行

实现复杂查询的并行处理：
1. 按层执行子任务 (同层并行)
2. 处理依赖关系
3. 聚合结果
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from ..agents import DataSniper, LogicArchitect, Judge
from ..db.connector import get_db_manager
from ..llm.deepseek import get_deepseek_client


@dataclass
class SubTaskResult:
    """子任务执行结果"""
    task_id: str
    success: bool
    sql: Optional[str] = None
    data: Optional[list[dict]] = None
    row_count: int = 0
    error: Optional[str] = None
    duration_ms: float = 0


class ParallelExecutor:
    """
    并行执行器

    负责执行 Supervisor 拆解后的子任务
    """

    def __init__(self, database: str = "default"):
        self.database = database
        self.llm = get_deepseek_client()

    async def execute_subtask(
        self,
        task_id: str,
        query: str,
        schema: str,
        context: Optional[dict] = None,
    ) -> SubTaskResult:
        """
        执行单个子任务

        Args:
            task_id: 子任务 ID
            query: 子任务查询
            schema: Schema 上下文
            context: 依赖任务的结果上下文

        Returns:
            子任务执行结果
        """
        start_time = datetime.utcnow()

        try:
            # 1. Data Sniper - Schema 剪枝
            sniper = DataSniper(self.database)
            sniper_result = await sniper.analyze(query)

            # 使用传入的 schema 或 sniper 获取的
            task_schema = schema or sniper_result.get("schema", "")

            # 2. Logic Architect - SQL 生成
            architect = LogicArchitect()

            # 如果有上下文，将其加入提示
            enhanced_query = query
            if context:
                context_info = self._format_context(context)
                enhanced_query = f"{query}\n\n参考上下文:\n{context_info}"

            candidates = await architect.generate_sql(
                query=enhanced_query,
                schema=task_schema,
                database=self.database,
                value_mappings=sniper_result.get("value_mappings"),
            )

            if not candidates:
                return SubTaskResult(
                    task_id=task_id,
                    success=False,
                    error="No SQL candidates generated",
                    duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                )

            winner_sql = candidates[0].get("sql")

            # 3. Judge - 校验和执行
            judge = Judge(self.database)
            judge_result = await judge.judge(winner_sql)

            if not judge_result.get("approved"):
                return SubTaskResult(
                    task_id=task_id,
                    success=False,
                    sql=winner_sql,
                    error=judge_result.get("reason", "SQL validation failed"),
                    duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                )

            # 4. 执行 SQL
            db_manager = get_db_manager()
            connector = db_manager.get(self.database)
            data = await connector.execute_query(winner_sql, limit=1000)

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SubTaskResult(
                task_id=task_id,
                success=True,
                sql=winner_sql,
                data=data,
                row_count=len(data),
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return SubTaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _format_context(self, context: dict) -> str:
        """格式化上下文信息"""
        parts = []
        for task_id, result in context.items():
            if result and result.get("success"):
                data = result.get("data", [])
                if data:
                    # 只取前几行作为上下文
                    sample = data[:5]
                    parts.append(f"子任务 {task_id} 结果 (前 {len(sample)} 行):\n{sample}")
        return "\n\n".join(parts)

    async def execute_plan(
        self,
        subtasks: list[dict],
        execution_order: list[list[str]],
        schema: str,
    ) -> dict[str, SubTaskResult]:
        """
        执行完整的执行计划

        Args:
            subtasks: 子任务列表
            execution_order: 执行顺序 (分层)
            schema: Schema 上下文

        Returns:
            所有子任务的结果
        """
        results: dict[str, SubTaskResult] = {}
        task_map = {t["id"]: t for t in subtasks}

        for layer in execution_order:
            # 收集本层要执行的任务
            tasks_to_run = []

            for task_id in layer:
                task = task_map.get(task_id)
                if not task:
                    continue

                # 检查依赖是否满足
                dependencies = task.get("dependencies", [])
                deps_satisfied = all(
                    dep in results and results[dep].success
                    for dep in dependencies
                )

                if not deps_satisfied:
                    # 依赖未满足，标记失败
                    results[task_id] = SubTaskResult(
                        task_id=task_id,
                        success=False,
                        error="Dependencies not satisfied",
                    )
                    continue

                # 构建上下文
                context = {
                    dep: {
                        "success": results[dep].success,
                        "data": results[dep].data,
                        "sql": results[dep].sql,
                    }
                    for dep in dependencies
                    if dep in results
                }

                tasks_to_run.append((task_id, task["query"], context))

            # 并行执行本层任务
            if tasks_to_run:
                layer_results = await asyncio.gather(*[
                    self.execute_subtask(
                        task_id=task_id,
                        query=query,
                        schema=schema,
                        context=context if context else None,
                    )
                    for task_id, query, context in tasks_to_run
                ], return_exceptions=True)

                # 收集结果
                for (task_id, _, _), result in zip(tasks_to_run, layer_results):
                    if isinstance(result, Exception):
                        results[task_id] = SubTaskResult(
                            task_id=task_id,
                            success=False,
                            error=str(result),
                        )
                    else:
                        results[task_id] = result

        return results

    async def aggregate_results(
        self,
        results: dict[str, SubTaskResult],
        strategy: str,
        original_query: str,
    ) -> dict:
        """
        聚合子任务结果

        Args:
            results: 子任务结果
            strategy: 聚合策略 (union/join/merge/none)
            original_query: 原始查询

        Returns:
            聚合后的结果
        """
        successful_results = {
            k: v for k, v in results.items() if v.success
        }

        if not successful_results:
            return {
                "success": False,
                "error": "All subtasks failed",
                "subtask_errors": {k: v.error for k, v in results.items()},
            }

        if strategy == "none" or len(successful_results) == 1:
            # 单任务或无需聚合
            first_result = list(successful_results.values())[0]
            return {
                "success": True,
                "sql": first_result.sql,
                "data": first_result.data,
                "row_count": first_result.row_count,
            }

        if strategy == "union":
            # 合并所有数据
            all_data = []
            all_sqls = []
            for result in successful_results.values():
                if result.data:
                    all_data.extend(result.data)
                if result.sql:
                    all_sqls.append(result.sql)

            return {
                "success": True,
                "sql": " UNION ALL ".join(f"({sql})" for sql in all_sqls),
                "data": all_data,
                "row_count": len(all_data),
                "aggregation": "union",
            }

        if strategy == "join":
            # 使用 LLM 智能合并
            return await self._llm_aggregate(successful_results, original_query)

        if strategy == "merge":
            # 保留所有结果
            return {
                "success": True,
                "merged_results": {
                    k: {
                        "sql": v.sql,
                        "data": v.data,
                        "row_count": v.row_count,
                    }
                    for k, v in successful_results.items()
                },
                "aggregation": "merge",
            }

        # 默认返回所有结果
        return {
            "success": True,
            "results": {k: v.__dict__ for k, v in successful_results.items()},
        }

    async def _llm_aggregate(
        self,
        results: dict[str, SubTaskResult],
        original_query: str,
    ) -> dict:
        """使用 LLM 智能聚合结果"""
        import json

        # 构建聚合提示
        results_summary = []
        for task_id, result in results.items():
            results_summary.append({
                "task_id": task_id,
                "sql": result.sql,
                "row_count": result.row_count,
                "sample_data": result.data[:5] if result.data else [],
            })

        prompt = f"""请将以下子查询结果合并成最终答案。

原始查询: {original_query}

子查询结果:
{json.dumps(results_summary, ensure_ascii=False, indent=2)}

请分析这些结果，生成一个合并后的数据集。
返回 JSON 格式:
```json
{{
    "merged_data": [...],
    "summary": "结果摘要",
    "insights": ["洞察1", "洞察2"]
}}
```
"""

        messages = [
            {"role": "system", "content": "你是一个数据分析专家，负责合并和分析多个查询结果。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.llm.chat(messages)
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "success": True,
                    "data": parsed.get("merged_data", []),
                    "row_count": len(parsed.get("merged_data", [])),
                    "summary": parsed.get("summary"),
                    "insights": parsed.get("insights"),
                    "aggregation": "llm_join",
                }
        except Exception as e:
            pass

        # 回退到简单合并
        all_data = []
        for result in results.values():
            if result.data:
                all_data.extend(result.data)

        return {
            "success": True,
            "data": all_data,
            "row_count": len(all_data),
            "aggregation": "fallback_union",
        }


async def parallel_executor_node(state: dict) -> dict:
    """
    并行执行器节点

    当 Supervisor 判断需要拆解时，执行此节点
    """
    import time
    start_time = time.time()

    task_plan = state.get("task_plan")
    if not task_plan or not task_plan.get("needs_decomposition"):
        # 不需要并行执行，跳过
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        agent_timings = dict(state.get("agent_timings", {}))
        agent_timings["parallel_executor"] = {
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_ms": duration_ms,
        }
        return {
            "current_agent": "parallel_executor",
            "next_agent": "data_sniper",
            "agent_timings": agent_timings,
        }

    database = state.get("database", "default")
    schema = state.get("schema_context", "")

    executor = ParallelExecutor(database)

    # 执行计划
    results = await executor.execute_plan(
        subtasks=task_plan.get("subtasks", []),
        execution_order=task_plan.get("execution_order", []),
        schema=schema,
    )

    # 聚合结果
    aggregated = await executor.aggregate_results(
        results=results,
        strategy=task_plan.get("aggregation_strategy", "none"),
        original_query=state.get("query", ""),
    )

    # 记录执行时间
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    agent_timings = dict(state.get("agent_timings", {}))
    agent_timings["parallel_executor"] = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_ms": duration_ms,
    }

    # 更新状态
    return {
        "subtask_results": {k: v.__dict__ for k, v in results.items()},
        "winner_sql": aggregated.get("sql"),
        "execution_result": {
            "success": aggregated.get("success", False),
            "data": aggregated.get("data", []),
            "row_count": aggregated.get("row_count", 0),
            "columns": list(aggregated.get("data", [{}])[0].keys()) if aggregated.get("data") else [],
            "execution_time_ms": sum(r.duration_ms for r in results.values()),
            "error": aggregated.get("error"),
        } if aggregated.get("success") else None,
        "current_agent": "parallel_executor",
        "next_agent": "viz_expert" if aggregated.get("success") else "__end__",
        "last_error": aggregated.get("error"),
        "agent_timings": agent_timings,
    }


__all__ = [
    "ParallelExecutor",
    "SubTaskResult",
    "parallel_executor_node",
]
