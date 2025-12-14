# -*- coding: utf-8 -*-
"""
Supervisor ReAct Mode - 管理智能体的推理执行模式

核心思想:
1. Supervisor 作为中央协调器，具备 ReAct (Reasoning + Acting) 能力
2. 可以调用所有专业 Agent (DataSniper, LogicArchitect, Judge, VizExpert)
3. 支持多轮推理，根据中间结果动态调整策略
4. 统一处理简单/中等/复杂查询的失败回退

触发条件:
- 简单查询: Judge 3次重试失败后触发
- 中等查询: 并行执行结果不完整时触发
- 复杂查询: 直接使用 Supervisor ReAct 模式
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from datetime import datetime

from ..llm.deepseek import get_deepseek_client
from ..db.connector import get_db_manager


class SupervisorAction(Enum):
    """Supervisor 可执行的动作"""
    ANALYZE_SCHEMA = "analyze_schema"      # 调用 DataSniper 分析 Schema
    GENERATE_SQL = "generate_sql"          # 调用 LogicArchitect 生成 SQL
    VALIDATE_SQL = "validate_sql"          # 调用 Judge 校验 SQL
    EXECUTE_SQL = "execute_sql"            # 执行 SQL 查询
    REFINE_SQL = "refine_sql"              # 修正 SQL
    VISUALIZE = "visualize"                # 调用 VizExpert 生成图表
    DECOMPOSE_TASK = "decompose_task"      # 重新拆解任务
    AGGREGATE_RESULTS = "aggregate_results" # 聚合多个结果
    FINISH = "finish"                      # 完成任务


@dataclass
class ReActStep:
    """ReAct 推理步骤"""
    step_num: int
    thought: str
    action: SupervisorAction
    action_input: dict
    observation: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class SupervisorReActState:
    """Supervisor ReAct 状态"""
    query: str
    database: str
    schema: str
    trigger_reason: str  # 触发原因: complex_query, judge_failed, incomplete_result
    previous_context: dict = field(default_factory=dict)  # 之前的执行上下文
    steps: list[ReActStep] = field(default_factory=list)
    intermediate_results: dict = field(default_factory=dict)
    final_sql: Optional[str] = None
    final_result: Optional[list] = None
    is_complete: bool = False
    error: Optional[str] = None
    max_iterations: int = 5  # 最大迭代次数


SUPERVISOR_REACT_PROMPT = """你是 DataPilot 的 Supervisor (管理智能体)，现在进入 ReAct 推理模式。

## 背景
你负责协调多个专业 Agent 来解决复杂的数据库查询问题。

## 触发原因
{trigger_reason}

## 之前的上下文
{previous_context}

## 可用动作

1. **analyze_schema**: 调用 DataSniper 分析数据库 Schema
   - 输入: {{"focus": "关注的方面"}}
   - 输出: 相关表、列、时间字段、枚举值、值映射

2. **generate_sql**: 调用 LogicArchitect 生成 SQL
   - 输入: {{"intent": "查询意图", "context": "额外上下文"}}
   - 输出: SQL 语句、解释、置信度

3. **validate_sql**: 调用 Judge 校验 SQL
   - 输入: {{"sql": "SQL语句"}}
   - 输出: 是否有效、原因、修复建议

4. **execute_sql**: 执行 SQL 查询
   - 输入: {{"sql": "SQL语句", "limit": 100}}
   - 输出: 查询结果

5. **refine_sql**: 修正 SQL (基于错误信息)
   - 输入: {{"original_sql": "原SQL", "error": "错误信息", "hint": "修正提示"}}
   - 输出: 修正后的 SQL

6. **decompose_task**: 重新拆解任务
   - 输入: {{"reason": "拆解原因", "strategy": "拆解策略"}}
   - 输出: 子任务列表

7. **aggregate_results**: 聚合多个结果
   - 输入: {{"results": [...], "strategy": "union|join|merge"}}
   - 输出: 聚合后的结果

8. **finish**: 完成任务
   - 输入: {{"sql": "最终SQL", "answer": "回答摘要"}}
   - **重要**: 必须在 sql 字段中提供最终 SQL!

## 推荐工作流程

### 复杂查询 (首次尝试):
1. analyze_schema → 理解数据结构
2. generate_sql → 生成 SQL
3. validate_sql → 校验 SQL
4. execute_sql → 执行并检查结果
5. finish → 返回结果

### 失败回退 (Judge 失败后):
1. 分析之前的错误
2. refine_sql → 修正 SQL
3. validate_sql → 重新校验
4. execute_sql → 执行
5. finish → 返回结果

### 结果不完整 (并行执行后):
1. 分析缺失的部分
2. generate_sql → 生成补充查询
3. execute_sql → 执行
4. aggregate_results → 合并结果
5. finish → 返回结果

## 当前状态

用户查询: {query}
数据库: {database}

Schema:
{schema}

历史步骤:
{history}

中间结果:
{intermediate_results}

## 输出格式
每一步输出 JSON:
```json
{{
    "thought": "我的思考过程...",
    "action": "动作名称",
    "action_input": {{...}}
}}
```

## 重要提醒
1. **必须调用 finish**: 任务完成时必须使用 finish 动作
2. **finish 必须包含 sql**: finish 的 action_input 必须包含 "sql" 字段
3. **复用已有结果**: 如果之前已经生成了 SQL，可以直接使用
4. **分析错误原因**: 失败时先分析原因，再决定下一步
"""


class SupervisorReAct:
    """
    Supervisor ReAct 模式

    让 Supervisor 具备多轮推理能力，可以调用各个专业 Agent
    """

    def __init__(self, database: str = "default"):
        self.database = database
        self.llm = get_deepseek_client()
        self.agents = {}  # 延迟加载的 Agent 实例

    def _get_agent(self, agent_name: str):
        """延迟加载 Agent"""
        if agent_name not in self.agents:
            if agent_name == "data_sniper":
                from ..agents import DataSniper
                self.agents[agent_name] = DataSniper(self.database)
            elif agent_name == "logic_architect":
                from ..agents import LogicArchitect
                self.agents[agent_name] = LogicArchitect()
            elif agent_name == "judge":
                from ..agents import Judge
                self.agents[agent_name] = Judge(self.database)
            elif agent_name == "viz_expert":
                from ..agents import VizExpert
                self.agents[agent_name] = VizExpert()
        return self.agents.get(agent_name)

    async def run(
        self,
        query: str,
        schema: str,
        trigger_reason: str = "complex_query",
        previous_context: dict = None,
    ) -> SupervisorReActState:
        """
        执行 Supervisor ReAct 推理循环

        Args:
            query: 用户查询
            schema: 数据库 Schema
            trigger_reason: 触发原因
            previous_context: 之前的执行上下文 (用于失败回退)

        Returns:
            SupervisorReActState: 最终状态
        """
        state = SupervisorReActState(
            query=query,
            database=self.database,
            schema=schema,
            trigger_reason=trigger_reason,
            previous_context=previous_context or {},
        )

        # 如果有之前的 SQL，保存到中间结果
        if previous_context:
            if previous_context.get("winner_sql"):
                state.intermediate_results["previous_sql"] = previous_context["winner_sql"]
            if previous_context.get("last_error"):
                state.intermediate_results["previous_error"] = previous_context["last_error"]

        iteration = 0
        while not state.is_complete and iteration < state.max_iterations:
            iteration += 1

            # 获取下一步动作
            step = await self._get_next_step(state)
            if step is None:
                state.error = "Failed to get next step"
                break

            state.steps.append(step)

            # 执行动作
            observation = await self._execute_action(step, state)
            step.observation = observation

            # 检查是否完成
            if step.action == SupervisorAction.FINISH:
                state.is_complete = True
                if "sql" in step.action_input:
                    state.final_sql = step.action_input["sql"]
                elif state.intermediate_results.get("generated_sql"):
                    state.final_sql = state.intermediate_results["generated_sql"]

        # 最终回退: 从中间结果获取 SQL
        if not state.final_sql and not state.error:
            if state.intermediate_results.get("generated_sql"):
                state.final_sql = state.intermediate_results["generated_sql"]
            elif state.intermediate_results.get("refined_sql"):
                state.final_sql = state.intermediate_results["refined_sql"]

        return state

    async def _get_next_step(self, state: SupervisorReActState) -> Optional[ReActStep]:
        """获取下一步动作"""
        # 构建 prompt
        history = self._format_history(state.steps)
        intermediate = json.dumps(state.intermediate_results, ensure_ascii=False, default=str)[:2000]
        previous_ctx = json.dumps(state.previous_context, ensure_ascii=False, default=str)[:1000]

        prompt = SUPERVISOR_REACT_PROMPT.format(
            trigger_reason=state.trigger_reason,
            previous_context=previous_ctx,
            query=state.query,
            database=state.database,
            schema=state.schema[:3000],  # 限制 Schema 长度
            history=history,
            intermediate_results=intermediate,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请分析当前状态，决定下一步动作。"},
        ]

        try:
            response = await self.llm.chat(messages)
            return self._parse_step(response, len(state.steps) + 1)
        except Exception as e:
            print(f"Supervisor ReAct get_next_step error: {e}")
            return None

    def _format_history(self, steps: list[ReActStep]) -> str:
        """格式化历史步骤"""
        if not steps:
            return "无"

        lines = []
        for step in steps:
            lines.append(f"【Step {step.step_num}】")
            lines.append(f"  思考: {step.thought}")
            lines.append(f"  动作: {step.action.value}")
            lines.append(f"  输入: {json.dumps(step.action_input, ensure_ascii=False)}")
            if step.observation:
                obs = step.observation[:500] if len(step.observation) > 500 else step.observation
                lines.append(f"  观察: {obs}")
        return "\n".join(lines)

    def _parse_step(self, response: str, step_num: int) -> Optional[ReActStep]:
        """解析 LLM 响应"""
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
            action_str = data.get("action", "").lower()

            # 映射动作
            action_map = {
                "analyze_schema": SupervisorAction.ANALYZE_SCHEMA,
                "generate_sql": SupervisorAction.GENERATE_SQL,
                "validate_sql": SupervisorAction.VALIDATE_SQL,
                "execute_sql": SupervisorAction.EXECUTE_SQL,
                "refine_sql": SupervisorAction.REFINE_SQL,
                "visualize": SupervisorAction.VISUALIZE,
                "decompose_task": SupervisorAction.DECOMPOSE_TASK,
                "aggregate_results": SupervisorAction.AGGREGATE_RESULTS,
                "finish": SupervisorAction.FINISH,
            }

            action = action_map.get(action_str, SupervisorAction.FINISH)

            return ReActStep(
                step_num=step_num,
                thought=data.get("thought", ""),
                action=action,
                action_input=data.get("action_input", {}),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    async def _execute_action(self, step: ReActStep, state: SupervisorReActState) -> str:
        """执行动作"""
        action = step.action
        action_input = step.action_input

        try:
            if action == SupervisorAction.ANALYZE_SCHEMA:
                return await self._action_analyze_schema(action_input, state)

            elif action == SupervisorAction.GENERATE_SQL:
                return await self._action_generate_sql(action_input, state)

            elif action == SupervisorAction.VALIDATE_SQL:
                return await self._action_validate_sql(action_input, state)

            elif action == SupervisorAction.EXECUTE_SQL:
                return await self._action_execute_sql(action_input, state)

            elif action == SupervisorAction.REFINE_SQL:
                return await self._action_refine_sql(action_input, state)

            elif action == SupervisorAction.VISUALIZE:
                return await self._action_visualize(action_input, state)

            elif action == SupervisorAction.DECOMPOSE_TASK:
                return await self._action_decompose_task(action_input, state)

            elif action == SupervisorAction.AGGREGATE_RESULTS:
                return await self._action_aggregate_results(action_input, state)

            elif action == SupervisorAction.FINISH:
                return self._action_finish(action_input, state)

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"Action execution error: {str(e)}"

    async def _action_analyze_schema(self, action_input: dict, state: SupervisorReActState) -> str:
        """调用 DataSniper 分析 Schema"""
        sniper = self._get_agent("data_sniper")
        result = await sniper.analyze(state.query)

        # 保存到中间结果
        state.intermediate_results["schema_analysis"] = {
            "tables": result.get("relevant_tables", []),
            "time_context": result.get("time_context", ""),
            "enum_context": result.get("enum_context", ""),
            "value_mappings": result.get("value_mappings", {}),
        }

        return json.dumps({
            "success": True,
            "tables": [t["name"] if isinstance(t, dict) else t for t in result.get("relevant_tables", [])],
            "time_fields": result.get("time_context", "")[:200],
            "enum_fields": result.get("enum_context", "")[:200],
        }, ensure_ascii=False)

    async def _action_generate_sql(self, action_input: dict, state: SupervisorReActState) -> str:
        """调用 LogicArchitect 生成 SQL"""
        architect = self._get_agent("logic_architect")

        # 获取上下文
        schema_info = state.intermediate_results.get("schema_analysis", {})

        candidates = await architect.generate_sql(
            query=action_input.get("intent", state.query),
            schema=state.schema,
            database=state.database,
            value_mappings=schema_info.get("value_mappings"),
            time_context=schema_info.get("time_context", ""),
            enum_context=schema_info.get("enum_context", ""),
        )

        if candidates:
            winner = candidates[0]
            sql = winner.get("sql", "")
            state.intermediate_results["generated_sql"] = sql
            return json.dumps({
                "success": True,
                "sql": sql,
                "explanation": winner.get("explanation", "")[:200],
                "confidence": winner.get("confidence", 0),
            }, ensure_ascii=False)

        return json.dumps({"success": False, "error": "Failed to generate SQL"}, ensure_ascii=False)

    async def _action_validate_sql(self, action_input: dict, state: SupervisorReActState) -> str:
        """调用 Judge 校验 SQL"""
        judge = self._get_agent("judge")
        sql = action_input.get("sql", state.intermediate_results.get("generated_sql", ""))

        if not sql:
            return json.dumps({"success": False, "error": "No SQL to validate"}, ensure_ascii=False)

        result = await judge.judge(sql)

        return json.dumps({
            "approved": result.get("approved", False),
            "reason": result.get("reason", ""),
            "suggestions": result.get("suggestions", [])[:3],
        }, ensure_ascii=False)

    async def _action_execute_sql(self, action_input: dict, state: SupervisorReActState) -> str:
        """执行 SQL 查询"""
        sql = action_input.get("sql", state.intermediate_results.get("generated_sql", ""))
        limit = action_input.get("limit", 100)

        if not sql:
            return json.dumps({"success": False, "error": "No SQL to execute"}, ensure_ascii=False)

        try:
            db_manager = get_db_manager()
            connector = db_manager.get(state.database)
            data = await connector.execute_query(sql, limit=limit)

            state.intermediate_results["execution_result"] = {
                "success": True,
                "data": data,
                "row_count": len(data),
            }
            state.final_result = data

            return json.dumps({
                "success": True,
                "row_count": len(data),
                "columns": list(data[0].keys()) if data else [],
                "sample": data[:3] if len(data) > 3 else data,
            }, ensure_ascii=False, default=str)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
            }, ensure_ascii=False)

    async def _action_refine_sql(self, action_input: dict, state: SupervisorReActState) -> str:
        """修正 SQL"""
        architect = self._get_agent("logic_architect")

        original_sql = action_input.get("original_sql", state.intermediate_results.get("generated_sql", ""))
        error = action_input.get("error", state.intermediate_results.get("previous_error", ""))
        hint = action_input.get("hint", "")

        if not original_sql:
            return json.dumps({"success": False, "error": "No SQL to refine"}, ensure_ascii=False)

        # 使用 LogicArchitect 的 refine 能力
        try:
            candidate = await architect.refine_sql(
                original_sql=original_sql,
                error_message=error,
                schema=state.schema,
                database=state.database,
                hint=hint,
            )

            if candidate:
                sql = candidate.get("sql", "")
                state.intermediate_results["refined_sql"] = sql
                return json.dumps({
                    "success": True,
                    "sql": sql,
                    "explanation": candidate.get("explanation", ""),
                }, ensure_ascii=False)
        except Exception as e:
            pass

        return json.dumps({"success": False, "error": "Failed to refine SQL"}, ensure_ascii=False)

    async def _action_visualize(self, action_input: dict, state: SupervisorReActState) -> str:
        """调用 VizExpert 生成图表配置"""
        viz = self._get_agent("viz_expert")
        data = state.final_result or state.intermediate_results.get("execution_result", {}).get("data", [])

        if not data:
            return json.dumps({"success": False, "error": "No data to visualize"}, ensure_ascii=False)

        analysis = await viz.analyze(data, state.query)

        state.intermediate_results["chart_config"] = analysis

        return json.dumps({
            "success": True,
            "chart_type": analysis.get("chart_type"),
            "config": analysis.get("config"),
        }, ensure_ascii=False)

    async def _action_decompose_task(self, action_input: dict, state: SupervisorReActState) -> str:
        """重新拆解任务"""
        from .supervisor import Supervisor

        supervisor = Supervisor()
        plan = await supervisor.analyze_and_plan(state.query, state.schema)

        state.intermediate_results["decomposed_plan"] = {
            "subtasks": [
                {"id": st.id, "query": st.query, "dependencies": st.dependencies}
                for st in plan.subtasks
            ],
            "execution_order": plan.execution_order,
        }

        return json.dumps({
            "success": True,
            "subtask_count": len(plan.subtasks),
            "subtasks": [st.query for st in plan.subtasks],
        }, ensure_ascii=False)

    async def _action_aggregate_results(self, action_input: dict, state: SupervisorReActState) -> str:
        """聚合多个结果"""
        results = action_input.get("results", [])
        strategy = action_input.get("strategy", "union")

        if not results:
            # 尝试从中间结果获取
            results = [
                state.intermediate_results.get("execution_result", {}),
            ]

        if strategy == "union":
            all_data = []
            for r in results:
                if isinstance(r, dict) and r.get("data"):
                    all_data.extend(r["data"])
            state.final_result = all_data
            return json.dumps({
                "success": True,
                "row_count": len(all_data),
                "strategy": "union",
            }, ensure_ascii=False)

        return json.dumps({"success": True, "strategy": strategy}, ensure_ascii=False)

    def _action_finish(self, action_input: dict, state: SupervisorReActState) -> str:
        """完成任务"""
        sql = action_input.get("sql")
        answer = action_input.get("answer", "")

        if sql:
            state.final_sql = sql
        elif state.intermediate_results.get("generated_sql"):
            state.final_sql = state.intermediate_results["generated_sql"]
        elif state.intermediate_results.get("refined_sql"):
            state.final_sql = state.intermediate_results["refined_sql"]

        return json.dumps({
            "finished": True,
            "sql": state.final_sql,
            "answer": answer,
        }, ensure_ascii=False)


# ============================================
# 集成到 LangGraph 工作流的节点函数
# ============================================

async def supervisor_react_node(state: dict) -> dict:
    """
    Supervisor ReAct 节点 - 用于复杂查询和失败回退

    触发条件:
    1. 复杂查询 (complexity == "high")
    2. Judge 失败回退 (retries >= 3)
    3. 并行执行结果不完整
    """
    import time
    start_time = time.time()

    query = state["query"]
    database = state.get("database", "default")
    schema = state.get("schema_context", "")

    # 确定触发原因
    trigger_reason = state.get("supervisor_react_trigger", "complex_query")

    # 收集之前的上下文
    previous_context = {
        "winner_sql": state.get("winner_sql"),
        "last_error": state.get("last_error"),
        "candidates": state.get("candidates", []),
        "execution_result": state.get("execution_result"),
    }

    # 如果没有 schema，获取它
    if not schema:
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        schema = await connector.get_schema()

    # 执行 Supervisor ReAct
    supervisor_react = SupervisorReAct(database)
    react_state = await supervisor_react.run(
        query=query,
        schema=schema,
        trigger_reason=trigger_reason,
        previous_context=previous_context,
    )

    # 构建返回结果
    result = {
        "current_agent": "supervisor_react",
        "supervisor_react_steps": len(react_state.steps),
    }

    if react_state.final_sql:
        result["winner_sql"] = react_state.final_sql

        # 如果有执行结果
        if react_state.final_result:
            result["execution_result"] = {
                "success": True,
                "data": react_state.final_result,
                "row_count": len(react_state.final_result),
                "columns": list(react_state.final_result[0].keys()) if react_state.final_result else [],
                "execution_time_ms": 0,
                "error": None,
            }
            result["next_agent"] = "viz_expert"
        else:
            # 有 SQL 但没执行，需要执行
            try:
                db_manager = get_db_manager()
                connector = db_manager.get(database)
                data = await connector.execute_query(react_state.final_sql, limit=1000)
                result["execution_result"] = {
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "columns": list(data[0].keys()) if data else [],
                    "execution_time_ms": 0,
                    "error": None,
                }
                result["next_agent"] = "viz_expert"
            except Exception as e:
                result["execution_result"] = {
                    "success": False,
                    "data": [],
                    "row_count": 0,
                    "columns": [],
                    "error": str(e),
                }
                result["next_agent"] = "__end__"
                result["last_error"] = str(e)
    else:
        # 没有生成 SQL
        result["next_agent"] = "__end__"
        result["last_error"] = react_state.error or "Supervisor ReAct failed to generate SQL"

    # 保存图表配置
    if react_state.intermediate_results.get("chart_config"):
        result["chart_config"] = react_state.intermediate_results["chart_config"]

    # 记录执行时间
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    agent_timings = dict(state.get("agent_timings", {}))
    agent_timings["supervisor_react"] = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_ms": duration_ms,
    }
    result["agent_timings"] = agent_timings

    return result


__all__ = [
    "SupervisorReAct",
    "SupervisorReActState",
    "SupervisorAction",
    "ReActStep",
    "supervisor_react_node",
]
