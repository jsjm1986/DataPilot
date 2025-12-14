# -*- coding: utf-8 -*-
"""
ReAct Agent - 推理-行动循环

实现真正的多轮推理能力：
1. Thought: 分析当前状态，决定下一步
2. Action: 执行工具调用
3. Observation: 观察结果
4. 循环直到完成

解决的问题：
- 复杂查询的多轮推理
- 动态工具选择
- 自我修正能力
- 依赖链处理
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from datetime import datetime

from ..llm.deepseek import get_deepseek_client


class ActionType(Enum):
    """可用的动作类型"""
    ANALYZE_SCHEMA = "analyze_schema"       # 分析 Schema
    EXTRACT_ENTITIES = "extract_entities"   # 提取实体
    MAP_VALUES = "map_values"               # 值映射
    GENERATE_SQL = "generate_sql"           # 生成 SQL
    VALIDATE_SQL = "validate_sql"           # 校验 SQL
    EXECUTE_SQL = "execute_sql"             # 执行 SQL
    ANALYZE_RESULT = "analyze_result"       # 分析结果
    REFINE_SQL = "refine_sql"               # 修正 SQL
    ASK_CLARIFICATION = "ask_clarification" # 请求澄清
    DECOMPOSE_QUERY = "decompose_query"     # 分解查询
    AGGREGATE_RESULTS = "aggregate_results" # 聚合结果
    FINISH = "finish"                       # 完成


@dataclass
class ThoughtStep:
    """推理步骤"""
    step_num: int
    thought: str
    action: ActionType
    action_input: dict
    observation: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class ReActState:
    """ReAct 状态"""
    query: str
    schema: str
    database: str
    steps: list[ThoughtStep] = field(default_factory=list)
    intermediate_results: dict = field(default_factory=dict)
    final_sql: Optional[str] = None
    final_result: Optional[list] = None
    is_complete: bool = False
    error: Optional[str] = None
    max_steps: int = 10


REACT_SYSTEM_PROMPT = """你是一个智能 SQL 生成助手，使用 ReAct (Reasoning + Acting) 模式解决问题。

## 工作流程
每一步你需要：
1. **Thought**: 分析当前状态，思考下一步该做什么
2. **Action**: 选择一个动作执行
3. **Observation**: 观察动作结果（系统会提供）

## 可用动作

1. **analyze_schema**: 分析数据库 Schema，理解表结构和关系
   - 输入: {"focus": "要关注的方面"}
   - 用途: 理解数据模型，找出相关表

2. **extract_entities**: 从查询中提取实体
   - 输入: {"query": "用户查询"}
   - 用途: 识别查询中的关键实体（表名、列名、值）

3. **map_values**: 将用户术语映射到数据库值
   - 输入: {"term": "用户术语", "candidates": ["候选值"]}
   - 用途: 模糊匹配，如 "苹果手机" → "iPhone"

4. **generate_sql**: 生成 SQL 查询
   - 输入: {"intent": "查询意图", "tables": ["相关表"], "conditions": {...}}
   - 用途: 根据分析结果生成 SQL

5. **validate_sql**: 校验 SQL 语法和安全性
   - 输入: {"sql": "SQL语句"}
   - 用途: 检查 SQL 是否正确

6. **execute_sql**: 执行 SQL 并获取结果
   - 输入: {"sql": "SQL语句", "limit": 100}
   - 用途: 执行查询获取数据

7. **analyze_result**: 分析查询结果
   - 输入: {"result": [...], "question": "要回答的问题"}
   - 用途: 检查结果是否符合预期

8. **refine_sql**: 根据错误修正 SQL
   - 输入: {"original_sql": "原SQL", "error": "错误信息", "hint": "修正提示"}
   - 用途: 针对性修复 SQL 错误

9. **ask_clarification**: 请求用户澄清
   - 输入: {"question": "问题", "options": ["选项"]}
   - 用途: 当查询有歧义时请求澄清

10. **decompose_query**: 分解复杂查询
    - 输入: {"query": "复杂查询", "reason": "分解原因"}
    - 用途: 将复杂查询拆分成子查询

11. **aggregate_results**: 聚合多个子查询结果
    - 输入: {"results": {...}, "strategy": "聚合策略"}
    - 用途: 合并子查询结果

12. **finish**: 完成任务
    - 输入: {"sql": "最终SQL", "answer": "回答"}
    - 用途: 任务完成，返回结果

## 输出格式
每一步输出 JSON:
```json
{
    "thought": "我的思考过程...",
    "action": "动作名称",
    "action_input": {...}
}
```

## 重要原则
1. 先理解再行动：先分析 Schema 和查询，再生成 SQL
2. 逐步验证：生成 SQL 后先校验再执行
3. 错误修正：执行失败时分析原因并修正
4. 知道何时停止：达到目标或无法继续时使用 finish
5. 不要重复相同的动作：如果一个动作失败，尝试不同的方法
"""


class ReActAgent:
    """
    ReAct Agent - 推理-行动循环

    实现多轮推理，支持复杂查询处理
    """

    def __init__(self, database: str = "default"):
        self.database = database
        self.llm = get_deepseek_client()
        self.tools: dict[ActionType, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        # 工具将在运行时注入
        pass

    def register_tool(self, action: ActionType, func: Callable):
        """注册工具"""
        self.tools[action] = func

    async def run(
        self,
        query: str,
        schema: str,
        tools: Optional[dict[ActionType, Callable]] = None,
    ) -> ReActState:
        """
        运行 ReAct 循环

        Args:
            query: 用户查询
            schema: 数据库 Schema
            tools: 工具函数映射

        Returns:
            最终状态
        """
        if tools:
            self.tools.update(tools)

        state = ReActState(
            query=query,
            schema=schema,
            database=self.database,
        )

        # 构建初始上下文
        context = self._build_context(state)

        while not state.is_complete and len(state.steps) < state.max_steps:
            # 1. 获取下一步动作
            step = await self._get_next_step(state, context)

            if step is None:
                state.error = "Failed to get next step"
                break

            state.steps.append(step)

            # 2. 执行动作
            observation = await self._execute_action(step, state)
            step.observation = observation

            # 3. 检查是否完成
            if step.action == ActionType.FINISH:
                state.is_complete = True
                if "sql" in step.action_input:
                    state.final_sql = step.action_input["sql"]

            # 4. 更新上下文
            context = self._build_context(state)

        return state

    def _build_context(self, state: ReActState) -> str:
        """构建上下文"""
        parts = [
            f"用户查询: {state.query}",
            f"\n数据库 Schema:\n{state.schema[:3000]}",
        ]

        if state.steps:
            parts.append("\n历史步骤:")
            for step in state.steps[-5:]:  # 只保留最近5步
                parts.append(f"\nStep {step.step_num}:")
                parts.append(f"  Thought: {step.thought}")
                parts.append(f"  Action: {step.action.value}")
                parts.append(f"  Input: {json.dumps(step.action_input, ensure_ascii=False)[:200]}")
                if step.observation:
                    parts.append(f"  Observation: {step.observation[:500]}")

        if state.intermediate_results:
            parts.append(f"\n中间结果: {json.dumps(state.intermediate_results, ensure_ascii=False, default=str)[:1000]}")

        return "\n".join(parts)

    async def _get_next_step(self, state: ReActState, context: str) -> Optional[ThoughtStep]:
        """获取下一步动作"""
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n请决定下一步动作:"},
        ]

        try:
            response = await self.llm.chat(messages)
            return self._parse_step(response, len(state.steps) + 1)
        except Exception as e:
            return ThoughtStep(
                step_num=len(state.steps) + 1,
                thought=f"Error: {str(e)}",
                action=ActionType.FINISH,
                action_input={"error": str(e)},
            )

    def _parse_step(self, response: str, step_num: int) -> Optional[ThoughtStep]:
        """解析 LLM 响应"""
        import re

        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())

            action_str = data.get("action", "finish").lower()
            try:
                action = ActionType(action_str)
            except ValueError:
                action = ActionType.FINISH

            return ThoughtStep(
                step_num=step_num,
                thought=data.get("thought", ""),
                action=action,
                action_input=data.get("action_input", {}),
            )

        except (json.JSONDecodeError, KeyError):
            return None

    async def _execute_action(self, step: ThoughtStep, state: ReActState) -> str:
        """执行动作"""
        action = step.action
        action_input = step.action_input

        # 检查是否有注册的工具
        if action in self.tools:
            try:
                result = await self.tools[action](action_input, state)
                return json.dumps(result, ensure_ascii=False, default=str)[:2000]
            except Exception as e:
                return f"Error executing {action.value}: {str(e)}"

        # 默认处理
        if action == ActionType.FINISH:
            return "Task completed"

        if action == ActionType.ANALYZE_SCHEMA:
            return f"Schema analyzed. Found tables and relationships in the schema."

        if action == ActionType.GENERATE_SQL:
            # 使用 LLM 生成 SQL
            sql = await self._generate_sql(action_input, state)
            state.intermediate_results["generated_sql"] = sql
            return f"Generated SQL: {sql}"

        if action == ActionType.REFINE_SQL:
            # 使用 LLM 修正 SQL
            sql = await self._refine_sql(action_input, state)
            state.intermediate_results["refined_sql"] = sql
            return f"Refined SQL: {sql}"

        if action == ActionType.DECOMPOSE_QUERY:
            # 分解查询
            subtasks = await self._decompose_query(action_input, state)
            state.intermediate_results["subtasks"] = subtasks
            return f"Decomposed into {len(subtasks)} subtasks: {subtasks}"

        return f"Action {action.value} executed with input: {action_input}"

    async def _generate_sql(self, action_input: dict, state: ReActState) -> str:
        """生成 SQL"""
        prompt = f"""根据以下信息生成 SQL:

查询意图: {action_input.get('intent', state.query)}
相关表: {action_input.get('tables', [])}
条件: {action_input.get('conditions', {})}

Schema:
{state.schema[:2000]}

只返回 SQL 语句，不要其他内容。
"""
        messages = [
            {"role": "system", "content": "你是一个 SQL 专家，只返回 SQL 语句。"},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat(messages)
        # 提取 SQL
        import re
        sql_match = re.search(r'(?:```sql\s*)?(SELECT[\s\S]*?)(?:```|$)', response, re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        return response.strip()

    async def _refine_sql(self, action_input: dict, state: ReActState) -> str:
        """修正 SQL"""
        prompt = f"""修正以下 SQL:

原 SQL: {action_input.get('original_sql', '')}
错误: {action_input.get('error', '')}
修正提示: {action_input.get('hint', '')}

Schema:
{state.schema[:2000]}

只返回修正后的 SQL 语句。
"""
        messages = [
            {"role": "system", "content": "你是一个 SQL 专家，修正 SQL 错误。"},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat(messages)
        import re
        sql_match = re.search(r'(?:```sql\s*)?(SELECT[\s\S]*?)(?:```|$)', response, re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        return response.strip()

    async def _decompose_query(self, action_input: dict, state: ReActState) -> list[dict]:
        """分解查询"""
        prompt = f"""将以下复杂查询分解成子查询:

查询: {action_input.get('query', state.query)}
分解原因: {action_input.get('reason', '')}

返回 JSON 格式:
```json
[
    {{"id": "task_1", "query": "子查询1", "dependencies": []}},
    {{"id": "task_2", "query": "子查询2", "dependencies": ["task_1"]}}
]
```
"""
        messages = [
            {"role": "system", "content": "你是一个查询分解专家。"},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat(messages)
        import re
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return [{"id": "main", "query": state.query, "dependencies": []}]


async def react_agent_node(state: dict) -> dict:
    """
    ReAct Agent 节点

    用于处理复杂查询，替代简单的串行流程
    """
    from ..agents import DataSniper, LogicArchitect, Judge
    from ..db.connector import get_db_manager

    query = state["query"]
    database = state.get("database", "default")
    schema = state.get("schema_context", "")

    # 如果没有 schema，先获取
    if not schema:
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        schema = await connector.get_schema()

    agent = ReActAgent(database)

    # 注册工具
    async def execute_sql_tool(action_input: dict, react_state: ReActState) -> dict:
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        sql = action_input.get("sql", react_state.intermediate_results.get("generated_sql", ""))
        limit = action_input.get("limit", 100)
        try:
            data = await connector.execute_query(sql, limit=limit)
            return {"success": True, "data": data[:10], "row_count": len(data)}  # 只返回前10行作为观察
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def validate_sql_tool(action_input: dict, react_state: ReActState) -> dict:
        judge = Judge(database)
        sql = action_input.get("sql", react_state.intermediate_results.get("generated_sql", ""))
        result = await judge.judge(sql)
        return {
            "valid": result.get("approved", False),
            "reason": result.get("reason", ""),
        }

    agent.register_tool(ActionType.EXECUTE_SQL, execute_sql_tool)
    agent.register_tool(ActionType.VALIDATE_SQL, validate_sql_tool)

    # 运行 ReAct 循环
    react_state = await agent.run(query, schema)

    # 转换结果
    if react_state.is_complete and react_state.final_sql:
        # 执行最终 SQL
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        try:
            data = await connector.execute_query(react_state.final_sql, limit=1000)
            return {
                "winner_sql": react_state.final_sql,
                "execution_result": {
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "columns": list(data[0].keys()) if data else [],
                    "execution_time_ms": 0,
                    "error": None,
                },
                "react_trace": {
                    "steps": [
                        {
                            "step": s.step_num,
                            "thought": s.thought,
                            "action": s.action.value,
                            "observation": s.observation[:200] if s.observation else None,
                        }
                        for s in react_state.steps
                    ],
                },
                "current_agent": "react_agent",
                "next_agent": "viz_expert",
            }
        except Exception as e:
            return {
                "current_agent": "react_agent",
                "next_agent": "__end__",
                "last_error": str(e),
            }

    return {
        "current_agent": "react_agent",
        "next_agent": "__end__",
        "last_error": react_state.error or "ReAct agent failed to complete",
    }


__all__ = [
    "ReActAgent",
    "ReActState",
    "ThoughtStep",
    "ActionType",
    "react_agent_node",
]
