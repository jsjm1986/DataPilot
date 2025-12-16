"""
DataPilot CLI 查询执行器
封装 LangGraph 工作流调用逻辑
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime

from ..core.graph import get_compiled_graph
from ..core.state import create_initial_state, DataPilotState
from ..db.connector import get_db_manager
from ..agents.ambi_resolver import AmbiResolver


@dataclass
class QueryResult:
    """查询结果"""
    success: bool
    sql: Optional[str] = None
    data: Optional[list[dict]] = None
    row_count: int = 0
    chart_config: Optional[dict] = None
    insight: Optional[str] = None
    python_code: Optional[str] = None
    sandbox_result: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: float = 0
    clarify_needed: bool = False
    clarify_options: Optional[dict] = None
    trace_id: Optional[str] = None


class QueryRunner:
    """CLI 查询执行器"""

    def __init__(
        self,
        database: str = "default",
        user_id: str = "cli_user",
        tenant_id: str = "default",
        viz_mode: str = "echarts",
        include_cache: bool = True,
        include_ambi_resolver: bool = True,
    ):
        self.database = database
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.viz_mode = viz_mode
        self.include_cache = include_cache
        self.include_ambi_resolver = include_ambi_resolver

        # 编译工作流
        self._graph = None

        # 待处理的澄清会话
        self._pending_clarify: Optional[dict] = None

    def _get_graph(self):
        """获取编译后的工作流（延迟初始化）"""
        if self._graph is None:
            self._graph = get_compiled_graph(
                include_ambi_resolver=self.include_ambi_resolver,
                include_semantic_cache=self.include_cache,
            )
        return self._graph

    async def run_query(self, query: str) -> QueryResult:
        """
        执行查询

        Args:
            query: 自然语言查询

        Returns:
            QueryResult 对象
        """
        start_time = datetime.utcnow()

        try:
            # 创建初始状态
            initial_state = create_initial_state(
                query=query,
                user_id=self.user_id,
                tenant_id=self.tenant_id,
                database=self.database,
                viz_mode=self.viz_mode,
            )

            trace_id = initial_state["trace_id"]

            # 配置
            config = {
                "configurable": {
                    "thread_id": trace_id,
                }
            }

            # 执行工作流
            graph = self._get_graph()
            final_state = await graph.ainvoke(initial_state, config=config)

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # 检查是否需要澄清
            if final_state.get("clarify_needed") and final_state.get("clarify_options"):
                self._pending_clarify = {
                    "original_query": query,
                    "clarify_options": final_state.get("clarify_options"),
                    "trace_id": trace_id,
                }
                return QueryResult(
                    success=True,
                    clarify_needed=True,
                    clarify_options=final_state.get("clarify_options"),
                    duration_ms=duration_ms,
                    trace_id=trace_id,
                )

            # 提取结果
            execution_result = final_state.get("execution_result", {})

            return QueryResult(
                success=execution_result.get("success", False) if execution_result else False,
                sql=final_state.get("winner_sql"),
                data=execution_result.get("data") if execution_result else None,
                row_count=execution_result.get("row_count", 0) if execution_result else 0,
                chart_config=final_state.get("chart_config"),
                insight=final_state.get("insight"),
                python_code=final_state.get("python_code"),
                sandbox_result=final_state.get("sandbox_result"),
                error=execution_result.get("error") if execution_result else None,
                duration_ms=duration_ms,
                trace_id=trace_id,
            )

        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return QueryResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def continue_with_clarify(self, choice_index: int) -> QueryResult:
        """
        用户选择澄清选项后继续执行

        Args:
            choice_index: 用户选择的选项索引

        Returns:
            QueryResult 对象
        """
        if not self._pending_clarify:
            return QueryResult(
                success=False,
                error="没有待处理的澄清请求",
            )

        try:
            original_query = self._pending_clarify["original_query"]
            clarify_options = self._pending_clarify["clarify_options"]
            options = clarify_options.get("options", [])

            if choice_index < 0 or choice_index >= len(options):
                return QueryResult(
                    success=False,
                    error=f"无效的选项索引: {choice_index}",
                )

            selected_option = options[choice_index]

            # 使用 AmbiResolver 解析用户选择，生成包含时间范围的新查询
            resolver = AmbiResolver()
            resolved_query = await resolver.resolve_with_selection(original_query, selected_option)

            # 清除待处理的澄清
            self._pending_clarify = None

            # 用解析后的查询重新执行
            return await self.run_query(resolved_query)

        except Exception as e:
            return QueryResult(
                success=False,
                error=str(e),
            )

    def list_databases(self) -> list[str]:
        """获取可用数据库列表"""
        try:
            db_manager = get_db_manager()
            return list(db_manager._connectors.keys())
        except Exception:
            return []

    def set_database(self, database: str):
        """切换数据库"""
        self.database = database

    def run_query_sync(self, query: str) -> QueryResult:
        """同步版本的查询执行"""
        return asyncio.run(self.run_query(query))

    def continue_with_clarify_sync(self, choice_index: int) -> QueryResult:
        """同步版本的澄清继续执行"""
        return asyncio.run(self.continue_with_clarify(choice_index))
