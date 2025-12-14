# -*- coding: utf-8 -*-
"""Read-only query MCP tool"""

from typing import Any, Optional

from ..base import MCPTool, ToolCategory, ToolParameter, ToolSchema, get_tool_registry
from ...db.connector import get_db_manager
from ...config.settings import get_settings


class ReadOnlyQueryTool(MCPTool):
    """
    只读 SQL 查询工具

    MCP 标准实现
    """

    @property
    def name(self) -> str:
        return "read_only_query"

    @property
    def description(self) -> str:
        return "执行只读 SQL 查询，仅允许 SELECT 语句，自动检测危险操作"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.QUERY

    @property
    def schema(self) -> ToolSchema:
        settings = get_settings()
        return ToolSchema(parameters=[
            ToolParameter(
                name="sql",
                type="string",
                description="SQL 查询语句 (仅支持 SELECT)",
                required=True,
            ),
            ToolParameter(
                name="database",
                type="string",
                description="数据库名称",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="结果行数限制",
                required=False,
                default=settings.mcp_row_limit,
                minimum=1,
                maximum=settings.mcp_row_limit,
            ),
        ])

    async def _execute(
        self,
        sql: str,
        database: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """执行只读查询"""
        # 使用配置的默认限制
        settings = get_settings()
        if limit is None:
            limit = settings.mcp_row_limit

        # 安全检查：只允许 SELECT
        sql_lower = sql.lower().strip()
        if not sql_lower.startswith("select"):
            raise ValueError("Only SELECT queries are allowed")

        # 检查危险关键字
        dangerous_keywords = ["insert", "update", "delete", "drop", "truncate", "alter", "create"]
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                raise ValueError(f"Dangerous keyword detected: {keyword}")

        db_manager = get_db_manager()
        connector = db_manager.get(database)

        data = await connector.execute_query(sql, limit=limit)
        columns = list(data[0].keys()) if data else []

        return {
            "database": database or "default",
            "sql": sql,
            "success": True,
            "row_count": len(data),
            "columns": columns,
            "data": data,
        }


# 创建工具实例并注册
_tool_instance = ReadOnlyQueryTool()
get_tool_registry().register(_tool_instance)


# 兼容旧接口的函数
async def read_only_query(
    sql: str,
    database: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    执行只读 SQL 查询 (兼容旧接口)

    Args:
        sql: SQL 查询语句
        database: 数据库名称
        limit: 结果行数限制

    Returns:
        查询结果
    """
    result = await _tool_instance.execute(sql=sql, database=database, limit=limit)
    if result.success:
        return result.data
    else:
        return {
            "database": database or "default",
            "sql": sql,
            "success": False,
            "error": result.error,
        }
