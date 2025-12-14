# -*- coding: utf-8 -*-
"""List tables MCP tool"""

from typing import Any, Optional

from ..base import MCPTool, ToolCategory, ToolParameter, ToolSchema, get_tool_registry
from ...db.connector import get_db_manager


class ListTablesTool(MCPTool):
    """
    列出数据库表工具

    MCP 标准实现
    """

    @property
    def name(self) -> str:
        return "list_tables"

    @property
    def description(self) -> str:
        return "列出数据库中的所有表，返回表名、注释等信息"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SCHEMA

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(parameters=[
            ToolParameter(
                name="database",
                type="string",
                description="数据库名称，为空时使用默认数据库",
                required=False,
            ),
        ])

    async def _execute(self, database: Optional[str] = None) -> dict:
        """执行列表查询"""
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        tables = await connector.get_tables()
        return {
            "database": database or "default",
            "tables": tables,
            "count": len(tables),
        }


# 创建工具实例并注册
_tool_instance = ListTablesTool()
get_tool_registry().register(_tool_instance)


# 兼容旧接口的函数
async def list_tables(database: Optional[str] = None) -> dict:
    """
    列出数据库中的所有表 (兼容旧接口)

    Args:
        database: 数据库名称，为空时使用默认数据库

    Returns:
        包含表列表的字典
    """
    result = await _tool_instance.execute(database=database)
    return result.data if result.success else {"error": result.error}
