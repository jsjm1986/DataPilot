# -*- coding: utf-8 -*-
"""Get DDL MCP tool"""

from typing import Any, Optional

from ..base import MCPTool, ToolCategory, ToolParameter, ToolSchema, get_tool_registry
from ...db.connector import get_db_manager


class GetDDLTool(MCPTool):
    """
    获取表 DDL 定义工具

    MCP 标准实现
    """

    @property
    def name(self) -> str:
        return "get_ddl"

    @property
    def description(self) -> str:
        return "获取表的 DDL 定义，包括列信息、类型、约束等"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SCHEMA

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(parameters=[
            ToolParameter(
                name="table_name",
                type="string",
                description="表名，如果为空则返回所有表的 DDL",
                required=False,
            ),
            ToolParameter(
                name="database",
                type="string",
                description="数据库名称，为空时使用默认数据库",
                required=False,
            ),
        ])

    async def _execute(
        self,
        table_name: Optional[str] = None,
        database: Optional[str] = None,
    ) -> dict:
        """执行 DDL 获取"""
        db_manager = get_db_manager()
        connector = db_manager.get(database)

        if table_name:
            tables = await connector.get_tables()
            target_table = None
            for t in tables:
                if t["name"] == table_name:
                    target_table = t
                    break

            if not target_table:
                raise ValueError(f"Table not found: {table_name}")

            columns = target_table.get("columns", [])
            col_defs = []
            for col in columns:
                col_def = f"  {col['name']} {col['type']}"
                if not col.get("nullable", True):
                    col_def += " NOT NULL"
                col_defs.append(col_def)

            ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"

            return {
                "database": database or "default",
                "table_name": table_name,
                "ddl": ddl,
                "columns": columns,
            }
        else:
            schema = await connector.get_schema()
            return {"database": database or "default", "ddl": schema}


# 创建工具实例并注册
_tool_instance = GetDDLTool()
get_tool_registry().register(_tool_instance)


# 兼容旧接口的函数
async def get_ddl(
    table_name: Optional[str] = None,
    database: Optional[str] = None,
) -> dict:
    """
    获取表的 DDL 定义 (兼容旧接口)

    Args:
        table_name: 表名，如果为空则返回所有表的 DDL
        database: 数据库名称

    Returns:
        包含 DDL 的字典
    """
    result = await _tool_instance.execute(table_name=table_name, database=database)
    if result.success:
        return result.data
    else:
        return {"error": result.error}
