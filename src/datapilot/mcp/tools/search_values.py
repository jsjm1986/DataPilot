# -*- coding: utf-8 -*-
"""Search values MCP tool"""

from typing import Any, Optional

from ..base import MCPTool, ToolCategory, ToolParameter, ToolSchema, get_tool_registry
from ...db.connector import get_db_manager
from ...config.settings import get_settings


class SearchValuesTool(MCPTool):
    """
    数据库值搜索工具

    MCP 标准实现
    """

    @property
    def name(self) -> str:
        return "search_values"

    @property
    def description(self) -> str:
        return "在数据库中搜索特定值，支持模糊匹配，用于实体映射"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATABASE

    @property
    def schema(self) -> ToolSchema:
        settings = get_settings()
        return ToolSchema(parameters=[
            ToolParameter(
                name="search_term",
                type="string",
                description="搜索词",
                required=True,
            ),
            ToolParameter(
                name="table_name",
                type="string",
                description="限定搜索的表名",
                required=False,
            ),
            ToolParameter(
                name="column_name",
                type="string",
                description="限定搜索的列名",
                required=False,
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
                description="返回结果数量限制",
                required=False,
                default=settings.mcp_page_limit,
                minimum=1,
                maximum=settings.mcp_page_limit,
            ),
        ])

    def _validate_identifier(self, name: str) -> str:
        """验证并清理标识符，防止 SQL 注入"""
        import re
        # 只允许字母、数字、下划线
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"Invalid identifier: {name}")
        return name

    def _quote_identifier(self, name: str, db_type: str) -> str:
        """根据数据库类型引用标识符"""
        safe_name = self._validate_identifier(name)
        if db_type in ("mysql", "clickhouse"):
            return f"`{safe_name}`"
        elif db_type in ("postgresql", "duckdb"):
            return f'"{safe_name}"'
        elif db_type == "sqlserver":
            return f"[{safe_name}]"
        else:  # sqlite
            return f'"{safe_name}"'

    def _get_like_cast(self, column: str, db_type: str) -> str:
        """根据数据库类型获取 CAST 语法"""
        if db_type == "clickhouse":
            return f"toString({column})"
        elif db_type == "sqlserver":
            return f"CAST({column} AS NVARCHAR(MAX))"
        else:
            return f"CAST({column} AS TEXT)"

    async def _execute(
        self,
        search_term: str,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
        database: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """执行值搜索"""
        # 使用配置的默认限制
        settings = get_settings()
        if limit is None:
            limit = settings.mcp_page_limit

        db_manager = get_db_manager()
        connector = db_manager.get(database)
        db_type = connector.db_type

        results = []

        if table_name and column_name:
            # 验证并引用标识符
            safe_table = self._quote_identifier(table_name, db_type)
            safe_column = self._quote_identifier(column_name, db_type)
            cast_expr = self._get_like_cast(safe_column, db_type)

            # 搜索特定表的特定列
            sql = f"""
                SELECT DISTINCT {safe_column} as value, COUNT(*) as count
                FROM {safe_table}
                WHERE {cast_expr} LIKE :pattern
                GROUP BY {safe_column}
                ORDER BY count DESC
                LIMIT :limit
            """
            data = await connector.execute_query(
                sql,
                {"pattern": f"%{search_term}%", "limit": limit}
            )
            results = [
                {
                    "table": table_name,
                    "column": column_name,
                    "value": str(row["value"]),
                    "count": row["count"],
                }
                for row in data
            ]
        else:
            # 搜索所有文本列
            tables = await connector.get_tables()
            for table in tables:
                if table_name and table["name"] != table_name:
                    continue

                for col in table.get("columns", []):
                    col_type = str(col["type"]).upper()
                    if any(t in col_type for t in ["TEXT", "VARCHAR", "CHAR", "STRING"]):
                        try:
                            # 验证并引用标识符
                            safe_table = self._quote_identifier(table["name"], db_type)
                            safe_column = self._quote_identifier(col["name"], db_type)
                            cast_expr = self._get_like_cast(safe_column, db_type)

                            sql = f"""
                                SELECT DISTINCT {safe_column} as value, COUNT(*) as count
                                FROM {safe_table}
                                WHERE {cast_expr} LIKE :pattern
                                GROUP BY {safe_column}
                                ORDER BY count DESC
                                LIMIT 5
                            """
                            data = await connector.execute_query(
                                sql,
                                {"pattern": f"%{search_term}%"}
                            )
                            for row in data:
                                results.append({
                                    "table": table["name"],
                                    "column": col["name"],
                                    "value": str(row["value"]),
                                    "count": row["count"],
                                })
                        except Exception:
                            continue

                if len(results) >= limit:
                    break

        return {
            "database": database or "default",
            "search_term": search_term,
            "result_count": len(results),
            "results": results[:limit],
        }


# 创建工具实例并注册
_tool_instance = SearchValuesTool()
get_tool_registry().register(_tool_instance)


# 兼容旧接口的函数
async def search_values(
    search_term: str,
    table_name: Optional[str] = None,
    column_name: Optional[str] = None,
    database: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    在数据库中搜索特定值 (兼容旧接口)

    Args:
        search_term: 搜索词
        table_name: 限定表名
        column_name: 限定列名
        database: 数据库名称
        limit: 返回结果数量限制

    Returns:
        包含搜索结果的字典
    """
    result = await _tool_instance.execute(
        search_term=search_term,
        table_name=table_name,
        column_name=column_name,
        database=database,
        limit=limit,
    )
    if result.success:
        return result.data
    else:
        return {"error": result.error}
