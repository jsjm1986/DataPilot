# -*- coding: utf-8 -*-
"""
SQL 解析器模块
基于 SQLGlot 的 SQL 解析和分析
"""

from typing import Optional
import sqlglot
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError


class SQLParser:
    """
    SQL 解析器

    功能:
    1. SQL 语法解析
    2. 提取表名、列名
    3. 分析查询结构
    4. SQL 格式化
    """

    def __init__(self, dialect: str = "sqlite"):
        """
        Args:
            dialect: SQL 方言 (mysql/postgresql/sqlite)
        """
        self.dialect = self._normalize_dialect(dialect)

    def _normalize_dialect(self, dialect: str) -> str:
        """标准化方言名称"""
        dialect_map = {
            "mysql": "mysql",
            "postgresql": "postgres",
            "postgres": "postgres",
            "sqlite": "sqlite",
            "sqlserver": "tsql",
            "mssql": "tsql",
        }
        return dialect_map.get(dialect.lower(), "sqlite")

    def parse(self, sql: str) -> Optional[exp.Expression]:
        """
        解析 SQL 语句

        Args:
            sql: SQL 语句

        Returns:
            解析后的表达式树，解析失败返回 None
        """
        try:
            return parse_one(sql, dialect=self.dialect)
        except ParseError:
            return None

    def is_valid(self, sql: str) -> bool:
        """检查 SQL 是否有效"""
        return self.parse(sql) is not None

    def extract_tables(self, sql: str) -> list[str]:
        """
        提取 SQL 中的表名

        Args:
            sql: SQL 语句

        Returns:
            表名列表
        """
        tree = self.parse(sql)
        if not tree:
            return []

        tables = set()
        for table in tree.find_all(exp.Table):
            table_name = table.name
            if table_name:
                tables.add(table_name)

        return list(tables)

    def extract_columns(self, sql: str) -> list[dict]:
        """
        提取 SQL 中的列引用

        Args:
            sql: SQL 语句

        Returns:
            列信息列表 [{"table": "t", "column": "c"}, ...]
        """
        tree = self.parse(sql)
        if not tree:
            return []

        columns = []
        for col in tree.find_all(exp.Column):
            columns.append({
                "table": col.table or None,
                "column": col.name,
            })

        return columns

    def extract_select_columns(self, sql: str) -> list[str]:
        """
        提取 SELECT 子句中的列名

        Args:
            sql: SQL 语句

        Returns:
            列名列表
        """
        tree = self.parse(sql)
        if not tree:
            return []

        columns = []
        select = tree.find(exp.Select)
        if select:
            for expr in select.expressions:
                if isinstance(expr, exp.Alias):
                    columns.append(expr.alias)
                elif isinstance(expr, exp.Column):
                    columns.append(expr.name)
                elif isinstance(expr, exp.Star):
                    columns.append("*")
                else:
                    # 其他表达式使用 SQL 表示
                    columns.append(expr.sql(dialect=self.dialect))

        return columns

    def get_query_type(self, sql: str) -> str:
        """
        获取查询类型

        Returns:
            查询类型: SELECT/INSERT/UPDATE/DELETE/CREATE/DROP/OTHER
        """
        tree = self.parse(sql)
        if not tree:
            return "OTHER"

        type_map = {
            exp.Select: "SELECT",
            exp.Insert: "INSERT",
            exp.Update: "UPDATE",
            exp.Delete: "DELETE",
            exp.Create: "CREATE",
            exp.Drop: "DROP",
            exp.Alter: "ALTER",
        }

        for expr_type, name in type_map.items():
            if isinstance(tree, expr_type):
                return name

        return "OTHER"

    def has_aggregation(self, sql: str) -> bool:
        """检查是否包含聚合函数"""
        tree = self.parse(sql)
        if not tree:
            return False

        agg_funcs = {"sum", "count", "avg", "min", "max", "group_concat"}
        for func in tree.find_all(exp.Func):
            if func.name.lower() in agg_funcs:
                return True

        return False

    def has_subquery(self, sql: str) -> bool:
        """检查是否包含子查询"""
        tree = self.parse(sql)
        if not tree:
            return False

        # 查找嵌套的 SELECT
        subqueries = list(tree.find_all(exp.Subquery))
        return len(subqueries) > 0

    def has_join(self, sql: str) -> bool:
        """检查是否包含 JOIN"""
        tree = self.parse(sql)
        if not tree:
            return False

        joins = list(tree.find_all(exp.Join))
        return len(joins) > 0

    def get_where_conditions(self, sql: str) -> list[str]:
        """
        提取 WHERE 条件

        Returns:
            条件表达式列表
        """
        tree = self.parse(sql)
        if not tree:
            return []

        conditions = []
        where = tree.find(exp.Where)
        if where:
            # 简单处理：返回整个 WHERE 表达式
            conditions.append(where.this.sql(dialect=self.dialect))

        return conditions

    def format_sql(self, sql: str, pretty: bool = True) -> str:
        """
        格式化 SQL

        Args:
            sql: SQL 语句
            pretty: 是否美化输出

        Returns:
            格式化后的 SQL
        """
        tree = self.parse(sql)
        if not tree:
            return sql

        return tree.sql(dialect=self.dialect, pretty=pretty)

    def analyze(self, sql: str) -> dict:
        """
        分析 SQL 结构

        Returns:
            分析结果字典
        """
        return {
            "valid": self.is_valid(sql),
            "type": self.get_query_type(sql),
            "tables": self.extract_tables(sql),
            "columns": self.extract_columns(sql),
            "select_columns": self.extract_select_columns(sql),
            "has_aggregation": self.has_aggregation(sql),
            "has_subquery": self.has_subquery(sql),
            "has_join": self.has_join(sql),
            "where_conditions": self.get_where_conditions(sql),
        }


# 便捷函数
def parse_sql(sql: str, dialect: str = "sqlite") -> Optional[exp.Expression]:
    """解析 SQL"""
    parser = SQLParser(dialect)
    return parser.parse(sql)


def analyze_sql(sql: str, dialect: str = "sqlite") -> dict:
    """分析 SQL"""
    parser = SQLParser(dialect)
    return parser.analyze(sql)


def format_sql(sql: str, dialect: str = "sqlite", pretty: bool = True) -> str:
    """格式化 SQL"""
    parser = SQLParser(dialect)
    return parser.format_sql(sql, pretty)


__all__ = [
    "SQLParser",
    "parse_sql",
    "analyze_sql",
    "format_sql",
]
