# -*- coding: utf-8 -*-
"""
SQL 方言转换器
支持不同数据库之间的 SQL 转换
"""

from typing import Optional
import sqlglot
from sqlglot import exp, transpile
from sqlglot.errors import ParseError


class SQLDialectConverter:
    """
    SQL 方言转换器

    支持的方言:
    - MySQL
    - PostgreSQL
    - SQLite
    - SQL Server (TSQL)
    """

    DIALECT_MAP = {
        "mysql": "mysql",
        "postgresql": "postgres",
        "postgres": "postgres",
        "sqlite": "sqlite",
        "sqlserver": "tsql",
        "mssql": "tsql",
        "tsql": "tsql",
        "clickhouse": "clickhouse",
        "duckdb": "duckdb",
    }

    def __init__(self, source_dialect: str = "sqlite"):
        """
        Args:
            source_dialect: 源方言
        """
        self.source_dialect = self._normalize_dialect(source_dialect)

    def _normalize_dialect(self, dialect: str) -> str:
        """标准化方言名称"""
        return self.DIALECT_MAP.get(dialect.lower(), "sqlite")

    def convert(
        self,
        sql: str,
        target_dialect: str,
        pretty: bool = False,
    ) -> str:
        """
        转换 SQL 到目标方言

        Args:
            sql: 源 SQL
            target_dialect: 目标方言
            pretty: 是否美化输出

        Returns:
            转换后的 SQL
        """
        target = self._normalize_dialect(target_dialect)

        try:
            result = transpile(
                sql,
                read=self.source_dialect,
                write=target,
                pretty=pretty,
            )
            return result[0] if result else sql
        except ParseError:
            return sql

    def to_mysql(self, sql: str, pretty: bool = False) -> str:
        """转换为 MySQL"""
        return self.convert(sql, "mysql", pretty)

    def to_postgres(self, sql: str, pretty: bool = False) -> str:
        """转换为 PostgreSQL"""
        return self.convert(sql, "postgres", pretty)

    def to_sqlite(self, sql: str, pretty: bool = False) -> str:
        """转换为 SQLite"""
        return self.convert(sql, "sqlite", pretty)

    def to_sqlserver(self, sql: str, pretty: bool = False) -> str:
        """转换为 SQL Server"""
        return self.convert(sql, "tsql", pretty)


class DialectDifferences:
    """
    方言差异处理

    处理不同数据库之间的语法差异
    """

    @staticmethod
    def get_limit_syntax(dialect: str, limit: int, offset: int = 0) -> str:
        """
        获取 LIMIT 语法

        Args:
            dialect: 方言
            limit: 限制数量
            offset: 偏移量

        Returns:
            LIMIT 子句
        """
        dialect = dialect.lower()

        if dialect in ("mysql", "postgresql", "postgres", "sqlite", "clickhouse", "duckdb"):
            if offset > 0:
                return f"LIMIT {limit} OFFSET {offset}"
            return f"LIMIT {limit}"

        elif dialect in ("sqlserver", "mssql", "tsql"):
            if offset > 0:
                return f"OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
            return f"TOP {limit}"

        return f"LIMIT {limit}"

    @staticmethod
    def get_string_concat(dialect: str, *args: str) -> str:
        """
        获取字符串连接语法

        Args:
            dialect: 方言
            args: 要连接的字符串/列

        Returns:
            连接表达式
        """
        dialect = dialect.lower()

        if dialect == "mysql":
            # MySQL 使用 CONCAT 函数
            return f"CONCAT({', '.join(args)})"

        elif dialect in ("sqlite", "duckdb", "postgresql", "postgres"):
            # SQLite, DuckDB, PostgreSQL 使用 || 操作符
            return " || ".join(args)

        elif dialect in ("sqlserver", "mssql", "tsql"):
            return " + ".join(args)

        elif dialect == "clickhouse":
            return f"concat({', '.join(args)})"

        return " || ".join(args)

    @staticmethod
    def get_current_timestamp(dialect: str) -> str:
        """获取当前时间戳函数"""
        dialect = dialect.lower()

        if dialect == "mysql":
            return "NOW()"
        elif dialect in ("postgresql", "postgres"):
            return "CURRENT_TIMESTAMP"
        elif dialect == "sqlite":
            return "datetime('now')"
        elif dialect in ("sqlserver", "mssql", "tsql"):
            return "GETDATE()"
        elif dialect == "clickhouse":
            return "now()"
        elif dialect == "duckdb":
            return "current_timestamp"

        return "CURRENT_TIMESTAMP"

    @staticmethod
    def get_date_diff(dialect: str, date1: str, date2: str, unit: str = "day") -> str:
        """
        获取日期差异函数

        Args:
            dialect: 方言
            date1: 日期1
            date2: 日期2
            unit: 单位 (day/month/year)

        Returns:
            日期差异表达式
        """
        dialect = dialect.lower()

        if dialect == "mysql":
            return f"DATEDIFF({date1}, {date2})"

        elif dialect in ("postgresql", "postgres"):
            if unit == "day":
                return f"({date1}::date - {date2}::date)"
            return f"EXTRACT({unit.upper()} FROM AGE({date1}, {date2}))"

        elif dialect == "sqlite":
            return f"JULIANDAY({date1}) - JULIANDAY({date2})"

        elif dialect in ("sqlserver", "mssql", "tsql"):
            return f"DATEDIFF({unit}, {date2}, {date1})"

        elif dialect == "clickhouse":
            return f"dateDiff('{unit}', {date2}, {date1})"

        elif dialect == "duckdb":
            return f"date_diff('{unit}', {date2}, {date1})"

        return f"DATEDIFF({date1}, {date2})"

    @staticmethod
    def get_isnull(dialect: str, expr: str, default: str) -> str:
        """
        获取 NULL 替换函数

        Args:
            dialect: 方言
            expr: 表达式
            default: 默认值

        Returns:
            NULL 替换表达式
        """
        dialect = dialect.lower()

        if dialect == "mysql":
            return f"IFNULL({expr}, {default})"

        elif dialect in ("postgresql", "postgres"):
            return f"COALESCE({expr}, {default})"

        elif dialect == "sqlite":
            return f"IFNULL({expr}, {default})"

        elif dialect in ("sqlserver", "mssql", "tsql"):
            return f"ISNULL({expr}, {default})"

        elif dialect == "clickhouse":
            return f"ifNull({expr}, {default})"

        elif dialect == "duckdb":
            return f"COALESCE({expr}, {default})"

        return f"COALESCE({expr}, {default})"


# 便捷函数
def convert_sql(
    sql: str,
    source_dialect: str,
    target_dialect: str,
    pretty: bool = False,
) -> str:
    """转换 SQL 方言"""
    converter = SQLDialectConverter(source_dialect)
    return converter.convert(sql, target_dialect, pretty)


def get_dialect_converter(source_dialect: str = "sqlite") -> SQLDialectConverter:
    """获取方言转换器"""
    return SQLDialectConverter(source_dialect)


__all__ = [
    "SQLDialectConverter",
    "DialectDifferences",
    "convert_sql",
    "get_dialect_converter",
]
