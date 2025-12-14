# -*- coding: utf-8 -*-
"""
Schema Introspector - 数据库元信息提取器

核心理念: 不做硬编码，一切从数据库元信息中动态获取
LLM 根据真实的数据库结构生成 SQL，而非依赖预设规则

功能:
1. 提取完整的表结构信息
2. 识别时间类型字段
3. 提取枚举值/样本值
4. 分析外键关系
5. 生成 LLM 可用的上下文
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class ColumnInfo:
    """列信息"""
    name: str
    data_type: str
    nullable: bool
    comment: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: Optional[str] = None  # "table.column"
    sample_values: list[Any] = field(default_factory=list)  # 样本值
    distinct_count: Optional[int] = None  # 不同值数量
    is_time_field: bool = False  # 是否是时间字段
    is_enum_like: bool = False  # 是否像枚举（有限的不同值）


@dataclass
class TableInfo:
    """表信息"""
    name: str
    comment: Optional[str] = None
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    primary_key: Optional[str] = None
    indexes: list[str] = field(default_factory=list)

    @property
    def time_columns(self) -> list[ColumnInfo]:
        """获取时间类型列"""
        return [c for c in self.columns if c.is_time_field]

    @property
    def enum_columns(self) -> list[ColumnInfo]:
        """获取枚举类型列"""
        return [c for c in self.columns if c.is_enum_like]


@dataclass
class SchemaMetadata:
    """完整的 Schema 元信息"""
    database: str
    dialect: str  # mysql, postgresql, sqlite
    tables: list[TableInfo] = field(default_factory=list)
    foreign_keys: list[dict] = field(default_factory=list)  # [{from_table, from_col, to_table, to_col}]
    extracted_at: str = ""

    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.utcnow().isoformat()


class SchemaIntrospector:
    """
    Schema 元信息提取器

    从数据库动态提取所有需要的元信息，
    供 LLM 理解数据库结构并生成正确的 SQL
    """

    # 时间类型识别
    TIME_TYPES = {
        'mysql': ['date', 'datetime', 'timestamp', 'time', 'year'],
        'postgresql': ['date', 'timestamp', 'timestamptz', 'time', 'timetz', 'interval'],
        'sqlite': ['date', 'datetime', 'timestamp'],
        'sqlserver': ['date', 'datetime', 'datetime2', 'smalldatetime', 'time', 'datetimeoffset'],
        'clickhouse': ['date', 'date32', 'datetime', 'datetime64'],
        'duckdb': ['date', 'time', 'timestamp', 'timestamptz', 'interval'],
    }

    # 时间字段名模式（作为辅助判断）
    TIME_COLUMN_PATTERNS = [
        'date', 'time', 'created', 'updated', 'modified', 'deleted',
        '_at', '_on', '_time', '_date',
    ]

    # 可能是枚举的列名模式（只对这些列采样）
    ENUM_COLUMN_PATTERNS = [
        'status', 'state', 'type', 'level', 'category', 'method',
        'gender', 'role', 'priority', 'source', 'channel', 'mode',
    ]

    # 全局缓存
    _cache: dict = {}
    _cache_ttl: int = 300  # 5分钟缓存

    def __init__(self, connector):
        """
        初始化

        Args:
            connector: 数据库连接器
        """
        self.connector = connector
        # 使用 db_type 属性获取数据库类型
        self.dialect = getattr(connector, 'db_type', 'sqlite')

    def _validate_identifier(self, name: str) -> str:
        """验证并清理标识符，防止 SQL 注入"""
        import re
        # 只允许字母、数字、下划线
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"Invalid identifier: {name}")
        return name

    def _quote_identifier(self, name: str) -> str:
        """根据数据库类型引用标识符"""
        safe_name = self._validate_identifier(name)
        if self.dialect in ("mysql", "clickhouse"):
            return f"`{safe_name}`"
        elif self.dialect in ("postgresql", "duckdb"):
            return f'"{safe_name}"'
        elif self.dialect == "sqlserver":
            return f"[{safe_name}]"
        else:  # sqlite
            return f'"{safe_name}"'

    def _get_cache_key(self) -> str:
        """生成缓存键"""
        db_name = getattr(self.connector, 'database', 'default')
        return f"{self.dialect}:{db_name}"

    async def introspect(
        self,
        include_samples: bool = True,
        sample_limit: int = 10,
        include_row_counts: bool = True,
    ) -> SchemaMetadata:
        """
        提取完整的 Schema 元信息

        Args:
            include_samples: 是否包含样本值
            sample_limit: 样本值数量限制
            include_row_counts: 是否包含行数统计

        Returns:
            完整的元信息
        """
        import time

        # 检查缓存
        cache_key = self._get_cache_key()
        if cache_key in SchemaIntrospector._cache:
            cached_data, cached_time = SchemaIntrospector._cache[cache_key]
            if time.time() - cached_time < SchemaIntrospector._cache_ttl:
                print(f"[SchemaIntrospector] Using cached metadata for {cache_key}")
                return cached_data

        print(f"[SchemaIntrospector] Introspecting schema for {cache_key}...")
        tables = await self._get_tables()
        table_infos = []

        for table in tables:
            table_info = await self._introspect_table(
                table['name'],
                include_samples=include_samples,
                sample_limit=sample_limit,
                include_row_counts=include_row_counts,
            )
            table_infos.append(table_info)

        # 提取外键关系
        foreign_keys = await self._get_foreign_keys()

        result = SchemaMetadata(
            database=getattr(self.connector, 'database', 'default'),
            dialect=self.dialect,
            tables=table_infos,
            foreign_keys=foreign_keys,
        )

        # 存入缓存
        SchemaIntrospector._cache[cache_key] = (result, time.time())
        print(f"[SchemaIntrospector] Cached metadata for {cache_key}")

        return result

    async def _get_tables(self) -> list[dict]:
        """获取所有表"""
        return await self.connector.get_tables()

    async def _introspect_table(
        self,
        table_name: str,
        include_samples: bool = True,
        sample_limit: int = 10,
        include_row_counts: bool = True,
    ) -> TableInfo:
        """提取单个表的信息"""
        # 基础列信息
        columns_raw = await self._get_columns(table_name)
        columns = []

        for col in columns_raw:
            column_info = ColumnInfo(
                name=col['name'],
                data_type=col.get('type', 'unknown'),
                nullable=col.get('nullable', True),
                comment=col.get('comment'),
                is_primary_key=col.get('is_primary_key', False),
            )

            # 判断是否是时间字段
            column_info.is_time_field = self._is_time_column(column_info)

            columns.append(column_info)

        # 获取样本值
        if include_samples:
            columns = await self._add_sample_values(table_name, columns, sample_limit)

        # 获取行数
        row_count = None
        if include_row_counts:
            row_count = await self._get_row_count(table_name)

        # 获取表注释
        table_comment = await self._get_table_comment(table_name)

        return TableInfo(
            name=table_name,
            comment=table_comment,
            columns=columns,
            row_count=row_count,
        )

    async def _get_columns(self, table_name: str) -> list[dict]:
        """获取列信息"""
        # 尝试从 connector 获取
        if hasattr(self.connector, 'get_columns'):
            return await self.connector.get_columns(table_name)

        # 验证表名，防止 SQL 注入
        safe_table = self._validate_identifier(table_name)
        quoted_table = self._quote_identifier(table_name)

        # 回退到查询 information_schema
        if self.dialect == 'mysql':
            sql = f"""
                SELECT
                    COLUMN_NAME as name,
                    DATA_TYPE as type,
                    IS_NULLABLE = 'YES' as nullable,
                    COLUMN_COMMENT as comment,
                    COLUMN_KEY = 'PRI' as is_primary_key
                FROM information_schema.COLUMNS
                WHERE TABLE_NAME = :table_name
                ORDER BY ORDINAL_POSITION
            """
            try:
                result = await self.connector.execute_query(sql, {"table_name": safe_table})
                return result
            except Exception:
                return []
        elif self.dialect == 'postgresql':
            sql = f"""
                SELECT
                    column_name as name,
                    data_type as type,
                    is_nullable = 'YES' as nullable
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """
            try:
                result = await self.connector.execute_query(sql, {"table_name": safe_table})
                return result
            except Exception:
                return []
        elif self.dialect == 'sqlserver':
            sql = f"""
                SELECT
                    c.name as name,
                    t.name as type,
                    c.is_nullable as nullable,
                    ep.value as comment
                FROM sys.columns c
                JOIN sys.types t ON c.user_type_id = t.user_type_id
                LEFT JOIN sys.extended_properties ep ON ep.major_id = c.object_id AND ep.minor_id = c.column_id
                WHERE c.object_id = OBJECT_ID(:table_name)
                ORDER BY c.column_id
            """
            try:
                result = await self.connector.execute_query(sql, {"table_name": safe_table})
                return result
            except Exception:
                return []
        elif self.dialect == 'clickhouse':
            sql = f"DESCRIBE TABLE {quoted_table}"
        elif self.dialect == 'duckdb':
            sql = f"DESCRIBE {quoted_table}"
        else:  # sqlite
            sql = f"PRAGMA table_info({quoted_table})"

        try:
            result = await self.connector.execute_query(sql)
            if self.dialect == 'sqlite':
                return [
                    {
                        'name': row.get('name'),
                        'type': row.get('type'),
                        'nullable': not row.get('notnull', False),
                        'is_primary_key': bool(row.get('pk')),
                    }
                    for row in result
                ]
            elif self.dialect == 'clickhouse':
                # ClickHouse DESCRIBE 返回格式: name, type, default_type, default_expression, comment, ...
                return [
                    {
                        'name': row.get('name'),
                        'type': row.get('type'),
                        'nullable': 'Nullable' in str(row.get('type', '')),
                        'comment': row.get('comment', ''),
                    }
                    for row in result
                ]
            elif self.dialect == 'duckdb':
                # DuckDB DESCRIBE 返回格式: column_name, column_type, null, key, default, extra
                return [
                    {
                        'name': row.get('column_name'),
                        'type': row.get('column_type'),
                        'nullable': row.get('null') == 'YES',
                    }
                    for row in result
                ]
            return result
        except Exception:
            return []

    def _is_time_column(self, column: ColumnInfo) -> bool:
        """判断是否是时间列"""
        # 1. 根据数据类型判断
        type_lower = column.data_type.lower()
        time_types = self.TIME_TYPES.get(self.dialect, [])
        if any(t in type_lower for t in time_types):
            return True

        # 2. 根据列名模式判断
        name_lower = column.name.lower()
        if any(p in name_lower for p in self.TIME_COLUMN_PATTERNS):
            return True

        return False

    def _is_likely_enum_column(self, column: ColumnInfo) -> bool:
        """判断列是否可能是枚举类型（基于列名模式）"""
        name_lower = column.name.lower()
        # 检查列名是否匹配枚举模式
        for pattern in self.ENUM_COLUMN_PATTERNS:
            if pattern in name_lower:
                return True
        # 排除明显不是枚举的类型
        type_lower = column.data_type.lower()
        if any(t in type_lower for t in ['text', 'blob', 'json', 'float', 'double', 'decimal']):
            return False
        return False

    async def _add_sample_values(
        self,
        table_name: str,
        columns: list[ColumnInfo],
        limit: int,
    ) -> list[ColumnInfo]:
        """添加样本值 - 优化版：只对可能是枚举的列采样"""
        # 验证表名
        quoted_table = self._quote_identifier(table_name)

        # 只对可能是枚举的列进行采样，大幅减少查询数量
        enum_candidates = [c for c in columns if self._is_likely_enum_column(c)]

        for column in enum_candidates:
            try:
                # 验证列名
                quoted_column = self._quote_identifier(column.name)

                # 直接获取样本值（限制数量），不先 COUNT
                # 这样只需要一次查询而不是两次
                sample_sql = f"SELECT DISTINCT {quoted_column} as val FROM {quoted_table} WHERE {quoted_column} IS NOT NULL LIMIT {limit + 1}"

                sample_result = await self.connector.execute_query(sample_sql)
                if sample_result:
                    values = [r.get('val') for r in sample_result if r.get('val') is not None]
                    # 如果返回的值数量 <= limit，说明是枚举类型
                    if len(values) <= limit:
                        column.is_enum_like = True
                        column.sample_values = values
                        column.distinct_count = len(values)

            except Exception:
                continue

        return columns

    async def _get_row_count(self, table_name: str) -> Optional[int]:
        """获取表行数"""
        try:
            # 验证并引用表名
            quoted_table = self._quote_identifier(table_name)
            sql = f"SELECT COUNT(*) as cnt FROM {quoted_table}"

            result = await self.connector.execute_query(sql)
            if result:
                return result[0].get('cnt', 0)
        except Exception:
            pass
        return None

    async def _get_table_comment(self, table_name: str) -> Optional[str]:
        """获取表注释"""
        try:
            # 验证表名
            safe_table = self._validate_identifier(table_name)

            if self.dialect == 'mysql':
                sql = """
                    SELECT TABLE_COMMENT
                    FROM information_schema.TABLES
                    WHERE TABLE_NAME = :table_name
                """
                result = await self.connector.execute_query(sql, {"table_name": safe_table})
                if result:
                    return result[0].get('TABLE_COMMENT')
            elif self.dialect == 'postgresql':
                # PostgreSQL 需要特殊处理，使用 regclass 转换
                sql = """
                    SELECT obj_description(c.oid) as comment
                    FROM pg_class c
                    WHERE c.relname = :table_name
                """
                result = await self.connector.execute_query(sql, {"table_name": safe_table})
                if result:
                    return result[0].get('comment')
        except Exception:
            pass
        return None

    async def _get_foreign_keys(self) -> list[dict]:
        """获取外键关系"""
        try:
            if self.dialect == 'mysql':
                sql = """
                    SELECT
                        TABLE_NAME as from_table,
                        COLUMN_NAME as from_column,
                        REFERENCED_TABLE_NAME as to_table,
                        REFERENCED_COLUMN_NAME as to_column
                    FROM information_schema.KEY_COLUMN_USAGE
                    WHERE REFERENCED_TABLE_NAME IS NOT NULL
                """
            elif self.dialect == 'postgresql':
                sql = """
                    SELECT
                        tc.table_name as from_table,
                        kcu.column_name as from_column,
                        ccu.table_name as to_table,
                        ccu.column_name as to_column
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                """
            else:
                return []

            result = await self.connector.execute_query(sql)
            return result
        except Exception:
            return []

    def generate_llm_context(self, metadata: SchemaMetadata) -> str:
        """
        生成 LLM 可用的上下文

        这是关键：将元信息转换为 LLM 能理解的格式
        """
        parts = [
            f"# 数据库: {metadata.database}",
            f"# 类型: {metadata.dialect}",
            f"# 表数量: {len(metadata.tables)}",
            "",
        ]

        for table in metadata.tables:
            parts.append(f"## 表: {table.name}")
            if table.comment:
                parts.append(f"   说明: {table.comment}")
            if table.row_count:
                parts.append(f"   行数: {table.row_count}")
            parts.append("")

            # 列信息
            parts.append("   列:")
            for col in table.columns:
                col_desc = f"   - {col.name} ({col.data_type})"
                if col.comment:
                    col_desc += f" -- {col.comment}"
                if col.is_time_field:
                    col_desc += " [时间字段]"
                if col.is_primary_key:
                    col_desc += " [主键]"
                if col.is_foreign_key:
                    col_desc += f" [外键 -> {col.foreign_key_ref}]"
                parts.append(col_desc)

                # 样本值
                if col.sample_values:
                    if col.is_enum_like:
                        parts.append(f"     可选值: {col.sample_values}")
                    else:
                        parts.append(f"     样本: {col.sample_values[:5]}")

            parts.append("")

        # 外键关系
        if metadata.foreign_keys:
            parts.append("## 表关系 (外键)")
            for fk in metadata.foreign_keys:
                parts.append(f"   {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}")

        return "\n".join(parts)

    def generate_time_fields_context(self, metadata: SchemaMetadata) -> str:
        """生成时间字段专用上下文"""
        parts = ["# 数据库中的时间字段:"]

        for table in metadata.tables:
            time_cols = table.time_columns
            if time_cols:
                for col in time_cols:
                    parts.append(f"- {table.name}.{col.name} ({col.data_type})")
                    if col.comment:
                        parts.append(f"  说明: {col.comment}")

        if len(parts) == 1:
            parts.append("  (未发现时间类型字段)")

        return "\n".join(parts)

    def generate_enum_fields_context(self, metadata: SchemaMetadata) -> str:
        """生成枚举字段专用上下文"""
        parts = ["# 数据库中的枚举/状态字段:"]

        for table in metadata.tables:
            enum_cols = table.enum_columns
            if enum_cols:
                for col in enum_cols:
                    parts.append(f"- {table.name}.{col.name}")
                    if col.sample_values:
                        parts.append(f"  可选值: {col.sample_values}")

        if len(parts) == 1:
            parts.append("  (未发现枚举类型字段)")

        return "\n".join(parts)


__all__ = [
    "SchemaIntrospector",
    "SchemaMetadata",
    "TableInfo",
    "ColumnInfo",
]
