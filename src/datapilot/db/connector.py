"""
DataPilot 多数据库连接器
支持 MySQL、PostgreSQL、SQLite、SQL Server、ClickHouse、DuckDB 等多种数据库
"""

import asyncio
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal, Optional

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram

from ..config.settings import get_settings

# 数据库类型
DBType = Literal["mysql", "postgresql", "sqlite", "sqlserver", "clickhouse", "duckdb"]

# Prometheus metrics
DB_QUERY_COUNT = Counter(
    "datapilot_db_query_total",
    "Database query count",
    labelnames=["db_type", "status"],
)
DB_QUERY_LATENCY = Histogram(
    "datapilot_db_query_latency_seconds",
    "Database query latency",
    labelnames=["db_type", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)


class DatabaseConnector(ABC):
    """数据库连接器抽象基类"""

    def __init__(self, db_type: DBType, connection_url: str):
        self.db_type = db_type
        self.connection_url = connection_url
        self._engine = None
        self._async_engine = None

    @abstractmethod
    async def get_schema(self) -> str:
        """获取数据库 Schema (DDL)"""
        pass

    @abstractmethod
    async def execute_query(self, sql: str, params: Optional[dict] = None) -> list[dict]:
        """执行查询并返回结果"""
        pass

    @abstractmethod
    async def get_tables(self) -> list[dict]:
        """获取所有表信息"""
        pass


class MySQLConnector(DatabaseConnector):
    """MySQL 数据库连接器"""

    def __init__(self, connection_url: Optional[str] = None):
        settings = get_settings()
        url = connection_url or settings.mysql_url
        super().__init__("mysql", url)

        # 创建异步引擎
        self._async_engine = create_async_engine(
            url,
            echo=settings.datapilot_debug,
            pool_size=5,
            max_overflow=10,
        )

        # 创建同步引擎（用于 Schema 获取）
        sync_url = connection_url or settings.mysql_sync_url
        self._sync_engine = create_engine(sync_url, echo=False)

        # Session 工厂
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_schema(self) -> str:
        """获取数据库 Schema"""
        inspector = inspect(self._sync_engine)
        schema_parts = []

        for table_name in inspector.get_table_names():
            # 获取表注释
            table_comment = inspector.get_table_comment(table_name).get("text", "")

            # 获取列信息
            columns = inspector.get_columns(table_name)
            pk_columns = inspector.get_pk_constraint(table_name).get("constrained_columns", [])

            # 构建 CREATE TABLE 语句
            col_defs = []
            for col in columns:
                col_def = f"  {col['name']} {col['type']}"
                if not col.get("nullable", True):
                    col_def += " NOT NULL"
                if col["name"] in pk_columns:
                    col_def += " PRIMARY KEY"
                if col.get("comment"):
                    col_def += f" COMMENT '{col['comment']}'"
                col_defs.append(col_def)

            create_stmt = f"-- Table: {table_name}"
            if table_comment:
                create_stmt += f"\n-- Comment: {table_comment}"
            create_stmt += f"\nCREATE TABLE {table_name} (\n"
            create_stmt += ",\n".join(col_defs)
            create_stmt += "\n);\n"

            schema_parts.append(create_stmt)

        return "\n".join(schema_parts)

    async def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """执行查询"""
        sql_lower = sql.lower().strip()
        if "limit" not in sql_lower and sql_lower.startswith("select"):
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        status = "success"
        from time import perf_counter

        start = perf_counter()
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            DB_QUERY_COUNT.labels(db_type=self.db_type, status=status).inc()
            DB_QUERY_LATENCY.labels(db_type=self.db_type, status=status).observe(duration)

    async def get_tables(self) -> list[dict]:
        """获取所有表信息"""
        inspector = inspect(self._sync_engine)
        tables = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            table_comment = inspector.get_table_comment(table_name).get("text", "")

            tables.append({
                "name": table_name,
                "comment": table_comment,
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "comment": col.get("comment", ""),
                    }
                    for col in columns
                ],
            })

        return tables

    async def close(self):
        """关闭连接"""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()


class SQLiteConnector(DatabaseConnector):
    """SQLite 数据库连接器"""

    def __init__(self, connection_url: Optional[str] = None):
        settings = get_settings()
        url = connection_url or settings.sqlite_url
        super().__init__("sqlite", url)

        # 创建异步引擎
        self._async_engine = create_async_engine(
            url,
            echo=settings.datapilot_debug,
        )

        # 创建同步引擎
        sync_url = connection_url or settings.sqlite_sync_url
        if sync_url.startswith("sqlite+aiosqlite"):
            sync_url = sync_url.replace("sqlite+aiosqlite", "sqlite")
        self._sync_engine = create_engine(sync_url, echo=False)

        # Session 工厂
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_schema(self) -> str:
        """获取数据库 Schema"""
        inspector = inspect(self._sync_engine)
        schema_parts = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            pk_columns = inspector.get_pk_constraint(table_name).get("constrained_columns", [])

            col_defs = []
            for col in columns:
                col_def = f"  {col['name']} {col['type']}"
                if not col.get("nullable", True):
                    col_def += " NOT NULL"
                if col["name"] in pk_columns:
                    col_def += " PRIMARY KEY"
                col_defs.append(col_def)

            create_stmt = f"-- Table: {table_name}"
            create_stmt += f"\nCREATE TABLE {table_name} (\n"
            create_stmt += ",\n".join(col_defs)
            create_stmt += "\n);\n"

            schema_parts.append(create_stmt)

        return "\n".join(schema_parts)

    async def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """执行查询"""
        sql_lower = sql.lower().strip()
        if "limit" not in sql_lower and sql_lower.startswith("select"):
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        status = "success"
        from time import perf_counter

        start = perf_counter()
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            DB_QUERY_COUNT.labels(db_type=self.db_type, status=status).inc()
            DB_QUERY_LATENCY.labels(db_type=self.db_type, status=status).observe(duration)

    async def get_tables(self) -> list[dict]:
        """获取所有表信息"""
        inspector = inspect(self._sync_engine)
        tables = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)

            tables.append({
                "name": table_name,
                "comment": "",
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "comment": "",
                    }
                    for col in columns
                ],
            })

        return tables

    async def close(self):
        """关闭连接"""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL 数据库连接器"""

    def __init__(self, connection_url: Optional[str] = None):
        settings = get_settings()
        url = connection_url or settings.postgres_url
        super().__init__("postgresql", url)

        self._async_engine = create_async_engine(
            url,
            echo=settings.datapilot_debug,
            pool_size=5,
            max_overflow=10,
        )

        sync_url = connection_url or settings.postgres_sync_url
        self._sync_engine = create_engine(sync_url, echo=False)

        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_schema(self) -> str:
        """获取数据库 Schema"""
        inspector = inspect(self._sync_engine)
        schema_parts = []

        for table_name in inspector.get_table_names():
            table_comment = inspector.get_table_comment(table_name).get("text", "")
            columns = inspector.get_columns(table_name)
            pk_columns = inspector.get_pk_constraint(table_name).get("constrained_columns", [])

            col_defs = []
            for col in columns:
                col_def = f"  {col['name']} {col['type']}"
                if not col.get("nullable", True):
                    col_def += " NOT NULL"
                if col["name"] in pk_columns:
                    col_def += " PRIMARY KEY"
                col_defs.append(col_def)

            create_stmt = f"-- Table: {table_name}"
            if table_comment:
                create_stmt += f"\n-- Comment: {table_comment}"
            create_stmt += f"\nCREATE TABLE {table_name} (\n"
            create_stmt += ",\n".join(col_defs)
            create_stmt += "\n);\n"

            schema_parts.append(create_stmt)

        return "\n".join(schema_parts)

    async def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """执行查询"""
        sql_lower = sql.lower().strip()
        if "limit" not in sql_lower and sql_lower.startswith("select"):
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        status = "success"
        from time import perf_counter

        start = perf_counter()
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            DB_QUERY_COUNT.labels(db_type=self.db_type, status=status).inc()
            DB_QUERY_LATENCY.labels(db_type=self.db_type, status=status).observe(duration)

    async def get_tables(self) -> list[dict]:
        """获取所有表信息"""
        inspector = inspect(self._sync_engine)
        tables = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            table_comment = inspector.get_table_comment(table_name).get("text", "")

            tables.append({
                "name": table_name,
                "comment": table_comment,
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "comment": col.get("comment", ""),
                    }
                    for col in columns
                ],
            })

        return tables

    async def close(self):
        """关闭连接"""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()


class SQLServerConnector(DatabaseConnector):
    """SQL Server 数据库连接器"""

    def __init__(self, connection_url: Optional[str] = None):
        try:
            import pyodbc  # noqa: F401
        except ImportError:
            raise ImportError(
                "SQL Server 驱动未安装。请运行: pip install datapilot[sqlserver]"
            )

        settings = get_settings()
        url = connection_url or settings.sqlserver_url
        super().__init__("sqlserver", url)

        # 创建异步引擎
        self._async_engine = create_async_engine(
            url,
            echo=settings.datapilot_debug,
            pool_size=5,
            max_overflow=10,
        )

        # 创建同步引擎
        sync_url = connection_url or settings.sqlserver_sync_url
        self._sync_engine = create_engine(sync_url, echo=False)

        # Session 工厂
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_schema(self) -> str:
        """获取数据库 Schema"""
        inspector = inspect(self._sync_engine)
        schema_parts = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            pk_columns = inspector.get_pk_constraint(table_name).get("constrained_columns", [])

            col_defs = []
            for col in columns:
                col_def = f"  {col['name']} {col['type']}"
                if not col.get("nullable", True):
                    col_def += " NOT NULL"
                if col["name"] in pk_columns:
                    col_def += " PRIMARY KEY"
                col_defs.append(col_def)

            create_stmt = f"-- Table: {table_name}"
            create_stmt += f"\nCREATE TABLE {table_name} (\n"
            create_stmt += ",\n".join(col_defs)
            create_stmt += "\n);\n"

            schema_parts.append(create_stmt)

        return "\n".join(schema_parts)

    async def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """执行查询"""
        import re
        sql_lower = sql.lower().strip()
        # SQL Server 使用 TOP 而不是 LIMIT
        if "top" not in sql_lower and sql_lower.startswith("select"):
            # 使用正则表达式只替换一次，避免重复替换
            sql = re.sub(r'^select\s+', f'SELECT TOP {limit} ', sql, count=1, flags=re.IGNORECASE)

        status = "success"
        from time import perf_counter

        start = perf_counter()
        try:
            async with self.get_session() as session:
                result = await session.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            DB_QUERY_COUNT.labels(db_type=self.db_type, status=status).inc()
            DB_QUERY_LATENCY.labels(db_type=self.db_type, status=status).observe(duration)

    async def get_tables(self) -> list[dict]:
        """获取所有表信息"""
        inspector = inspect(self._sync_engine)
        tables = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)

            tables.append({
                "name": table_name,
                "comment": "",
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "comment": "",
                    }
                    for col in columns
                ],
            })

        return tables

    async def close(self):
        """关闭连接"""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()


class ClickHouseConnector(DatabaseConnector):
    """ClickHouse 数据库连接器

    使用 asyncio.to_thread 包装同步客户端，避免阻塞事件循环
    """

    def __init__(self, connection_url: Optional[str] = None, **kwargs):
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError(
                "ClickHouse 驱动未安装。请运行: pip install datapilot[clickhouse]"
            )

        settings = get_settings()
        super().__init__("clickhouse", connection_url or settings.clickhouse_url)

        # 支持通过 kwargs 或 settings 获取连接参数
        host = kwargs.get("host") or settings.clickhouse_host
        port = kwargs.get("port") or settings.clickhouse_port
        username = kwargs.get("username") or kwargs.get("user") or settings.clickhouse_user
        password = kwargs.get("password") or settings.clickhouse_password
        database = kwargs.get("database") or settings.clickhouse_database

        # ClickHouse 使用 clickhouse-connect 客户端
        self._client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )
        self._settings = settings
        self._host = host
        self._port = port
        self._database = database

    def _sync_query(self, sql: str, params: Optional[dict] = None):
        """同步查询方法，供 asyncio.to_thread 调用"""
        return self._client.query(sql, parameters=params)

    async def get_schema(self) -> str:
        """获取数据库 Schema (异步包装)"""
        return await asyncio.to_thread(self._sync_get_schema)

    def _validate_identifier(self, name: str) -> str:
        """验证并清理标识符，防止 SQL 注入"""
        import re
        # 只允许字母、数字、下划线
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"Invalid identifier: {name}")
        return name

    def _sync_get_schema(self) -> str:
        """同步获取 Schema"""
        schema_parts = []

        # 获取所有表
        tables_result = self._client.query("SHOW TABLES")
        for row in tables_result.result_rows:
            table_name = row[0]

            # 验证表名，防止 SQL 注入
            try:
                safe_table_name = self._validate_identifier(table_name)
                safe_database = self._validate_identifier(self._database)
            except ValueError:
                continue  # 跳过无效的表名

            # 获取表注释 (使用参数化查询)
            table_comment = ""
            try:
                comment_result = self._client.query(
                    "SELECT comment FROM system.tables WHERE database = {db:String} AND name = {tbl:String}",
                    parameters={"db": safe_database, "tbl": safe_table_name}
                )
                if comment_result.result_rows:
                    table_comment = comment_result.result_rows[0][0] or ""
            except Exception:
                pass

            # 获取表结构 (使用反引号包裹标识符)
            desc_result = self._client.query(f"DESCRIBE TABLE `{safe_table_name}`")
            col_defs = []
            for col_row in desc_result.result_rows:
                col_name = col_row[0]
                col_type = col_row[1]
                col_comment = col_row[4] if len(col_row) > 4 else ""
                col_def = f"  {col_name} {col_type}"
                if col_comment:
                    # 转义单引号
                    col_comment_escaped = col_comment.replace("'", "''")
                    col_def += f" COMMENT '{col_comment_escaped}'"
                col_defs.append(col_def)

            create_stmt = f"-- Table: {safe_table_name}"
            if table_comment:
                create_stmt += f"\n-- Comment: {table_comment}"
            create_stmt += f"\nCREATE TABLE {safe_table_name} (\n"
            create_stmt += ",\n".join(col_defs)
            create_stmt += "\n) ENGINE = MergeTree();\n"

            schema_parts.append(create_stmt)

        return "\n".join(schema_parts)

    async def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """执行查询 (异步包装)"""
        sql_lower = sql.lower().strip()
        if "limit" not in sql_lower and sql_lower.startswith("select"):
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        status = "success"
        from time import perf_counter

        start = perf_counter()
        try:
            # 使用 asyncio.to_thread 避免阻塞事件循环
            result = await asyncio.to_thread(self._sync_query, sql, params)
            columns = result.column_names
            return [dict(zip(columns, row)) for row in result.result_rows]
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            DB_QUERY_COUNT.labels(db_type=self.db_type, status=status).inc()
            DB_QUERY_LATENCY.labels(db_type=self.db_type, status=status).observe(duration)

    async def get_tables(self) -> list[dict]:
        """获取所有表信息 (异步包装)"""
        return await asyncio.to_thread(self._sync_get_tables)

    async def get_columns(self, table_name: str) -> list[dict]:
        """获取指定表的列信息 (异步包装)"""
        return await asyncio.to_thread(self._sync_get_columns, table_name)

    def _sync_get_columns(self, table_name: str) -> list[dict]:
        """同步获取列信息"""
        try:
            safe_table_name = self._validate_identifier(table_name)
        except ValueError:
            return []

        columns = []
        desc_result = self._client.query(f"DESCRIBE TABLE `{safe_table_name}`")
        for col_row in desc_result.result_rows:
            columns.append({
                "name": col_row[0],
                "type": col_row[1],
                "nullable": "Nullable" in col_row[1],
                "comment": col_row[4] if len(col_row) > 4 else "",
            })
        return columns

    def _sync_get_tables(self) -> list[dict]:
        """同步获取表信息"""
        tables = []

        tables_result = self._client.query("SHOW TABLES")
        for row in tables_result.result_rows:
            table_name = row[0]

            # 验证表名，防止 SQL 注入
            try:
                safe_table_name = self._validate_identifier(table_name)
                safe_database = self._validate_identifier(self._database)
            except ValueError:
                continue  # 跳过无效的表名

            # 获取表注释 (使用参数化查询)
            table_comment = ""
            try:
                comment_result = self._client.query(
                    "SELECT comment FROM system.tables WHERE database = {db:String} AND name = {tbl:String}",
                    parameters={"db": safe_database, "tbl": safe_table_name}
                )
                if comment_result.result_rows:
                    table_comment = comment_result.result_rows[0][0] or ""
            except Exception:
                pass

            # 获取列信息 (使用反引号包裹标识符)
            desc_result = self._client.query(f"DESCRIBE TABLE `{safe_table_name}`")
            columns = []
            for col_row in desc_result.result_rows:
                columns.append({
                    "name": col_row[0],
                    "type": col_row[1],
                    "nullable": "Nullable" in col_row[1],
                    "comment": col_row[4] if len(col_row) > 4 else "",
                })

            tables.append({
                "name": safe_table_name,
                "comment": table_comment,
                "columns": columns,
            })

        return tables

    async def close(self):
        """关闭连接"""
        if self._client:
            await asyncio.to_thread(self._client.close)


class DuckDBConnector(DatabaseConnector):
    """DuckDB 数据库连接器

    使用 asyncio.to_thread 包装同步操作，并使用线程锁保证线程安全
    DuckDB 连接不是线程安全的，需要加锁保护
    """

    def __init__(self, connection_url: Optional[str] = None, **kwargs):
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "DuckDB 驱动未安装。请运行: pip install datapilot[duckdb]"
            )

        settings = get_settings()
        # 支持通过 kwargs 或 connection_url 或 settings 获取数据库路径
        db_path = kwargs.get("database") or settings.duckdb_database

        # 如果提供了 connection_url，尝试从中提取路径
        if connection_url and connection_url.startswith("duckdb:///"):
            db_path = connection_url.replace("duckdb:///", "")

        super().__init__("duckdb", f"duckdb:///{db_path}")

        # DuckDB 使用原生连接
        self._conn = duckdb.connect(db_path)
        self._db_path = db_path
        # 线程锁，保证 DuckDB 连接的线程安全
        self._lock = threading.Lock()

    def _sync_execute(self, sql: str, params: Optional[dict] = None):
        """同步执行查询，带线程锁保护"""
        with self._lock:
            if params:
                result = self._conn.execute(sql, params)
            else:
                result = self._conn.execute(sql)
            # 立即获取结果，避免锁释放后结果失效
            columns = [desc[0] for desc in result.description] if result.description else []
            rows = result.fetchall()
            return columns, rows

    async def get_schema(self) -> str:
        """获取数据库 Schema (异步包装)"""
        return await asyncio.to_thread(self._sync_get_schema)

    def _sync_get_schema(self) -> str:
        """同步获取 Schema"""
        schema_parts = []

        with self._lock:
            # 获取所有表
            tables = self._conn.execute("SHOW TABLES").fetchall()
            for row in tables:
                # 安全获取表名 (可能是元组或单值)
                table_name = row[0] if isinstance(row, (tuple, list)) else row
                # 获取表结构
                columns = self._conn.execute(f'DESCRIBE "{table_name}"').fetchall()
                col_defs = []
                for col in columns:
                    col_name = col[0] if len(col) > 0 else "unknown"
                    col_type = col[1] if len(col) > 1 else "unknown"
                    # 安全获取 nullable，处理不同 DuckDB 版本
                    nullable = True
                    if len(col) > 2:
                        nullable_val = col[2]
                        if isinstance(nullable_val, str):
                            nullable = nullable_val.upper() == "YES"
                        elif isinstance(nullable_val, bool):
                            nullable = nullable_val
                    col_def = f"  {col_name} {col_type}"
                    if not nullable:
                        col_def += " NOT NULL"
                    col_defs.append(col_def)

                create_stmt = f"-- Table: {table_name}"
                create_stmt += f"\nCREATE TABLE {table_name} (\n"
                create_stmt += ",\n".join(col_defs)
                create_stmt += "\n);\n"

                schema_parts.append(create_stmt)

        return "\n".join(schema_parts)

    async def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """执行查询 (异步包装)"""
        sql_lower = sql.lower().strip()
        if "limit" not in sql_lower and sql_lower.startswith("select"):
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        status = "success"
        from time import perf_counter

        start = perf_counter()
        try:
            # 使用 asyncio.to_thread 避免阻塞事件循环
            columns, rows = await asyncio.to_thread(self._sync_execute, sql, params)
            return [dict(zip(columns, row)) for row in rows]
        except Exception:
            status = "error"
            raise
        finally:
            duration = perf_counter() - start
            DB_QUERY_COUNT.labels(db_type=self.db_type, status=status).inc()
            DB_QUERY_LATENCY.labels(db_type=self.db_type, status=status).observe(duration)

    async def get_tables(self) -> list[dict]:
        """获取所有表信息 (异步包装)"""
        return await asyncio.to_thread(self._sync_get_tables)

    async def get_columns(self, table_name: str) -> list[dict]:
        """获取指定表的列信息 (异步包装)"""
        return await asyncio.to_thread(self._sync_get_columns, table_name)

    def _sync_get_columns(self, table_name: str) -> list[dict]:
        """同步获取列信息"""
        import re
        # 验证表名，防止 SQL 注入
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            return []

        columns = []
        with self._lock:
            columns_result = self._conn.execute(f'DESCRIBE "{table_name}"').fetchall()
            for col in columns_result:
                col_name = col[0] if len(col) > 0 else "unknown"
                col_type = col[1] if len(col) > 1 else "unknown"
                # 安全获取 nullable，处理不同 DuckDB 版本
                nullable = True
                if len(col) > 2:
                    nullable_val = col[2]
                    if isinstance(nullable_val, str):
                        nullable = nullable_val.upper() == "YES"
                    elif isinstance(nullable_val, bool):
                        nullable = nullable_val
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "nullable": nullable,
                    "comment": "",
                })
        return columns

    def _sync_get_tables(self) -> list[dict]:
        """同步获取表信息"""
        tables = []

        with self._lock:
            tables_result = self._conn.execute("SHOW TABLES").fetchall()
            for row in tables_result:
                # 安全获取表名 (可能是元组或单值)
                table_name = row[0] if isinstance(row, (tuple, list)) else row
                # 获取列信息
                columns_result = self._conn.execute(f'DESCRIBE "{table_name}"').fetchall()
                columns = []
                for col in columns_result:
                    col_name = col[0] if len(col) > 0 else "unknown"
                    col_type = col[1] if len(col) > 1 else "unknown"
                    # 安全获取 nullable，处理不同 DuckDB 版本
                    nullable = True
                    if len(col) > 2:
                        nullable_val = col[2]
                        if isinstance(nullable_val, str):
                            nullable = nullable_val.upper() == "YES"
                        elif isinstance(nullable_val, bool):
                            nullable = nullable_val
                    columns.append({
                        "name": col_name,
                        "type": col_type,
                        "nullable": nullable,
                        "comment": "",
                    })

                tables.append({
                    "name": table_name,
                    "comment": "",
                    "columns": columns,
                })

        return tables

    async def close(self):
        """关闭连接"""
        if self._conn:
            with self._lock:
                self._conn.close()


class DatabaseManager:
    """数据库管理器 - 管理多个数据库连接"""

    def __init__(self):
        self._connectors: dict[str, DatabaseConnector] = {}
        self._default_db: Optional[str] = None

    def register(
        self,
        name: str,
        db_type: DBType,
        connection_url: Optional[str] = None,
        is_default: bool = False,
    ) -> DatabaseConnector:
        """注册数据库连接"""
        if db_type == "mysql":
            connector = MySQLConnector(connection_url)
        elif db_type == "postgresql":
            connector = PostgreSQLConnector(connection_url)
        elif db_type == "sqlite":
            connector = SQLiteConnector(connection_url)
        elif db_type == "sqlserver":
            connector = SQLServerConnector(connection_url)
        elif db_type == "clickhouse":
            connector = ClickHouseConnector(connection_url)
        elif db_type == "duckdb":
            connector = DuckDBConnector(connection_url)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        self._connectors[name] = connector

        if is_default or self._default_db is None:
            self._default_db = name

        return connector

    def get(self, name: Optional[str] = None) -> DatabaseConnector:
        """获取数据库连接器"""
        db_name = name or self._default_db

        if db_name is None:
            raise ValueError("No database registered")

        if db_name not in self._connectors:
            raise ValueError(f"Database not found: {db_name}")

        return self._connectors[db_name]

    @property
    def default(self) -> DatabaseConnector:
        """获取默认数据库连接器"""
        return self.get()

    async def close_all(self):
        """关闭所有连接"""
        for connector in self._connectors.values():
            await connector.close()


# 全局数据库管理器
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取数据库管理器单例

    根据用户配置的 default_db_type 注册默认数据库连接
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()

        # 根据用户配置注册默认数据库
        settings = get_settings()
        db_type = settings.default_db_type

        # 注册默认数据库连接
        _db_manager.register("default", db_type, is_default=True)

    return _db_manager


def reset_db_manager() -> None:
    """重置数据库管理器 (用于配置变更后重新初始化)"""
    global _db_manager
    if _db_manager is not None:
        # 异步关闭需要在事件循环中执行
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_db_manager.close_all())
        except RuntimeError:
            # 没有运行中的事件循环，同步关闭
            pass
        _db_manager = None


# 导出
__all__ = [
    "DatabaseConnector",
    "MySQLConnector",
    "PostgreSQLConnector",
    "SQLiteConnector",
    "SQLServerConnector",
    "ClickHouseConnector",
    "DuckDBConnector",
    "DatabaseManager",
    "get_db_manager",
    "reset_db_manager",
    "DBType",
]
