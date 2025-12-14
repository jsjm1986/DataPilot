# -*- coding: utf-8 -*-
"""
Audit Logging System (P2 增强版)

提供完整的审计日志功能:
- 结构化日志记录
- SQLite 持久化存储
- 自动日志保留策略 (可配置天数)
- 查询和搜索功能
- 统计分析支持
- 异步批量写入
"""

import asyncio
import json
import time
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any, Literal
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from threading import Lock
import structlog

logger = structlog.get_logger("audit")


@dataclass
class AuditEntry:
    """审计日志条目"""
    # 必填字段
    user_id: str
    tenant_id: str
    trace_id: str
    session_id: str
    database: str
    query: str
    status: str  # success, error, rejected, timeout

    # 可选字段
    sql: Optional[str] = None
    row_count: Optional[int] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    # 执行详情
    agent_path: list[str] = field(default_factory=list)  # 执行的 Agent 路径
    agent_details: Optional[dict] = None  # Agent 详细执行信息 (包含每个 Agent 的 steps, llm_calls 等)
    cache_hit: bool = False
    cost_analysis: Optional[dict] = None
    chart_type: Optional[str] = None

    # 元数据
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class AuditStore:
    """
    审计日志存储管理器 (P2 新增)

    功能:
    - SQLite 持久化存储
    - 异步批量写入
    - 自动日志保留
    - 查询和搜索
    """

    def __init__(
        self,
        db_path: str = "data/audit/audit.db",
        retention_days: int = 90,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
    ):
        """
        Args:
            db_path: SQLite 数据库路径
            retention_days: 日志保留天数
            batch_size: 批量写入大小
            flush_interval_seconds: 刷新间隔
        """
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds

        # 写入缓冲
        self._buffer: list[AuditEntry] = []
        self._lock = Lock()

        # 后台任务
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # 初始化数据库
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    database TEXT NOT NULL,
                    query TEXT NOT NULL,
                    sql TEXT,
                    status TEXT NOT NULL,
                    row_count INTEGER,
                    duration_ms REAL,
                    error TEXT,
                    agent_path TEXT,
                    agent_details TEXT,
                    cache_hit INTEGER,
                    cost_analysis TEXT,
                    chart_type TEXT,
                    extra TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 尝试添加 agent_details 列（如果不存在）
            try:
                conn.execute("ALTER TABLE audit_logs ADD COLUMN agent_details TEXT")
            except sqlite3.OperationalError:
                pass  # 列已存在

            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tenant_id ON audit_logs(tenant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_id ON audit_logs(trace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON audit_logs(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_database ON audit_logs(database)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON audit_logs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON audit_logs(created_at)")

            conn.commit()

    async def start(self):
        """启动后台刷新任务"""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush())
        logger.info("audit_store_started", db_path=str(self.db_path))

    async def stop(self):
        """停止后台任务并刷新剩余数据"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # 刷新剩余数据
        await self._flush()
        logger.info("audit_store_stopped")

    async def _background_flush(self):
        """后台定期刷新"""
        while self._running:
            await asyncio.sleep(self.flush_interval_seconds)
            await self._flush()

    async def _flush(self):
        """刷新缓冲区到数据库"""
        with self._lock:
            if not self._buffer:
                return
            entries = self._buffer.copy()
            self._buffer.clear()

        if not entries:
            return

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._write_batch, entries
            )
            logger.debug("audit_flushed", count=len(entries))
        except Exception as e:
            logger.error("audit_flush_error", error=str(e))
            # 重新放回缓冲区
            with self._lock:
                self._buffer = entries + self._buffer

    def _write_batch(self, entries: list[AuditEntry]):
        """批量写入数据库 (同步)"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executemany(
                """
                INSERT INTO audit_logs (
                    id, timestamp, user_id, tenant_id, trace_id, session_id,
                    database, query, sql, status, row_count, duration_ms, error,
                    agent_path, agent_details, cache_hit, cost_analysis, chart_type, extra
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.id,
                        e.timestamp,
                        e.user_id,
                        e.tenant_id,
                        e.trace_id,
                        e.session_id,
                        e.database,
                        e.query,
                        e.sql,
                        e.status,
                        e.row_count,
                        e.duration_ms,
                        e.error,
                        json.dumps(e.agent_path),
                        json.dumps(e.agent_details) if e.agent_details else None,
                        1 if e.cache_hit else 0,
                        json.dumps(e.cost_analysis) if e.cost_analysis else None,
                        e.chart_type,
                        json.dumps(e.extra) if e.extra else None,
                    )
                    for e in entries
                ]
            )
            conn.commit()

    def log(self, entry: AuditEntry):
        """添加日志条目到缓冲区"""
        with self._lock:
            self._buffer.append(entry)

            # 达到批量大小时立即刷新
            if len(self._buffer) >= self.batch_size:
                entries = self._buffer.copy()
                self._buffer.clear()

        # 在后台刷新
        if len(entries) >= self.batch_size if 'entries' in dir() else False:
            asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, self._write_batch, entries
                )
            )

        # 同时写入 structlog
        logger.info(
            "audit_query",
            **{k: v for k, v in entry.to_dict().items() if v is not None}
        )

    async def cleanup_old_logs(self):
        """清理过期日志"""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        cutoff_str = cutoff.isoformat() + "Z"

        def _cleanup():
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_logs WHERE timestamp < ?",
                    (cutoff_str,)
                )
                deleted = cursor.rowcount
                conn.commit()
                return deleted

        deleted = await asyncio.get_event_loop().run_in_executor(None, _cleanup)
        logger.info("audit_cleanup", deleted=deleted, cutoff=cutoff_str)
        return deleted

    async def query(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        database: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "timestamp",
        order_dir: Literal["ASC", "DESC"] = "DESC",
    ) -> list[dict]:
        """
        查询审计日志

        Args:
            user_id: 用户 ID 过滤
            tenant_id: 租户 ID 过滤
            trace_id: 追踪 ID 过滤
            session_id: 会话 ID 过滤
            database: 数据库过滤
            status: 状态过滤
            start_time: 开始时间 (ISO 格式)
            end_time: 结束时间 (ISO 格式)
            search_query: 搜索查询文本
            limit: 返回数量
            offset: 偏移量
            order_by: 排序字段
            order_dir: 排序方向

        Returns:
            日志条目列表
        """
        def _query():
            conditions = []
            params = []

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if tenant_id:
                conditions.append("tenant_id = ?")
                params.append(tenant_id)
            if trace_id:
                conditions.append("trace_id = ?")
                params.append(trace_id)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            if database:
                conditions.append("database = ?")
                params.append(database)
            if status:
                conditions.append("status = ?")
                params.append(status)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            if search_query:
                conditions.append("(query LIKE ? OR sql LIKE ?)")
                params.extend([f"%{search_query}%", f"%{search_query}%"])

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # 验证排序字段
            valid_order_fields = ["timestamp", "user_id", "database", "status", "duration_ms", "row_count"]
            if order_by not in valid_order_fields:
                order_by_safe = "timestamp"
            else:
                order_by_safe = order_by

            sql = f"""
                SELECT id, timestamp, user_id, tenant_id, trace_id, session_id,
                       database, query, sql, status, row_count, duration_ms, error,
                       agent_path, agent_details, cache_hit, cost_analysis, chart_type, extra
                FROM audit_logs
                WHERE {where_clause}
                ORDER BY {order_by_safe} {order_dir}
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "user_id": row["user_id"],
                    "tenant_id": row["tenant_id"],
                    "trace_id": row["trace_id"],
                    "session_id": row["session_id"],
                    "database": row["database"],
                    "query": row["query"],
                    "sql": row["sql"],
                    "status": row["status"],
                    "row_count": row["row_count"],
                    "duration_ms": row["duration_ms"],
                    "error": row["error"],
                    "agent_path": json.loads(row["agent_path"]) if row["agent_path"] else [],
                    "agent_details": json.loads(row["agent_details"]) if row["agent_details"] else None,
                    "cache_hit": bool(row["cache_hit"]),
                    "cost_analysis": json.loads(row["cost_analysis"]) if row["cost_analysis"] else None,
                    "chart_type": row["chart_type"],
                    "extra": json.loads(row["extra"]) if row["extra"] else {},
                }
                for row in rows
            ]

        return await asyncio.get_event_loop().run_in_executor(None, _query)

    async def get_statistics(
        self,
        tenant_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> dict:
        """
        获取统计信息

        Args:
            tenant_id: 租户 ID 过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            统计信息
        """
        def _stats():
            conditions = []
            params = []

            if tenant_id:
                conditions.append("tenant_id = ?")
                params.append(tenant_id)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            with sqlite3.connect(str(self.db_path)) as conn:
                # 总体统计
                cursor = conn.execute(f"""
                    SELECT
                        COUNT(*) as total_queries,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                        SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected_count,
                        SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                        AVG(duration_ms) as avg_duration_ms,
                        MAX(duration_ms) as max_duration_ms,
                        MIN(duration_ms) as min_duration_ms,
                        SUM(row_count) as total_rows
                    FROM audit_logs
                    WHERE {where_clause}
                """, params)
                row = cursor.fetchone()

                overall = {
                    "total_queries": row[0] or 0,
                    "success_count": row[1] or 0,
                    "error_count": row[2] or 0,
                    "rejected_count": row[3] or 0,
                    "cache_hits": row[4] or 0,
                    "cache_hit_rate": (row[4] or 0) / (row[0] or 1),
                    "avg_duration_ms": row[5] or 0,
                    "max_duration_ms": row[6] or 0,
                    "min_duration_ms": row[7] or 0,
                    "total_rows": row[8] or 0,
                }

                # 按数据库统计
                cursor = conn.execute(f"""
                    SELECT database, COUNT(*) as count, AVG(duration_ms) as avg_duration
                    FROM audit_logs
                    WHERE {where_clause}
                    GROUP BY database
                """, params)
                by_database = {
                    row[0]: {"count": row[1], "avg_duration_ms": row[2] or 0}
                    for row in cursor.fetchall()
                }

                # 按状态统计
                cursor = conn.execute(f"""
                    SELECT status, COUNT(*) as count
                    FROM audit_logs
                    WHERE {where_clause}
                    GROUP BY status
                """, params)
                by_status = {row[0]: row[1] for row in cursor.fetchall()}

                # 按用户统计 (Top 10)
                cursor = conn.execute(f"""
                    SELECT user_id, COUNT(*) as count
                    FROM audit_logs
                    WHERE {where_clause}
                    GROUP BY user_id
                    ORDER BY count DESC
                    LIMIT 10
                """, params)
                top_users = [
                    {"user_id": row[0], "count": row[1]}
                    for row in cursor.fetchall()
                ]

                # 每日趋势 (最近 30 天)
                cursor = conn.execute(f"""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM audit_logs
                    WHERE {where_clause}
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 30
                """, params)
                daily_trend = [
                    {"date": row[0], "count": row[1]}
                    for row in cursor.fetchall()
                ]

                return {
                    "overall": overall,
                    "by_database": by_database,
                    "by_status": by_status,
                    "top_users": top_users,
                    "daily_trend": daily_trend,
                }

        return await asyncio.get_event_loop().run_in_executor(None, _stats)

    async def get_by_trace_id(self, trace_id: str) -> Optional[dict]:
        """根据 trace_id 获取单条日志"""
        results = await self.query(trace_id=trace_id, limit=1)
        return results[0] if results else None

    async def count(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> int:
        """获取日志数量"""
        def _count():
            conditions = []
            params = []

            if tenant_id:
                conditions.append("tenant_id = ?")
                params.append(tenant_id)
            if status:
                conditions.append("status = ?")
                params.append(status)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM audit_logs WHERE {where_clause}",
                    params
                )
                return cursor.fetchone()[0]

        return await asyncio.get_event_loop().run_in_executor(None, _count)


# 全局审计存储实例
_audit_store: Optional[AuditStore] = None


def get_audit_store() -> AuditStore:
    """获取全局审计存储实例"""
    global _audit_store
    if _audit_store is None:
        from ..config.settings import get_settings
        settings = get_settings()
        _audit_store = AuditStore(
            db_path="data/audit/audit.db",
            retention_days=settings.audit_retention_days,
        )
    return _audit_store


async def init_audit_store():
    """初始化审计存储 (在应用启动时调用)"""
    store = get_audit_store()
    await store.start()
    return store


async def shutdown_audit_store():
    """关闭审计存储 (在应用关闭时调用)"""
    global _audit_store
    if _audit_store:
        await _audit_store.stop()
        _audit_store = None


class AuditTimer:
    """Context manager for timing durations."""

    def __init__(self):
        self._start: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


@asynccontextmanager
async def audit_context(
    user_id: str,
    tenant_id: str,
    trace_id: str,
    session_id: str,
    database: str,
    query: str,
):
    """
    审计上下文管理器 (P2 新增)

    自动记录查询的开始和结束，包括执行时间和状态

    Usage:
        async with audit_context(user_id="u1", ...) as audit:
            # 执行查询
            audit.sql = "SELECT ..."
            audit.row_count = 100
    """
    entry = AuditEntry(
        user_id=user_id,
        tenant_id=tenant_id,
        trace_id=trace_id,
        session_id=session_id,
        database=database,
        query=query,
        status="success",
    )

    start_time = time.perf_counter()
    error_occurred = False

    try:
        yield entry
    except Exception as e:
        error_occurred = True
        entry.status = "error"
        entry.error = str(e)
        raise
    finally:
        entry.duration_ms = (time.perf_counter() - start_time) * 1000

        # 写入审计日志
        store = get_audit_store()
        store.log(entry)


def log_audit(
    *,
    user_id: str,
    tenant_id: str,
    trace_id: str,
    session_id: str,
    database: str,
    query: str,
    sql: Optional[str],
    status: str,
    row_count: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    agent_path: Optional[list[str]] = None,
    agent_details: Optional[dict] = None,
    cache_hit: bool = False,
    cost_analysis: Optional[dict] = None,
    chart_type: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Write a structured audit log entry (兼容原接口).

    Args:
        user_id: 用户 ID
        tenant_id: 租户 ID
        trace_id: 追踪 ID
        session_id: 会话 ID
        database: 数据库名
        query: 原始查询
        sql: 生成的 SQL
        status: 状态 (success/error/rejected/timeout)
        row_count: 返回行数
        duration_ms: 执行时间
        error: 错误信息
        agent_path: Agent 执行路径
        agent_details: Agent 详细执行信息
        cache_hit: 是否缓存命中
        cost_analysis: 成本分析结果
        chart_type: 图表类型
        extra: 额外信息
    """
    entry = AuditEntry(
        user_id=user_id,
        tenant_id=tenant_id,
        trace_id=trace_id,
        session_id=session_id,
        database=database,
        query=query,
        sql=sql,
        status=status,
        row_count=row_count,
        duration_ms=duration_ms,
        error=error,
        agent_path=agent_path or [],
        agent_details=agent_details,
        cache_hit=cache_hit,
        cost_analysis=cost_analysis,
        chart_type=chart_type,
        extra=extra or {},
    )

    # 写入存储
    try:
        store = get_audit_store()
        store.log(entry)
    except Exception as e:
        # 存储失败时至少写入 structlog
        logger.warning("audit_store_error", error=str(e))
        logger.info(
            "audit_query",
            **{k: v for k, v in entry.to_dict().items() if v is not None}
        )


__all__ = [
    # 数据类
    "AuditEntry",
    # 存储
    "AuditStore",
    "get_audit_store",
    "init_audit_store",
    "shutdown_audit_store",
    # 工具
    "AuditTimer",
    "audit_context",
    # 函数
    "log_audit",
]
