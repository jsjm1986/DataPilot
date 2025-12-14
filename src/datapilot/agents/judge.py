# -*- coding: utf-8 -*-
"""
Judge Agent
负责 SQL 校验、安全检查、成本熔断和执行计划分析

增强功能:
- SQL 修复建议生成
- 自动修复尝试
- **完整 EXPLAIN 成本分析** (P2 增强)
- **智能成本熔断策略** (P2 增强)
"""

import re
from typing import Any, Optional
from dataclasses import dataclass, field
from ..core.state import DataPilotState
from ..db.connector import get_db_manager
from ..config.settings import get_settings
from ..mcp.tools import plan_explain
from .sql_fixer import SQLFixer, FixSuggestion

# Prometheus 指标 (可选)
try:
    from ..observability.metrics import record_cost_rejection, record_error
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    def record_cost_rejection(db_type: str, reason: str): pass
    def record_error(error_type: str): pass


@dataclass
class CostAnalysisResult:
    """成本分析结果"""
    analyzed: bool = False
    sql: str = ""
    db_type: str = ""

    # 成本指标
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    rows_examined: int = 0

    # 执行计划信息
    scan_type: str = "unknown"
    uses_index: bool = False
    tables_accessed: list = field(default_factory=list)
    index_used: str = ""

    # 风险评估
    risks: list = field(default_factory=list)
    risk_score: float = 0.0  # 0-1, 越高风险越大

    # 熔断决策
    should_reject: bool = False
    reject_reason: str = ""

    # 建议
    recommendations: list = field(default_factory=list)

    # 原始计划
    raw_plan: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "analyzed": self.analyzed,
            "sql": self.sql,
            "db_type": self.db_type,
            "estimated_cost": self.estimated_cost,
            "estimated_rows": self.estimated_rows,
            "rows_examined": self.rows_examined,
            "scan_type": self.scan_type,
            "uses_index": self.uses_index,
            "tables_accessed": self.tables_accessed,
            "index_used": self.index_used,
            "risks": self.risks,
            "risk_score": self.risk_score,
            "should_reject": self.should_reject,
            "reject_reason": self.reject_reason,
            "recommendations": self.recommendations,
            "plan": self.raw_plan,
        }


class Judge:
    """
    裁判 Agent - 负责 SQL 校验、安全检查和成本熔断

    职责:
    1. SQL 语法和安全校验
    2. 执行计划分析
    3. 成本熔断 (防止高成本查询)
    4. 全表扫描/笛卡尔积检测
    """

    # 危险关键字
    DANGEROUS_KEYWORDS = [
        "drop", "truncate", "delete", "update", "insert",
        "alter", "create", "grant", "revoke", "exec", "execute",
    ]

    # SQL 注入模式
    INJECTION_PATTERNS = [
        r";\s*--",           # 注释注入
        r";\s*drop",         # 多语句注入
        r"union\s+select",   # UNION 注入
        r"or\s+1\s*=\s*1",   # 永真条件
        r"'\s*or\s*'",       # 字符串注入
        r"--\s*$",           # 行尾注释
        r"/\*.*\*/",         # 块注释
    ]

    # 笛卡尔积检测模式
    CARTESIAN_PATTERNS = [
        r"cross\s+join",                    # 显式 CROSS JOIN
        r"from\s+\w+\s*,\s*\w+",            # 隐式笛卡尔积 (FROM a, b)
    ]

    def __init__(self, database: Optional[str] = None):
        self.database = database or 'default'
        self.settings = get_settings()
        self.db_manager = get_db_manager()
        self.sql_fixer = SQLFixer()
        self._schema_info: Optional[dict] = None  # 缓存 Schema 信息

    async def validate(self, sql: str) -> dict:
        """
        验证 SQL 语句

        Args:
            sql: SQL 语句

        Returns:
            验证结果
        """
        issues = []
        warnings = []

        # 1. 基本格式检查
        sql_clean = sql.strip()
        if not sql_clean:
            return {"valid": False, "issues": ["SQL is empty"]}

        sql_lower = sql_clean.lower()

        # 2. 只允许 SELECT
        if not sql_lower.startswith("select"):
            issues.append("Only SELECT queries are allowed")

        # 3. 危险关键字检查
        for keyword in self.DANGEROUS_KEYWORDS:
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, sql_lower):
                issues.append(f"Dangerous keyword detected: {keyword.upper()}")

        # 4. SQL 注入检查
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, sql_lower):
                issues.append("Potential SQL injection detected")
                break

        # 5. 笛卡尔积检测
        if not self.settings.judge_allow_full_scan:
            for pattern in self.CARTESIAN_PATTERNS:
                if re.search(pattern, sql_lower):
                    # 检查是否有 WHERE 或 ON 条件
                    if "where" not in sql_lower and "on" not in sql_lower:
                        issues.append("Cartesian product detected without join condition")
                    else:
                        warnings.append("Potential Cartesian product - verify join conditions")
                    break

        # 6. 语法警告
        if "select *" in sql_lower:
            warnings.append("Using SELECT * is not recommended")

        if "limit" not in sql_lower:
            warnings.append("No LIMIT clause - may return large result set")

        return {
            "valid": len(issues) == 0,
            "sql": sql,
            "issues": issues,
            "warnings": warnings,
        }

    async def analyze_cost(self, sql: str) -> dict:
        """
        分析 SQL 执行成本并进行熔断检查 (P2 增强版)

        增强功能:
        - 完整的执行计划解析
        - 智能风险评分
        - 多维度成本评估
        - 详细的优化建议

        Args:
            sql: SQL 语句

        Returns:
            成本分析结果 (CostAnalysisResult.to_dict())
        """
        result = await plan_explain(sql, self.database)

        # 获取数据库类型
        connector = self.db_manager.get(self.database)
        db_type = connector.db_type

        if not result.get("success"):
            analysis = CostAnalysisResult(
                analyzed=False,
                sql=sql,
                db_type=db_type,
            )
            return analysis.to_dict()

        # 解析执行计划
        plan = result.get("plan", [])
        summary = result.get("summary", {})

        # 创建分析结果
        analysis = CostAnalysisResult(
            analyzed=True,
            sql=sql,
            db_type=db_type,
            raw_plan=plan,
            scan_type=summary.get("scan_type", "unknown"),
            uses_index=summary.get("uses_index", False),
            tables_accessed=summary.get("tables_accessed", []),
            index_used=summary.get("key", "") or summary.get("using_index", ""),
        )

        # 根据数据库类型进行详细分析
        if db_type == "mysql":
            self._analyze_mysql_cost(summary, analysis)
        elif db_type == "postgresql":
            self._analyze_postgres_cost(summary, analysis)
        elif db_type == "sqlite":
            self._analyze_sqlite_cost(summary, analysis)
        elif db_type == "sqlserver":
            self._analyze_sqlserver_cost(summary, analysis)
        elif db_type == "clickhouse":
            self._analyze_clickhouse_cost(summary, analysis)
        elif db_type == "duckdb":
            self._analyze_duckdb_cost(summary, analysis)

        # 计算综合风险评分
        analysis.risk_score = self._calculate_risk_score(analysis)

        # 基于风险阈值决定是否拒绝
        if analysis.risk_score >= self.settings.judge_risk_threshold and not analysis.should_reject:
            analysis.should_reject = True
            analysis.reject_reason = f"风险评分 ({analysis.risk_score:.2f}) 超过阈值 ({self.settings.judge_risk_threshold})"

        # 全表扫描检测 (通用)
        if not self.settings.judge_allow_full_scan:
            self._check_full_scan_risk(analysis)

        # 生成优化建议
        analysis.recommendations = self._generate_recommendations(analysis)

        # 记录指标
        if analysis.should_reject and HAS_METRICS:
            reason = "cost_exceeded" if "cost" in analysis.reject_reason.lower() else "rows_exceeded"
            if "full" in analysis.reject_reason.lower() or "scan" in analysis.reject_reason.lower():
                reason = "full_scan"
            record_cost_rejection(db_type, reason)

        return analysis.to_dict()

    def _analyze_mysql_cost(self, summary: dict, analysis: CostAnalysisResult):
        """详细分析 MySQL 成本 (P2 增强)"""
        # 提取指标
        analysis.rows_examined = summary.get("rows_examined", 0)
        analysis.estimated_cost = summary.get("estimated_cost", 0) or summary.get("cost", 0)
        analysis.estimated_rows = summary.get("estimated_rows", 0)

        access_type = summary.get("access_type", "")
        extra = summary.get("extra", "")

        # rows_examined 检查
        if analysis.rows_examined > self.settings.judge_mysql_rows_examined_limit:
            analysis.should_reject = True
            analysis.reject_reason = (
                f"rows_examined ({analysis.rows_examined:,}) exceeds limit "
                f"({self.settings.judge_mysql_rows_examined_limit:,})"
            )
            analysis.risks.append(analysis.reject_reason)
        elif analysis.rows_examined > self.settings.judge_mysql_rows_examined_limit * 0.5:
            analysis.risks.append(f"High rows_examined: {analysis.rows_examined:,}")

        # cost 检查
        if analysis.estimated_cost > self.settings.judge_mysql_cost_limit:
            analysis.should_reject = True
            analysis.reject_reason = (
                f"Query cost ({analysis.estimated_cost:,.2f}) exceeds limit "
                f"({self.settings.judge_mysql_cost_limit:,})"
            )
            analysis.risks.append(analysis.reject_reason)
        elif analysis.estimated_cost > self.settings.judge_mysql_cost_limit * 0.5:
            analysis.risks.append(f"High query cost: {analysis.estimated_cost:,.2f}")

        # 访问类型检查
        if access_type == "ALL":
            analysis.scan_type = "table_scan"
            analysis.risks.append("Full table scan detected (type=ALL)")
        elif access_type == "index":
            analysis.scan_type = "index_scan"
            analysis.risks.append("Full index scan detected (type=index)")
        elif access_type in ("range", "ref", "eq_ref", "const"):
            analysis.scan_type = "index_seek"
            analysis.uses_index = True

        # Extra 字段检查
        extra_lower = extra.lower()
        if "using filesort" in extra_lower:
            analysis.risks.append("Using filesort - ORDER BY not optimized")
        if "using temporary" in extra_lower:
            analysis.risks.append("Using temporary table - GROUP BY/DISTINCT not optimized")
        if "using where" not in extra_lower and access_type in ("ALL", "index"):
            analysis.risks.append("No WHERE filtering on full scan")
        if "using join buffer" in extra_lower:
            analysis.risks.append("Using join buffer - missing index on join column")
        if "using index condition" in extra_lower:
            analysis.risks.append("Index condition pushdown - partial index usage")

    def _analyze_postgres_cost(self, summary: dict, analysis: CostAnalysisResult):
        """详细分析 PostgreSQL 成本 (P2 增强)"""
        # 提取指标
        analysis.estimated_cost = summary.get("total_cost", 0)
        analysis.estimated_rows = summary.get("plan_rows", 0)

        node_type = summary.get("node_type", "")
        plan_width = summary.get("plan_width", 0)
        actual_rows = summary.get("actual_rows")
        actual_time = summary.get("actual_time")

        # total_cost 检查
        if analysis.estimated_cost > self.settings.judge_postgres_cost_limit:
            analysis.should_reject = True
            analysis.reject_reason = (
                f"Query cost ({analysis.estimated_cost:,.2f}) exceeds limit "
                f"({self.settings.judge_postgres_cost_limit:,})"
            )
            analysis.risks.append(analysis.reject_reason)
        elif analysis.estimated_cost > self.settings.judge_postgres_cost_limit * 0.5:
            analysis.risks.append(f"High query cost: {analysis.estimated_cost:,.2f}")

        # 节点类型检查
        if node_type == "Seq Scan":
            analysis.scan_type = "table_scan"
            if analysis.estimated_rows > 10000:
                analysis.risks.append(
                    f"Sequential scan on {analysis.estimated_rows:,} rows"
                )
        elif node_type in ("Index Scan", "Index Only Scan"):
            analysis.scan_type = "index_seek"
            analysis.uses_index = True
        elif node_type == "Bitmap Heap Scan":
            analysis.scan_type = "bitmap_scan"
            analysis.uses_index = True
        elif node_type == "Nested Loop":
            if analysis.estimated_rows > 100000:
                analysis.risks.append("Nested loop on large dataset")
        elif node_type == "Hash Join":
            if analysis.estimated_rows > 1000000:
                analysis.risks.append("Large hash join - ensure sufficient work_mem")
        elif node_type == "Sort":
            if analysis.estimated_rows > 100000:
                analysis.risks.append("Sorting large result set")

        # 行宽度检查
        if plan_width > 1000 and analysis.estimated_rows > 10000:
            analysis.risks.append(
                f"Wide rows ({plan_width} bytes) with many rows - consider fewer columns"
            )

        # EXPLAIN ANALYZE 检查
        if actual_rows is not None and analysis.estimated_rows > 0:
            estimate_ratio = actual_rows / analysis.estimated_rows
            if estimate_ratio > 10 or estimate_ratio < 0.1:
                analysis.risks.append(
                    f"Row estimate off by {estimate_ratio:.1f}x - run ANALYZE"
                )

        if actual_time is not None and actual_time > 5000:
            analysis.risks.append(f"Actual execution time {actual_time:.0f}ms is high")

    def _analyze_sqlite_cost(self, summary: dict, analysis: CostAnalysisResult):
        """详细分析 SQLite 成本 (P2 增强)"""
        # 提取指标
        analysis.estimated_rows = summary.get("estimated_rows", 0)
        detail = summary.get("detail", "")
        detail_lower = detail.lower()

        sqlite_limit = 500000

        # 行数检查
        if analysis.estimated_rows > sqlite_limit:
            analysis.should_reject = True
            analysis.reject_reason = (
                f"Estimated rows ({analysis.estimated_rows:,}) exceeds limit ({sqlite_limit:,})"
            )
            analysis.risks.append(analysis.reject_reason)
        elif analysis.estimated_rows > sqlite_limit * 0.5:
            analysis.risks.append(f"High estimated rows: {analysis.estimated_rows:,}")

        # 扫描类型检查
        if "scan table" in detail_lower:
            if "using covering index" in detail_lower:
                analysis.scan_type = "covering_index_scan"
                analysis.uses_index = True
            elif "using index" in detail_lower:
                analysis.scan_type = "index_scan"
                analysis.uses_index = True
            else:
                analysis.scan_type = "table_scan"
                analysis.risks.append("Full table scan detected")
        elif "search table" in detail_lower:
            analysis.scan_type = "index_seek"
            analysis.uses_index = True

        # 临时 B-Tree 检查
        if "temp b-tree" in detail_lower:
            if "order by" in detail_lower:
                analysis.risks.append("Using temp B-tree for ORDER BY")
            elif "group by" in detail_lower:
                analysis.risks.append("Using temp B-tree for GROUP BY")
            elif "distinct" in detail_lower:
                analysis.risks.append("Using temp B-tree for DISTINCT")
            else:
                analysis.risks.append("Using temp B-tree")

        # 子查询检查
        if "subquery" in detail_lower or "correlated" in detail_lower:
            analysis.risks.append("Correlated subquery detected")

        # 自动索引检查
        if "automatic index" in detail_lower or "auto-index" in detail_lower:
            analysis.risks.append("SQLite creating automatic index at runtime")

        # Compound 查询检查
        if "compound" in detail_lower:
            analysis.risks.append("Compound query (UNION/INTERSECT/EXCEPT)")

    def _analyze_sqlserver_cost(self, summary: dict, analysis: CostAnalysisResult):
        """详细分析 SQL Server 成本"""
        # 提取指标
        analysis.estimated_rows = summary.get("estimated_rows", 0)
        analysis.estimated_cost = summary.get("estimated_cost", 0)
        scan_type = summary.get("scan_type", "unknown")

        sqlserver_limit = 1000000

        # 行数检查
        if analysis.estimated_rows > sqlserver_limit:
            analysis.should_reject = True
            analysis.reject_reason = (
                f"Estimated rows ({analysis.estimated_rows:,}) exceeds limit ({sqlserver_limit:,})"
            )
            analysis.risks.append(analysis.reject_reason)
        elif analysis.estimated_rows > sqlserver_limit * 0.5:
            analysis.risks.append(f"High estimated rows: {analysis.estimated_rows:,}")

        # 扫描类型检查
        if scan_type == "table_scan":
            analysis.scan_type = "table_scan"
            analysis.risks.append("Full table scan detected")
        elif scan_type == "index_scan":
            analysis.scan_type = "index_scan"
            analysis.uses_index = True
        elif scan_type == "index_seek":
            analysis.scan_type = "index_seek"
            analysis.uses_index = True

    def _analyze_clickhouse_cost(self, summary: dict, analysis: CostAnalysisResult):
        """详细分析 ClickHouse 成本"""
        # 提取指标
        analysis.estimated_rows = summary.get("estimated_rows", 0)
        scan_type = summary.get("scan_type", "unknown")

        # ClickHouse 是列式存储，通常可以处理大量数据
        clickhouse_limit = 10000000

        # 行数检查
        if analysis.estimated_rows > clickhouse_limit:
            analysis.should_reject = True
            analysis.reject_reason = (
                f"Estimated rows ({analysis.estimated_rows:,}) exceeds limit ({clickhouse_limit:,})"
            )
            analysis.risks.append(analysis.reject_reason)
        elif analysis.estimated_rows > clickhouse_limit * 0.5:
            analysis.risks.append(f"High estimated rows: {analysis.estimated_rows:,}")

        # 扫描类型检查
        if scan_type == "mergetree_read":
            analysis.scan_type = "mergetree_read"
            analysis.uses_index = True
        elif scan_type == "expression":
            analysis.scan_type = "expression"

    def _analyze_duckdb_cost(self, summary: dict, analysis: CostAnalysisResult):
        """详细分析 DuckDB 成本"""
        # 提取指标
        analysis.estimated_rows = summary.get("estimated_rows", 0)
        scan_type = summary.get("scan_type", "unknown")

        # DuckDB 是分析型数据库，可以处理较大数据量
        duckdb_limit = 5000000

        # 行数检查
        if analysis.estimated_rows > duckdb_limit:
            analysis.should_reject = True
            analysis.reject_reason = (
                f"Estimated rows ({analysis.estimated_rows:,}) exceeds limit ({duckdb_limit:,})"
            )
            analysis.risks.append(analysis.reject_reason)
        elif analysis.estimated_rows > duckdb_limit * 0.5:
            analysis.risks.append(f"High estimated rows: {analysis.estimated_rows:,}")

        # 扫描类型检查
        if scan_type == "table_scan":
            analysis.scan_type = "table_scan"
            analysis.risks.append("Full table scan detected")
        elif scan_type == "index_scan":
            analysis.scan_type = "index_scan"
            analysis.uses_index = True
        elif scan_type == "filter":
            analysis.scan_type = "filter"

    def _calculate_risk_score(self, analysis: CostAnalysisResult) -> float:
        """
        计算综合风险评分 (0-1)

        评分因素:
        - 扫描类型
        - 索引使用
        - 估算行数
        - 成本值
        - 风险数量
        """
        score = 0.0

        # 扫描类型评分 (0-0.3)
        scan_scores = {
            "table_scan": 0.3,
            "index_scan": 0.15,
            "bitmap_scan": 0.1,
            "index_seek": 0.05,
            "covering_index_scan": 0.02,
            "pk_lookup": 0.01,
        }
        score += scan_scores.get(analysis.scan_type, 0.2)

        # 索引使用 (0-0.1)
        if not analysis.uses_index:
            score += 0.1

        # 估算行数 (0-0.3)
        if analysis.estimated_rows > 1000000:
            score += 0.3
        elif analysis.estimated_rows > 100000:
            score += 0.2
        elif analysis.estimated_rows > 10000:
            score += 0.1
        elif analysis.estimated_rows > 1000:
            score += 0.05

        # 风险数量 (0-0.3)
        risk_count = len(analysis.risks)
        score += min(risk_count * 0.05, 0.3)

        return min(score, 1.0)

    def _check_full_scan_risk(self, analysis: CostAnalysisResult):
        """检查全表扫描风险"""
        if analysis.scan_type in ("table_scan", "ALL", "Seq Scan"):
            if analysis.estimated_rows > 10000:
                if f"Full table scan on {analysis.estimated_rows:,} rows" not in analysis.risks:
                    analysis.risks.append(f"Full table scan on {analysis.estimated_rows:,} rows")
                if analysis.estimated_rows > 100000 and not analysis.should_reject:
                    analysis.should_reject = True
                    analysis.reject_reason = (
                        f"Full table scan on large table ({analysis.estimated_rows:,} rows)"
                    )

    def _generate_recommendations(self, analysis: CostAnalysisResult) -> list:
        """生成优化建议 (P2 增强)"""
        recommendations = []

        for risk in analysis.risks:
            risk_lower = risk.lower()

            # 全表扫描
            if "full table scan" in risk_lower or "sequential scan" in risk_lower:
                recommendations.append(
                    "Add WHERE clause with indexed columns or create index on filtered columns"
                )

            # 行数过多
            elif "rows_examined" in risk_lower or "estimated rows" in risk_lower:
                recommendations.append(
                    "Add more specific filters, use LIMIT, or optimize query structure"
                )

            # 成本过高
            elif "cost" in risk_lower and "exceeds" in risk_lower:
                recommendations.append(
                    "Optimize query structure, add composite indexes, or reduce result set"
                )

            # 排序问题
            elif "filesort" in risk_lower or ("temp b-tree" in risk_lower and "order" in risk_lower):
                recommendations.append(
                    "Add index on ORDER BY columns to avoid sorting"
                )

            # 临时表问题
            elif "temporary" in risk_lower or ("temp b-tree" in risk_lower and "group" in risk_lower):
                recommendations.append(
                    "Add covering index for GROUP BY columns"
                )

            # 索引扫描
            elif "index scan" in risk_lower and "full" in risk_lower:
                recommendations.append(
                    "Add more selective WHERE conditions to use index seek"
                )

            # 嵌套循环
            elif "nested loop" in risk_lower:
                recommendations.append(
                    "Consider using hash join or add index on join columns"
                )

            # Hash Join
            elif "hash join" in risk_lower:
                recommendations.append(
                    "Ensure sufficient work_mem (PostgreSQL) or join_buffer_size (MySQL)"
                )

            # 统计信息
            elif "estimate off" in risk_lower or "analyze" in risk_lower:
                recommendations.append(
                    "Run ANALYZE on affected tables to update statistics"
                )

            # 子查询
            elif "subquery" in risk_lower or "correlated" in risk_lower:
                recommendations.append(
                    "Rewrite correlated subquery as JOIN for better performance"
                )

            # 自动索引
            elif "automatic index" in risk_lower:
                recommendations.append(
                    "Create permanent index to avoid runtime index creation overhead"
                )

            # Join buffer
            elif "join buffer" in risk_lower:
                recommendations.append(
                    "Add index on join column to avoid join buffer usage"
                )

            # 宽行
            elif "wide rows" in risk_lower:
                recommendations.append(
                    "Select only needed columns instead of SELECT *"
                )

        # 去重
        return list(dict.fromkeys(recommendations))

    async def judge(
        self,
        sql: str,
        schema: str = "",
        available_columns: list[str] = None,
        available_tables: list[str] = None,
    ) -> dict:
        """
        完整的 SQL 审判流程

        Args:
            sql: SQL 语句
            schema: Schema 信息 (用于生成修复建议)
            available_columns: 可用列名列表
            available_tables: 可用表名列表

        Returns:
            审判结果
        """
        # 1. 验证
        validation = await self.validate(sql)
        if not validation["valid"]:
            # 生成修复建议
            fix_suggestions = self.sql_fixer.generate_suggestions(
                sql=sql,
                error="; ".join(validation["issues"]),
                schema=schema,
                available_columns=available_columns,
                available_tables=available_tables,
            )

            return {
                "approved": False,
                "sql": sql,
                "reason": "Validation failed: " + "; ".join(validation["issues"]),
                "details": validation,
                "fix_suggestions": [
                    {
                        "type": s.type,
                        "description": s.description,
                        "original": s.original,
                        "suggested": s.suggested,
                        "confidence": s.confidence,
                        "auto_fixable": s.auto_fixable,
                    }
                    for s in fix_suggestions
                ],
            }

        # 2. 成本分析
        cost_analysis = await self.analyze_cost(sql)

        # 3. 成本熔断检查
        if cost_analysis.get("should_reject"):
            # 从 recommendations 列表中获取建议
            recommendations = cost_analysis.get("recommendations", [])
            recommendation_text = recommendations[0] if recommendations else "优化查询"

            return {
                "approved": False,
                "sql": sql,
                "reason": f"Cost circuit breaker triggered: {cost_analysis.get('reject_reason')}",
                "validation": validation,
                "cost_analysis": cost_analysis,
                "fix_suggestions": [{
                    "type": "optimization",
                    "description": recommendation_text,
                    "original": sql,
                    "suggested": sql,
                    "confidence": 0.5,
                    "auto_fixable": False,
                }],
            }

        # 4. 最终决定
        approved = True
        reason = "Query approved"

        if cost_analysis.get("risks"):
            reason = f"Query approved with warnings: {', '.join(cost_analysis['risks'])}"

        return {
            "approved": approved,
            "sql": sql,
            "reason": reason,
            "validation": validation,
            "cost_analysis": cost_analysis,
            "fix_suggestions": [],
        }

    async def judge_with_auto_fix(
        self,
        sql: str,
        error: str,
        schema: str = "",
        available_columns: list[str] = None,
        available_tables: list[str] = None,
    ) -> dict:
        """
        审判并尝试自动修复

        Args:
            sql: SQL 语句
            error: 错误信息
            schema: Schema 信息
            available_columns: 可用列名列表
            available_tables: 可用表名列表

        Returns:
            包含修复建议和自动修复结果的字典
        """
        # 生成修复建议
        fix_suggestions = self.sql_fixer.generate_suggestions(
            sql=sql,
            error=error,
            schema=schema,
            available_columns=available_columns,
            available_tables=available_tables,
        )

        # 尝试自动修复
        auto_fixed_sql = self.sql_fixer.auto_fix(
            sql=sql,
            error=error,
            schema=schema,
            available_columns=available_columns,
            available_tables=available_tables,
        )

        return {
            "original_sql": sql,
            "error": error,
            "fix_suggestions": [
                {
                    "type": s.type,
                    "description": s.description,
                    "original": s.original,
                    "suggested": s.suggested,
                    "confidence": s.confidence,
                    "auto_fixable": s.auto_fixable,
                }
                for s in fix_suggestions
            ],
            "auto_fixed_sql": auto_fixed_sql,
            "auto_fix_available": auto_fixed_sql is not None,
        }

    async def run(self, state: DataPilotState) -> dict[str, Any]:
        """
        执行 Judge Agent

        Args:
            state: 当前工作流状态

        Returns:
            状态更新
        """
        candidates = state.get("candidates", [])
        if not candidates:
            return {
                "current_agent": "judge",
                "next_agent": "logic_architect",
                "error_context": "No SQL candidates to judge",
            }

        # 获取最新的 SQL 候选
        latest_candidate = candidates[-1]
        sql = latest_candidate.get("sql", "")

        # 执行审判
        result = await self.judge(sql)

        if result["approved"]:
            # 审判通过，执行 SQL
            connector = self.db_manager.get(self.database)
            try:
                exec_result = await connector.execute_query(sql)
                return {
                    "current_agent": "judge",
                    "next_agent": "viz_expert",
                    "winner_sql": sql,
                    "execution_result": {
                        "success": True,
                        "data": exec_result,
                        "row_count": len(exec_result),
                        "columns": list(exec_result[0].keys()) if exec_result else [],
                        "execution_time_ms": 0,
                        "error": None,
                    },
                    "execution_plan": str(result.get("cost_analysis", {}).get("plan", [])),
                }
            except Exception as e:
                # 执行失败，返回错误
                retries = state.get("retries", {})
                judge_retries = retries.get("judge", 0) + 1

                if judge_retries >= self.settings.judge_max_retries:
                    return {
                        "current_agent": "judge",
                        "next_agent": None,
                        "human_handoff": True,
                        "error_context": f"SQL execution failed after {judge_retries} retries: {str(e)}",
                        "last_error": str(e),
                    }

                return {
                    "current_agent": "judge",
                    "next_agent": "logic_architect",
                    "error_context": f"SQL execution error: {str(e)}",
                    "last_error": str(e),
                    "retries": {**retries, "judge": judge_retries},
                }
        else:
            # 审判不通过
            retries = state.get("retries", {})
            architect_retries = retries.get("architect", 0) + 1

            if architect_retries >= self.settings.judge_max_retries:
                return {
                    "current_agent": "judge",
                    "next_agent": None,
                    "human_handoff": True,
                    "error_context": f"SQL rejected after {architect_retries} attempts: {result['reason']}",
                    "last_error": result["reason"],
                }

            return {
                "current_agent": "judge",
                "next_agent": "logic_architect",
                "error_context": result["reason"],
                "last_error": result["reason"],
                "retries": {**retries, "architect": architect_retries},
            }


# LangGraph 节点函数
async def judge_node(state: DataPilotState) -> dict[str, Any]:
    """LangGraph 节点：Judge"""
    database = state.get("database", "default")
    judge = Judge(database)
    return await judge.run(state)


__all__ = ["Judge", "judge_node", "CostAnalysisResult"]
