# -*- coding: utf-8 -*-
"""
多 Judge 博弈机制

实现多个 Judge 的投票、共识和冲突解决

功能:
1. 多 Judge 并行评估
2. 投票和共识机制
3. 冲突解决策略
4. LLM 驱动的语义评估
5. 评估结果聚合
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json

from prometheus_client import Counter, Histogram

from ..llm.deepseek import get_deepseek_client
from ..db.connector import get_db_manager
from ..config.settings import get_settings


# ============================================
# Prometheus 指标
# ============================================

JUDGE_VOTES = Counter(
    "datapilot_judge_votes_total",
    "Total judge votes",
    labelnames=["judge_type", "verdict"],  # approve, reject
)

JUDGE_CONFLICTS = Counter(
    "datapilot_judge_conflicts_total",
    "Total judge conflicts requiring resolution",
)

JUDGE_LATENCY = Histogram(
    "datapilot_judge_latency_seconds",
    "Judge evaluation latency",
    labelnames=["judge_type"],
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)


# ============================================
# 数据结构
# ============================================

class Verdict(Enum):
    """裁决结果"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"  # 弃权


class ConflictResolution(Enum):
    """冲突解决策略"""
    MAJORITY = "majority"  # 多数决
    UNANIMOUS = "unanimous"  # 全票通过
    WEIGHTED = "weighted"  # 加权投票
    LLM_ARBITER = "llm_arbiter"  # LLM 仲裁


@dataclass
class JudgeVote:
    """Judge 投票"""
    judge_id: str
    judge_type: str
    verdict: Verdict
    confidence: float  # 0-1
    reason: str
    details: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class JudgmentResult:
    """最终裁决结果"""
    approved: bool
    sql: str
    votes: list[JudgeVote]
    consensus_method: str
    final_confidence: float
    reason: str
    conflicts: list[str] = field(default_factory=list)
    arbiter_decision: Optional[dict] = None


# ============================================
# Judge 基类
# ============================================

class BaseJudge(ABC):
    """Judge 基类"""

    def __init__(self, judge_id: str):
        self.judge_id = judge_id
        self.settings = get_settings()

    @property
    @abstractmethod
    def judge_type(self) -> str:
        """Judge 类型"""
        pass

    @property
    def weight(self) -> float:
        """投票权重 (用于加权投票)"""
        return 1.0

    @abstractmethod
    async def evaluate(
        self,
        sql: str,
        schema: str = "",
        context: Optional[dict] = None,
    ) -> JudgeVote:
        """
        评估 SQL

        Args:
            sql: SQL 语句
            schema: Schema 信息
            context: 额外上下文

        Returns:
            投票结果
        """
        pass


# ============================================
# 规则 Judge (基于规则的评估)
# ============================================

class RuleBasedJudge(BaseJudge):
    """
    规则 Judge

    基于预定义规则进行评估
    """

    DANGEROUS_KEYWORDS = [
        "drop", "truncate", "delete", "update", "insert",
        "alter", "create", "grant", "revoke",
    ]

    @property
    def judge_type(self) -> str:
        return "rule_based"

    @property
    def weight(self) -> float:
        return self.settings.multi_judge_rule_weight

    async def evaluate(
        self,
        sql: str,
        schema: str = "",
        context: Optional[dict] = None,
    ) -> JudgeVote:
        """基于规则评估"""
        import re

        issues = []
        sql_lower = sql.lower().strip()

        # 检查是否为 SELECT
        if not sql_lower.startswith("select"):
            issues.append("Only SELECT queries allowed")

        # 检查危险关键字
        for keyword in self.DANGEROUS_KEYWORDS:
            if re.search(rf"\b{keyword}\b", sql_lower):
                issues.append(f"Dangerous keyword: {keyword}")

        # 检查 SQL 注入模式
        injection_patterns = [
            r";\s*--", r";\s*drop", r"union\s+select",
            r"or\s+1\s*=\s*1", r"'\s*or\s*'",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, sql_lower):
                issues.append("Potential SQL injection")
                break

        if issues:
            verdict = Verdict.REJECT
            confidence = 0.95
            reason = "; ".join(issues)
        else:
            verdict = Verdict.APPROVE
            confidence = 0.8
            reason = "Passed rule-based checks"

        JUDGE_VOTES.labels(judge_type=self.judge_type, verdict=verdict.value).inc()

        return JudgeVote(
            judge_id=self.judge_id,
            judge_type=self.judge_type,
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            details={"issues": issues},
        )


# ============================================
# 成本 Judge (基于执行计划)
# ============================================

class CostJudge(BaseJudge):
    """
    成本 Judge

    基于执行计划和成本分析进行评估
    """

    def __init__(self, judge_id: str, database: str = "default"):
        super().__init__(judge_id)
        self.database = database
        self.db_manager = get_db_manager()

    @property
    def judge_type(self) -> str:
        return "cost_based"

    @property
    def weight(self) -> float:
        return self.settings.multi_judge_cost_weight

    async def evaluate(
        self,
        sql: str,
        schema: str = "",
        context: Optional[dict] = None,
    ) -> JudgeVote:
        """基于成本评估"""
        from ..mcp.tools import plan_explain

        risks = []

        try:
            result = await plan_explain(sql, self.database)

            if not result.get("success"):
                return JudgeVote(
                    judge_id=self.judge_id,
                    judge_type=self.judge_type,
                    verdict=Verdict.ABSTAIN,
                    confidence=0.3,
                    reason=f"Could not analyze: {result.get('error')}",
                )

            summary = result.get("summary", {})

            # 检查全表扫描
            scan_type = summary.get("scan_type", "")
            if scan_type in ["table_scan", "ALL", "Seq Scan"]:
                estimated_rows = summary.get("estimated_rows", 0)
                if estimated_rows > 100000:
                    risks.append(f"Full table scan on {estimated_rows:,} rows")

            # 检查索引使用
            if not summary.get("uses_index", True):
                risks.append("No index used")

            if risks:
                verdict = Verdict.REJECT
                confidence = 0.85
                reason = "; ".join(risks)
            else:
                verdict = Verdict.APPROVE
                confidence = 0.9
                reason = "Cost analysis passed"

        except Exception as e:
            verdict = Verdict.ABSTAIN
            confidence = 0.3
            reason = f"Analysis error: {str(e)}"

        JUDGE_VOTES.labels(judge_type=self.judge_type, verdict=verdict.value).inc()

        return JudgeVote(
            judge_id=self.judge_id,
            judge_type=self.judge_type,
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            details={"risks": risks},
        )


# ============================================
# 语义 Judge (LLM 驱动)
# ============================================

SEMANTIC_JUDGE_PROMPT = """你是一个专业的 SQL 审核专家。请评估以下 SQL 查询的质量和安全性。

## 数据库 Schema
{schema}

## SQL 查询
```sql
{sql}
```

## 用户原始问题
{query}

## 评估维度
1. **语义正确性**: SQL 是否正确回答了用户问题
2. **安全性**: 是否存在安全风险
3. **性能**: 是否有明显的性能问题
4. **最佳实践**: 是否符合 SQL 最佳实践

## 输出格式
返回 JSON:
```json
{{
    "verdict": "approve" 或 "reject",
    "confidence": 0.0-1.0,
    "reason": "评估理由",
    "semantic_correct": true/false,
    "security_issues": ["问题列表"],
    "performance_issues": ["问题列表"],
    "suggestions": ["改进建议"]
}}
```
"""


class SemanticJudge(BaseJudge):
    """
    语义 Judge

    使用 LLM 进行语义级别的评估
    """

    def __init__(self, judge_id: str):
        super().__init__(judge_id)
        self.llm = get_deepseek_client()

    @property
    def judge_type(self) -> str:
        return "semantic"

    @property
    def weight(self) -> float:
        return self.settings.multi_judge_semantic_weight

    async def evaluate(
        self,
        sql: str,
        schema: str = "",
        context: Optional[dict] = None,
    ) -> JudgeVote:
        """基于 LLM 语义评估"""
        import re

        query = context.get("query", "") if context else ""

        prompt = SEMANTIC_JUDGE_PROMPT.format(
            schema=schema or "未提供",
            sql=sql,
            query=query or "未提供",
        )

        try:
            response = await self.llm.chat([
                {"role": "system", "content": "你是一个 SQL 审核专家。"},
                {"role": "user", "content": prompt},
            ])

            # 解析 JSON 响应
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())

                verdict_str = result.get("verdict", "reject").lower()
                verdict = Verdict.APPROVE if verdict_str == "approve" else Verdict.REJECT

                confidence = float(result.get("confidence", 0.7))
                reason = result.get("reason", "LLM evaluation")

                details = {
                    "semantic_correct": result.get("semantic_correct", False),
                    "security_issues": result.get("security_issues", []),
                    "performance_issues": result.get("performance_issues", []),
                    "suggestions": result.get("suggestions", []),
                }
            else:
                verdict = Verdict.ABSTAIN
                confidence = 0.3
                reason = "Could not parse LLM response"
                details = {}

        except Exception as e:
            verdict = Verdict.ABSTAIN
            confidence = 0.3
            reason = f"LLM evaluation error: {str(e)}"
            details = {}

        JUDGE_VOTES.labels(judge_type=self.judge_type, verdict=verdict.value).inc()

        return JudgeVote(
            judge_id=self.judge_id,
            judge_type=self.judge_type,
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            details=details,
        )


# ============================================
# LLM 仲裁者
# ============================================

ARBITER_PROMPT = """你是一个公正的仲裁者。多个 Judge 对同一个 SQL 查询产生了分歧，请做出最终裁决。

## SQL 查询
```sql
{sql}
```

## 各 Judge 的投票
{votes_summary}

## 冲突点
{conflicts}

## 任务
分析各 Judge 的理由，做出最终裁决。

## 输出格式
返回 JSON:
```json
{{
    "final_verdict": "approve" 或 "reject",
    "confidence": 0.0-1.0,
    "reasoning": "裁决理由",
    "key_factors": ["关键考虑因素"]
}}
```
"""


class LLMArbiter:
    """LLM 仲裁者"""

    def __init__(self):
        self.llm = get_deepseek_client()

    async def arbitrate(
        self,
        sql: str,
        votes: list[JudgeVote],
        conflicts: list[str],
    ) -> dict:
        """
        仲裁冲突

        Args:
            sql: SQL 语句
            votes: 各 Judge 的投票
            conflicts: 冲突点

        Returns:
            仲裁结果
        """
        import re

        # 构建投票摘要
        votes_summary = "\n".join([
            f"- {v.judge_type} Judge: {v.verdict.value} (置信度: {v.confidence:.2f})\n  理由: {v.reason}"
            for v in votes
        ])

        prompt = ARBITER_PROMPT.format(
            sql=sql,
            votes_summary=votes_summary,
            conflicts="\n".join(f"- {c}" for c in conflicts) if conflicts else "无明显冲突",
        )

        try:
            response = await self.llm.chat([
                {"role": "system", "content": "你是一个公正的仲裁者。"},
                {"role": "user", "content": prompt},
            ])

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "verdict": result.get("final_verdict", "reject"),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                    "key_factors": result.get("key_factors", []),
                }

        except Exception as e:
            pass

        # 默认拒绝
        return {
            "verdict": "reject",
            "confidence": 0.5,
            "reasoning": "Arbiter could not reach decision",
            "key_factors": [],
        }


# ============================================
# 多 Judge 协调器
# ============================================

class MultiJudgeCoordinator:
    """
    多 Judge 协调器

    协调多个 Judge 的评估和投票
    """

    def __init__(
        self,
        judges: Optional[list[BaseJudge]] = None,
        resolution: Optional[ConflictResolution] = None,
        database: str = "default",
    ):
        # 从 settings 读取配置
        self.settings = get_settings()
        self.judges = judges or self._create_default_judges(database)
        # 使用 settings 中的 resolution 配置，如果未指定
        if resolution is None:
            resolution_str = self.settings.multi_judge_resolution
            self.resolution = ConflictResolution(resolution_str)
        else:
            self.resolution = resolution
        self.arbiter = LLMArbiter()

    def _create_default_judges(self, database: str) -> list[BaseJudge]:
        """创建默认 Judge 集合"""
        return [
            RuleBasedJudge("rule_judge_1"),
            CostJudge("cost_judge_1", database),
            SemanticJudge("semantic_judge_1"),
        ]

    async def evaluate(
        self,
        sql: str,
        schema: str = "",
        context: Optional[dict] = None,
    ) -> JudgmentResult:
        """
        多 Judge 评估

        Args:
            sql: SQL 语句
            schema: Schema 信息
            context: 额外上下文

        Returns:
            最终裁决结果
        """
        # 并行执行所有 Judge 评估
        tasks = [
            judge.evaluate(sql, schema, context)
            for judge in self.judges
        ]

        votes = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常
        valid_votes = [
            v for v in votes
            if isinstance(v, JudgeVote)
        ]

        if not valid_votes:
            return JudgmentResult(
                approved=False,
                sql=sql,
                votes=[],
                consensus_method="none",
                final_confidence=0,
                reason="No valid votes received",
            )

        # 根据策略达成共识
        return await self._reach_consensus(sql, valid_votes)

    async def _reach_consensus(
        self,
        sql: str,
        votes: list[JudgeVote],
    ) -> JudgmentResult:
        """达成共识"""
        # 统计投票
        approve_votes = [v for v in votes if v.verdict == Verdict.APPROVE]
        reject_votes = [v for v in votes if v.verdict == Verdict.REJECT]
        abstain_votes = [v for v in votes if v.verdict == Verdict.ABSTAIN]

        # 检测冲突
        conflicts = []
        if approve_votes and reject_votes:
            conflicts.append(
                f"Approve ({len(approve_votes)}) vs Reject ({len(reject_votes)})"
            )
            JUDGE_CONFLICTS.inc()

        # 根据策略决定
        if self.resolution == ConflictResolution.MAJORITY:
            approved, confidence, reason = self._majority_vote(
                approve_votes, reject_votes, abstain_votes
            )
            method = "majority"

        elif self.resolution == ConflictResolution.UNANIMOUS:
            approved, confidence, reason = self._unanimous_vote(
                approve_votes, reject_votes, votes
            )
            method = "unanimous"

        elif self.resolution == ConflictResolution.WEIGHTED:
            approved, confidence, reason = self._weighted_vote(
                votes, approve_votes, reject_votes
            )
            method = "weighted"

        elif self.resolution == ConflictResolution.LLM_ARBITER:
            if conflicts:
                arbiter_result = await self.arbiter.arbitrate(sql, votes, conflicts)
                approved = arbiter_result["verdict"] == "approve"
                confidence = arbiter_result["confidence"]
                reason = arbiter_result["reasoning"]
                method = "llm_arbiter"
            else:
                approved, confidence, reason = self._majority_vote(
                    approve_votes, reject_votes, abstain_votes
                )
                method = "majority_no_conflict"
                arbiter_result = None
        else:
            approved, confidence, reason = self._majority_vote(
                approve_votes, reject_votes, abstain_votes
            )
            method = "default"

        return JudgmentResult(
            approved=approved,
            sql=sql,
            votes=votes,
            consensus_method=method,
            final_confidence=confidence,
            reason=reason,
            conflicts=conflicts,
            arbiter_decision=arbiter_result if self.resolution == ConflictResolution.LLM_ARBITER and conflicts else None,
        )

    def _majority_vote(
        self,
        approve_votes: list[JudgeVote],
        reject_votes: list[JudgeVote],
        abstain_votes: list[JudgeVote],
    ) -> tuple[bool, float, str]:
        """多数决"""
        total_votes = len(approve_votes) + len(reject_votes)
        if total_votes == 0:
            return False, 0.3, "No decisive votes"

        if len(approve_votes) > len(reject_votes):
            avg_confidence = sum(v.confidence for v in approve_votes) / len(approve_votes)
            return True, avg_confidence, f"Majority approved ({len(approve_votes)}/{total_votes})"
        else:
            avg_confidence = sum(v.confidence for v in reject_votes) / len(reject_votes)
            return False, avg_confidence, f"Majority rejected ({len(reject_votes)}/{total_votes})"

    def _unanimous_vote(
        self,
        approve_votes: list[JudgeVote],
        reject_votes: list[JudgeVote],
        all_votes: list[JudgeVote],
    ) -> tuple[bool, float, str]:
        """全票通过"""
        if not reject_votes and approve_votes:
            avg_confidence = sum(v.confidence for v in approve_votes) / len(approve_votes)
            return True, avg_confidence, "Unanimous approval"
        else:
            # 任何拒绝都导致最终拒绝
            if reject_votes:
                avg_confidence = sum(v.confidence for v in reject_votes) / len(reject_votes)
                return False, avg_confidence, f"Rejected by {len(reject_votes)} judge(s)"
            return False, 0.5, "No unanimous approval"

    def _weighted_vote(
        self,
        all_votes: list[JudgeVote],
        approve_votes: list[JudgeVote],
        reject_votes: list[JudgeVote],
    ) -> tuple[bool, float, str]:
        """加权投票"""
        # 获取每个 Judge 的权重
        judge_weights = {j.judge_id: j.weight for j in self.judges}

        approve_weight = sum(
            judge_weights.get(v.judge_id, 1.0) * v.confidence
            for v in approve_votes
        )
        reject_weight = sum(
            judge_weights.get(v.judge_id, 1.0) * v.confidence
            for v in reject_votes
        )

        total_weight = approve_weight + reject_weight
        if total_weight == 0:
            return False, 0.3, "No weighted votes"

        if approve_weight > reject_weight:
            confidence = approve_weight / total_weight
            return True, confidence, f"Weighted approval ({approve_weight:.2f} vs {reject_weight:.2f})"
        else:
            confidence = reject_weight / total_weight
            return False, confidence, f"Weighted rejection ({reject_weight:.2f} vs {approve_weight:.2f})"


# ============================================
# 全局实例
# ============================================

_coordinator: Optional[MultiJudgeCoordinator] = None


def get_multi_judge_coordinator(
    database: str = "default",
    resolution: ConflictResolution = ConflictResolution.WEIGHTED,
) -> MultiJudgeCoordinator:
    """获取多 Judge 协调器"""
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiJudgeCoordinator(
            database=database,
            resolution=resolution,
        )
    return _coordinator


# ============================================
# 导出
# ============================================

__all__ = [
    "BaseJudge",
    "RuleBasedJudge",
    "CostJudge",
    "SemanticJudge",
    "LLMArbiter",
    "MultiJudgeCoordinator",
    "JudgeVote",
    "JudgmentResult",
    "Verdict",
    "ConflictResolution",
    "get_multi_judge_coordinator",
    "JUDGE_VOTES",
    "JUDGE_CONFLICTS",
    "JUDGE_LATENCY",
]
