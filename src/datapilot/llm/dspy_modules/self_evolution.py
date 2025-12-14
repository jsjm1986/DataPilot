# -*- coding: utf-8 -*-
"""
DSPy 自进化模块

实现自动收集训练数据、触发优化、管理模块版本的完整自进化流程

功能:
1. 自动收集成功查询作为训练数据
2. 定期/手动触发 DSPy 优化
3. 模块版本管理和回滚
4. A/B 测试新旧模块
5. 性能指标追踪
"""

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional
from enum import Enum
import threading
import hashlib

from prometheus_client import Counter, Gauge, Histogram

# DSPy 可选导入
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False


# ============================================
# Prometheus 指标
# ============================================

EVOLUTION_RUNS = Counter(
    "datapilot_dspy_evolution_runs_total",
    "Total DSPy evolution runs",
    labelnames=["status"],  # success, failed, skipped
)

TRAINING_SAMPLES = Gauge(
    "datapilot_dspy_training_samples",
    "Number of training samples collected",
)

MODULE_VERSION = Gauge(
    "datapilot_dspy_module_version",
    "Current active module version",
    labelnames=["module_type"],
)

EVOLUTION_ACCURACY = Gauge(
    "datapilot_dspy_evolution_accuracy",
    "Accuracy of the latest evolution",
    labelnames=["module_type"],
)


# ============================================
# 数据结构
# ============================================

class EvolutionStatus(Enum):
    """进化状态"""
    IDLE = "idle"
    COLLECTING = "collecting"
    OPTIMIZING = "optimizing"
    EVALUATING = "evaluating"
    DEPLOYING = "deploying"
    FAILED = "failed"


@dataclass
class TrainingSample:
    """训练样本"""
    id: str
    question: str
    schema: str
    dialect: str
    gold_sql: str
    execution_result: Optional[list] = None
    created_at: str = ""
    source: str = "user_query"  # user_query, manual, synthetic
    verified: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.id:
            self.id = hashlib.md5(
                f"{self.question}:{self.gold_sql}".encode()
            ).hexdigest()[:12]


@dataclass
class ModuleVersion:
    """模块版本"""
    version: int
    module_type: str  # text2sql, decompose, refine
    path: str
    accuracy: float
    created_at: str
    optimizer: str
    train_samples: int
    is_active: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class EvolutionConfig:
    """进化配置"""
    # 数据收集
    min_samples_for_evolution: int = 50  # 最少样本数
    max_samples: int = 1000  # 最大样本数
    sample_retention_days: int = 90  # 样本保留天数

    # 优化触发
    auto_evolution_enabled: bool = True
    evolution_interval_hours: int = 24  # 自动进化间隔
    accuracy_threshold: float = 0.7  # 部署阈值

    # 优化参数
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    train_split: float = 0.8

    # 版本管理
    max_versions: int = 10  # 最大保留版本数
    rollback_on_degradation: bool = True
    degradation_threshold: float = 0.05  # 性能下降阈值


@dataclass
class EvolutionResult:
    """进化结果"""
    success: bool
    module_type: str
    old_version: int
    new_version: int
    old_accuracy: float
    new_accuracy: float
    train_samples: int
    eval_samples: int
    duration_seconds: float
    error: Optional[str] = None
    deployed: bool = False


# ============================================
# 训练数据收集器
# ============================================

class TrainingDataCollector:
    """
    训练数据收集器

    自动收集成功的查询作为训练数据
    """

    def __init__(
        self,
        data_dir: str = "data/dspy_training",
        config: Optional[EvolutionConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or EvolutionConfig()
        self._samples: list[TrainingSample] = []
        self._lock = threading.Lock()

        # 加载已有数据
        self._load_samples()

    def _load_samples(self):
        """加载已有样本"""
        samples_file = self.data_dir / "samples.json"
        if samples_file.exists():
            try:
                with open(samples_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._samples = [TrainingSample(**s) for s in data]
            except Exception as e:
                print(f"Failed to load training samples: {e}")
                self._samples = []

        TRAINING_SAMPLES.set(len(self._samples))

    def _save_samples(self):
        """保存样本"""
        samples_file = self.data_dir / "samples.json"
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in self._samples], f, ensure_ascii=False, indent=2)

    def add_sample(
        self,
        question: str,
        schema: str,
        gold_sql: str,
        dialect: str = "sqlite",
        execution_result: Optional[list] = None,
        source: str = "user_query",
        verified: bool = False,
    ) -> TrainingSample:
        """
        添加训练样本

        Args:
            question: 用户问题
            schema: 数据库 Schema
            gold_sql: 正确的 SQL
            dialect: SQL 方言
            execution_result: 执行结果
            source: 数据来源
            verified: 是否已验证

        Returns:
            添加的样本
        """
        sample = TrainingSample(
            id="",
            question=question,
            schema=schema,
            dialect=dialect,
            gold_sql=gold_sql,
            execution_result=execution_result,
            source=source,
            verified=verified,
        )

        with self._lock:
            # 检查重复
            existing_ids = {s.id for s in self._samples}
            if sample.id in existing_ids:
                return sample

            # 添加样本
            self._samples.append(sample)

            # 限制最大样本数
            if len(self._samples) > self.config.max_samples:
                # 移除最旧的未验证样本
                unverified = [s for s in self._samples if not s.verified]
                if unverified:
                    unverified.sort(key=lambda x: x.created_at)
                    self._samples.remove(unverified[0])

            # 保存
            self._save_samples()
            TRAINING_SAMPLES.set(len(self._samples))

        return sample

    def get_samples(
        self,
        verified_only: bool = False,
        min_date: Optional[datetime] = None,
    ) -> list[TrainingSample]:
        """获取样本"""
        samples = self._samples.copy()

        if verified_only:
            samples = [s for s in samples if s.verified]

        if min_date:
            min_date_str = min_date.isoformat()
            samples = [s for s in samples if s.created_at >= min_date_str]

        return samples

    def verify_sample(self, sample_id: str, verified: bool = True):
        """验证样本"""
        with self._lock:
            for sample in self._samples:
                if sample.id == sample_id:
                    sample.verified = verified
                    self._save_samples()
                    break

    def delete_sample(self, sample_id: str):
        """删除样本"""
        with self._lock:
            self._samples = [s for s in self._samples if s.id != sample_id]
            self._save_samples()
            TRAINING_SAMPLES.set(len(self._samples))

    def cleanup_old_samples(self):
        """清理过期样本"""
        cutoff = datetime.utcnow() - timedelta(days=self.config.sample_retention_days)
        cutoff_str = cutoff.isoformat()

        with self._lock:
            # 保留已验证的样本
            self._samples = [
                s for s in self._samples
                if s.verified or s.created_at >= cutoff_str
            ]
            self._save_samples()
            TRAINING_SAMPLES.set(len(self._samples))

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = len(self._samples)
        verified = sum(1 for s in self._samples if s.verified)

        sources = {}
        for s in self._samples:
            sources[s.source] = sources.get(s.source, 0) + 1

        return {
            "total_samples": total,
            "verified_samples": verified,
            "unverified_samples": total - verified,
            "sources": sources,
            "data_dir": str(self.data_dir),
        }


# ============================================
# 模块版本管理器
# ============================================

class ModuleVersionManager:
    """
    模块版本管理器

    管理优化后的模块版本
    """

    def __init__(
        self,
        modules_dir: str = "data/dspy_optimized",
        config: Optional[EvolutionConfig] = None,
    ):
        self.modules_dir = Path(modules_dir)
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or EvolutionConfig()
        self._versions: dict[str, list[ModuleVersion]] = {}
        self._lock = threading.Lock()

        # 加载版本信息
        self._load_versions()

    def _load_versions(self):
        """加载版本信息"""
        versions_file = self.modules_dir / "versions.json"
        if versions_file.exists():
            try:
                with open(versions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for module_type, versions in data.items():
                    self._versions[module_type] = [
                        ModuleVersion(**v) for v in versions
                    ]
            except Exception as e:
                print(f"Failed to load module versions: {e}")

    def _save_versions(self):
        """保存版本信息"""
        versions_file = self.modules_dir / "versions.json"
        data = {
            module_type: [asdict(v) for v in versions]
            for module_type, versions in self._versions.items()
        }
        with open(versions_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_version(
        self,
        module_type: str,
        module_path: str,
        accuracy: float,
        optimizer: str,
        train_samples: int,
        metadata: Optional[dict] = None,
    ) -> ModuleVersion:
        """添加新版本"""
        with self._lock:
            if module_type not in self._versions:
                self._versions[module_type] = []

            # 计算新版本号
            existing = self._versions[module_type]
            new_version = max([v.version for v in existing], default=0) + 1

            # 创建版本记录
            version = ModuleVersion(
                version=new_version,
                module_type=module_type,
                path=module_path,
                accuracy=accuracy,
                created_at=datetime.utcnow().isoformat(),
                optimizer=optimizer,
                train_samples=train_samples,
                is_active=False,
                metadata=metadata or {},
            )

            self._versions[module_type].append(version)

            # 清理旧版本
            self._cleanup_old_versions(module_type)

            self._save_versions()

            return version

    def _cleanup_old_versions(self, module_type: str):
        """清理旧版本"""
        versions = self._versions.get(module_type, [])
        if len(versions) <= self.config.max_versions:
            return

        # 保留活跃版本和最新版本
        to_keep = []
        to_delete = []

        for v in sorted(versions, key=lambda x: x.version, reverse=True):
            if v.is_active or len(to_keep) < self.config.max_versions:
                to_keep.append(v)
            else:
                to_delete.append(v)

        # 删除旧版本文件
        for v in to_delete:
            try:
                if os.path.exists(v.path):
                    os.remove(v.path)
            except Exception:
                pass

        self._versions[module_type] = to_keep

    def activate_version(self, module_type: str, version: int) -> bool:
        """激活指定版本"""
        with self._lock:
            versions = self._versions.get(module_type, [])

            for v in versions:
                if v.version == version:
                    # 取消其他版本的激活状态
                    for other in versions:
                        other.is_active = False
                    v.is_active = True
                    self._save_versions()
                    MODULE_VERSION.labels(module_type=module_type).set(version)
                    return True

            return False

    def get_active_version(self, module_type: str) -> Optional[ModuleVersion]:
        """获取活跃版本"""
        versions = self._versions.get(module_type, [])
        for v in versions:
            if v.is_active:
                return v
        return None

    def get_latest_version(self, module_type: str) -> Optional[ModuleVersion]:
        """获取最新版本"""
        versions = self._versions.get(module_type, [])
        if not versions:
            return None
        return max(versions, key=lambda x: x.version)

    def rollback(self, module_type: str) -> Optional[ModuleVersion]:
        """回滚到上一个版本"""
        with self._lock:
            versions = self._versions.get(module_type, [])
            if len(versions) < 2:
                return None

            # 找到当前活跃版本
            active = None
            for v in versions:
                if v.is_active:
                    active = v
                    break

            if not active:
                return None

            # 找到上一个版本
            sorted_versions = sorted(versions, key=lambda x: x.version, reverse=True)
            prev_version = None
            for v in sorted_versions:
                if v.version < active.version:
                    prev_version = v
                    break

            if prev_version:
                active.is_active = False
                prev_version.is_active = True
                self._save_versions()
                MODULE_VERSION.labels(module_type=module_type).set(prev_version.version)
                return prev_version

            return None

    def get_all_versions(self, module_type: str) -> list[ModuleVersion]:
        """获取所有版本"""
        return self._versions.get(module_type, []).copy()


# ============================================
# 自进化引擎
# ============================================

class SelfEvolutionEngine:
    """
    DSPy 自进化引擎

    协调数据收集、优化、部署的完整流程
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        data_dir: str = "data/dspy_training",
        modules_dir: str = "data/dspy_optimized",
    ):
        self.config = config or EvolutionConfig()
        self.collector = TrainingDataCollector(data_dir, self.config)
        self.version_manager = ModuleVersionManager(modules_dir, self.config)

        self._status = EvolutionStatus.IDLE
        self._last_evolution: Optional[datetime] = None
        self._evolution_lock = threading.Lock()
        self._auto_evolution_task: Optional[asyncio.Task] = None

    @property
    def status(self) -> EvolutionStatus:
        return self._status

    def record_successful_query(
        self,
        question: str,
        schema: str,
        sql: str,
        dialect: str = "sqlite",
        execution_result: Optional[list] = None,
    ):
        """
        记录成功的查询

        在查询成功执行后调用此方法收集训练数据
        """
        self.collector.add_sample(
            question=question,
            schema=schema,
            gold_sql=sql,
            dialect=dialect,
            execution_result=execution_result,
            source="user_query",
            verified=True,  # 执行成功的查询视为已验证
        )

    def should_evolve(self) -> tuple[bool, str]:
        """
        检查是否应该触发进化

        Returns:
            (should_evolve, reason)
        """
        if not HAS_DSPY:
            return False, "DSPy not installed"

        if not self.config.auto_evolution_enabled:
            return False, "Auto evolution disabled"

        if self._status != EvolutionStatus.IDLE:
            return False, f"Evolution in progress: {self._status.value}"

        # 检查样本数量
        samples = self.collector.get_samples(verified_only=True)
        if len(samples) < self.config.min_samples_for_evolution:
            return False, f"Not enough samples: {len(samples)} < {self.config.min_samples_for_evolution}"

        # 检查时间间隔
        if self._last_evolution:
            elapsed = datetime.utcnow() - self._last_evolution
            if elapsed < timedelta(hours=self.config.evolution_interval_hours):
                remaining = timedelta(hours=self.config.evolution_interval_hours) - elapsed
                return False, f"Too soon since last evolution, wait {remaining}"

        return True, "Ready for evolution"

    async def evolve(
        self,
        module_type: str = "text2sql",
        force: bool = False,
    ) -> EvolutionResult:
        """
        执行进化

        Args:
            module_type: 模块类型
            force: 强制执行（忽略检查）

        Returns:
            进化结果
        """
        if not HAS_DSPY:
            return EvolutionResult(
                success=False,
                module_type=module_type,
                old_version=0,
                new_version=0,
                old_accuracy=0,
                new_accuracy=0,
                train_samples=0,
                eval_samples=0,
                duration_seconds=0,
                error="DSPy not installed",
            )

        # 检查是否可以进化
        if not force:
            should, reason = self.should_evolve()
            if not should:
                EVOLUTION_RUNS.labels(status="skipped").inc()
                return EvolutionResult(
                    success=False,
                    module_type=module_type,
                    old_version=0,
                    new_version=0,
                    old_accuracy=0,
                    new_accuracy=0,
                    train_samples=0,
                    eval_samples=0,
                    duration_seconds=0,
                    error=reason,
                )

        # 获取锁
        if not self._evolution_lock.acquire(blocking=False):
            return EvolutionResult(
                success=False,
                module_type=module_type,
                old_version=0,
                new_version=0,
                old_accuracy=0,
                new_accuracy=0,
                train_samples=0,
                eval_samples=0,
                duration_seconds=0,
                error="Evolution already in progress",
            )

        start_time = datetime.utcnow()

        try:
            self._status = EvolutionStatus.COLLECTING

            # 1. 收集训练数据
            samples = self.collector.get_samples(verified_only=True)

            # 2. 准备 DSPy 数据集
            self._status = EvolutionStatus.OPTIMIZING

            # 划分训练集和验证集
            split_idx = int(len(samples) * self.config.train_split)
            train_samples = samples[:split_idx]
            eval_samples = samples[split_idx:]

            # 转换为 DSPy Example
            trainset = [
                dspy.Example(
                    question=s.question,
                    db_schema=s.schema,
                    dialect=s.dialect,
                    gold_sql=s.gold_sql,
                ).with_inputs('question', 'db_schema', 'dialect')
                for s in train_samples
            ]

            evalset = [
                dspy.Example(
                    question=s.question,
                    db_schema=s.schema,
                    dialect=s.dialect,
                    gold_sql=s.gold_sql,
                ).with_inputs('question', 'db_schema', 'dialect')
                for s in eval_samples
            ]

            # 3. 获取当前版本信息
            current_version = self.version_manager.get_active_version(module_type)
            old_version = current_version.version if current_version else 0
            old_accuracy = current_version.accuracy if current_version else 0

            # 4. 创建并优化模块
            from .sql_generator import Text2SQLModule, DecomposeAndGenerateModule, SQLRefineModule

            if module_type == "text2sql":
                module = Text2SQLModule()
            elif module_type == "decompose":
                module = DecomposeAndGenerateModule()
            elif module_type == "refine":
                module = SQLRefineModule()
            else:
                raise ValueError(f"Unknown module type: {module_type}")

            # 定义评估指标
            def metric(example, pred, trace=None):
                pred_sql = getattr(pred, 'sql', '').strip().lower()
                gold_sql = example.gold_sql.strip().lower()

                # 简单的标准化比较
                import re
                pred_norm = re.sub(r'\s+', ' ', pred_sql).rstrip(';')
                gold_norm = re.sub(r'\s+', ' ', gold_sql).rstrip(';')

                if pred_norm == gold_norm:
                    return 1.0

                # 部分匹配
                pred_words = set(pred_norm.split())
                gold_words = set(gold_norm.split())
                if gold_words:
                    return len(pred_words & gold_words) / len(pred_words | gold_words)
                return 0.0

            # 运行优化
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
            )

            optimized_module = optimizer.compile(module, trainset=trainset)

            # 5. 评估新模块
            self._status = EvolutionStatus.EVALUATING

            correct = 0
            for example in evalset:
                try:
                    pred = optimized_module(
                        question=example.question,
                        db_schema=example.db_schema,
                        dialect=example.dialect,
                    )
                    if metric(example, pred) >= 0.5:
                        correct += 1
                except Exception:
                    pass

            new_accuracy = correct / len(evalset) if evalset else 0

            # 6. 决定是否部署
            self._status = EvolutionStatus.DEPLOYING

            deployed = False
            if new_accuracy >= self.config.accuracy_threshold:
                # 检查是否比旧版本好
                if new_accuracy >= old_accuracy - self.config.degradation_threshold:
                    # 保存模块
                    new_version_num = old_version + 1
                    module_path = str(
                        self.version_manager.modules_dir /
                        f"{module_type}_v{new_version_num}.json"
                    )
                    optimized_module.save(module_path)

                    # 添加版本记录
                    version = self.version_manager.add_version(
                        module_type=module_type,
                        module_path=module_path,
                        accuracy=new_accuracy,
                        optimizer="BootstrapFewShot",
                        train_samples=len(train_samples),
                        metadata={
                            "eval_samples": len(eval_samples),
                            "old_accuracy": old_accuracy,
                        },
                    )

                    # 激活新版本
                    self.version_manager.activate_version(module_type, version.version)
                    deployed = True

                    EVOLUTION_ACCURACY.labels(module_type=module_type).set(new_accuracy)

            duration = (datetime.utcnow() - start_time).total_seconds()
            self._last_evolution = datetime.utcnow()

            EVOLUTION_RUNS.labels(status="success").inc()

            return EvolutionResult(
                success=True,
                module_type=module_type,
                old_version=old_version,
                new_version=old_version + 1 if deployed else old_version,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                train_samples=len(train_samples),
                eval_samples=len(eval_samples),
                duration_seconds=duration,
                deployed=deployed,
            )

        except Exception as e:
            EVOLUTION_RUNS.labels(status="failed").inc()
            return EvolutionResult(
                success=False,
                module_type=module_type,
                old_version=0,
                new_version=0,
                old_accuracy=0,
                new_accuracy=0,
                train_samples=0,
                eval_samples=0,
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                error=str(e),
            )

        finally:
            self._status = EvolutionStatus.IDLE
            self._evolution_lock.release()

    async def start_auto_evolution(self):
        """启动自动进化任务"""
        if self._auto_evolution_task is not None:
            return

        async def auto_evolve_loop():
            while True:
                try:
                    should, _ = self.should_evolve()
                    if should:
                        await self.evolve(module_type="text2sql")
                except Exception as e:
                    print(f"Auto evolution error: {e}")

                # 等待下一次检查
                await asyncio.sleep(3600)  # 每小时检查一次

        self._auto_evolution_task = asyncio.create_task(auto_evolve_loop())

    def stop_auto_evolution(self):
        """停止自动进化任务"""
        if self._auto_evolution_task:
            self._auto_evolution_task.cancel()
            self._auto_evolution_task = None

    def get_status(self) -> dict:
        """获取状态信息"""
        return {
            "status": self._status.value,
            "last_evolution": self._last_evolution.isoformat() if self._last_evolution else None,
            "auto_evolution_enabled": self.config.auto_evolution_enabled,
            "collector_stats": self.collector.get_stats(),
            "active_versions": {
                module_type: asdict(v) if v else None
                for module_type in ["text2sql", "decompose", "refine"]
                for v in [self.version_manager.get_active_version(module_type)]
            },
        }


# ============================================
# 全局实例
# ============================================

_evolution_engine: Optional[SelfEvolutionEngine] = None


def get_evolution_engine() -> SelfEvolutionEngine:
    """获取自进化引擎单例"""
    global _evolution_engine
    if _evolution_engine is None:
        _evolution_engine = SelfEvolutionEngine()
    return _evolution_engine


# ============================================
# 导出
# ============================================

__all__ = [
    "SelfEvolutionEngine",
    "TrainingDataCollector",
    "ModuleVersionManager",
    "EvolutionConfig",
    "EvolutionResult",
    "EvolutionStatus",
    "TrainingSample",
    "ModuleVersion",
    "get_evolution_engine",
]
