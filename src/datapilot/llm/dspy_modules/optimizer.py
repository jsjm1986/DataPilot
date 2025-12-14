# -*- coding: utf-8 -*-
"""
DSPy 优化器模块 (P2 增强版)

用于优化 Prompt 和 Few-shot 示例

增强功能:
- 完整的训练流程管理
- 验证集评估
- 多种优化策略
- 训练历史记录
- 自动模型选择
"""

import json
import time
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False

from .sql_generator import Text2SQLModule, MultiPathSQLGenerator


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本配置
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    num_candidate_programs: int = 10
    num_threads: int = 4

    # 优化策略
    optimizer_type: str = "bootstrap"  # bootstrap, random_search, mipro

    # 验证配置
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    min_accuracy_improvement: float = 0.01

    # 保存配置
    save_dir: str = "data/dspy_optimized"
    save_best_only: bool = True


@dataclass
class TrainingResult:
    """训练结果"""
    # 基本信息
    module_name: str = ""
    optimizer_type: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    # 数据集信息
    train_size: int = 0
    val_size: int = 0

    # 评估结果
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    best_accuracy: float = 0.0

    # 训练历史
    history: list = field(default_factory=list)

    # 保存路径
    model_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SQLGenerationMetric:
    """
    SQL 生成评估指标 (P2 增强)

    评估生成的 SQL 是否正确
    支持多种评估模式
    """

    def __init__(
        self,
        executor: Optional[Callable] = None,
        use_execution: bool = True,
        use_semantic: bool = True,
    ):
        """
        Args:
            executor: SQL 执行器函数
            use_execution: 是否使用执行结果评估
            use_semantic: 是否使用语义评估
        """
        self.executor = executor
        self.use_execution = use_execution
        self.use_semantic = use_semantic
        self._evaluation_cache: dict[str, float] = {}

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Optional[list] = None,
    ) -> float:
        """
        评估预测结果

        Args:
            example: 包含 gold_sql 的示例
            prediction: 模型预测
            trace: 推理轨迹

        Returns:
            评分 0-1
        """
        if not HAS_DSPY:
            return 0.0

        pred_sql = getattr(prediction, 'sql', '').strip().lower()
        gold_sql = getattr(example, 'gold_sql', '').strip().lower()

        if not pred_sql or not gold_sql:
            return 0.0

        # 缓存键
        cache_key = f"{pred_sql}||{gold_sql}"
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        score = 0.0

        # 1. 精确匹配 (最高分)
        if self._normalize_sql(pred_sql) == self._normalize_sql(gold_sql):
            score = 1.0
        # 2. 执行结果匹配
        elif self.use_execution and self.executor:
            try:
                pred_result = self.executor(prediction.sql)
                gold_result = self.executor(example.gold_sql)
                if self._compare_results(pred_result, gold_result):
                    score = 1.0
                else:
                    score = self._result_similarity(pred_result, gold_result)
            except Exception:
                score = self._partial_match_score(pred_sql, gold_sql)
        # 3. 语义相似度评分
        elif self.use_semantic:
            score = self._semantic_similarity(pred_sql, gold_sql)
        # 4. 部分匹配评分
        else:
            score = self._partial_match_score(pred_sql, gold_sql)

        self._evaluation_cache[cache_key] = score
        return score

    def _normalize_sql(self, sql: str) -> str:
        """标准化 SQL"""
        import re
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.rstrip(';')
        return sql.strip()

    def _compare_results(self, result1: list, result2: list) -> bool:
        """比较执行结果"""
        if not result1 and not result2:
            return True
        if len(result1) != len(result2):
            return False

        try:
            set1 = {tuple(sorted(r.items())) for r in result1}
            set2 = {tuple(sorted(r.items())) for r in result2}
            return set1 == set2
        except (TypeError, AttributeError):
            return False

    def _result_similarity(self, result1: list, result2: list) -> float:
        """结果相似度评分"""
        if not result1 or not result2:
            return 0.0

        # 行数相似度
        row_ratio = min(len(result1), len(result2)) / max(len(result1), len(result2))

        # 列名相似度
        if result1 and result2:
            cols1 = set(result1[0].keys()) if isinstance(result1[0], dict) else set()
            cols2 = set(result2[0].keys()) if isinstance(result2[0], dict) else set()
            if cols1 and cols2:
                col_ratio = len(cols1 & cols2) / len(cols1 | cols2)
            else:
                col_ratio = 0.0
        else:
            col_ratio = 0.0

        return (row_ratio * 0.5 + col_ratio * 0.5)

    def _semantic_similarity(self, pred: str, gold: str) -> float:
        """语义相似度评分"""
        # 提取 SQL 关键组件
        components = ['select', 'from', 'where', 'group by', 'order by', 'limit', 'join']

        pred_components = {}
        gold_components = {}

        for comp in components:
            pred_components[comp] = comp in pred
            gold_components[comp] = comp in gold

        # 组件匹配分数
        matches = sum(1 for c in components if pred_components[c] == gold_components[c])
        component_score = matches / len(components)

        # 关键词重叠分数
        keyword_score = self._partial_match_score(pred, gold)

        return (component_score * 0.4 + keyword_score * 0.6)

    def _partial_match_score(self, pred: str, gold: str) -> float:
        """部分匹配评分"""
        pred_keywords = set(pred.split())
        gold_keywords = set(gold.split())

        if not gold_keywords:
            return 0.0

        intersection = pred_keywords & gold_keywords
        union = pred_keywords | gold_keywords

        return len(intersection) / len(union) if union else 0.0


class DSPyTrainer:
    """
    DSPy 训练器 (P2 新增)

    完整的训练流程管理
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Args:
            config: 训练配置
        """
        if not HAS_DSPY:
            raise ImportError("DSPy is required for training. Install with: pip install dspy-ai")

        self.config = config or TrainingConfig()
        self._training_history: list[TrainingResult] = []

    def train(
        self,
        module: Any,
        trainset: list,
        valset: Optional[list] = None,
        metric: Optional[Callable] = None,
        module_name: str = "text2sql",
    ) -> TrainingResult:
        """
        训练模块

        Args:
            module: 要训练的 DSPy 模块
            trainset: 训练集 (dspy.Example 列表)
            valset: 验证集 (可选，如果不提供则从训练集分割)
            metric: 评估指标
            module_name: 模块名称 (用于保存)

        Returns:
            训练结果
        """
        result = TrainingResult(
            module_name=module_name,
            optimizer_type=self.config.optimizer_type,
            started_at=datetime.now().isoformat(),
        )

        start_time = time.time()

        # 分割验证集
        if valset is None and self.config.validation_split > 0:
            split_idx = int(len(trainset) * (1 - self.config.validation_split))
            trainset, valset = trainset[:split_idx], trainset[split_idx:]

        result.train_size = len(trainset)
        result.val_size = len(valset) if valset else 0

        # 设置评估指标
        if metric is None:
            metric = SQLGenerationMetric()

        # 选择优化器
        if self.config.optimizer_type == "random_search":
            optimizer = BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                num_candidate_programs=self.config.num_candidate_programs,
                num_threads=self.config.num_threads,
            )
            optimized_module = optimizer.compile(
                module,
                trainset=trainset,
                valset=valset,
            )
        else:
            # 默认使用 BootstrapFewShot
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
            )
            optimized_module = optimizer.compile(
                module,
                trainset=trainset,
            )

        # 评估训练集准确率
        result.train_accuracy = self._evaluate(optimized_module, trainset, metric)

        # 评估验证集准确率
        if valset:
            result.val_accuracy = self._evaluate(optimized_module, valset, metric)
            result.best_accuracy = result.val_accuracy
        else:
            result.best_accuracy = result.train_accuracy

        # 记录训练历史
        result.history.append({
            "epoch": 1,
            "train_accuracy": result.train_accuracy,
            "val_accuracy": result.val_accuracy,
        })

        # 保存模型
        if self.config.save_dir:
            save_path = self._save_module(
                optimized_module,
                module_name,
                result,
            )
            result.model_path = save_path

        # 记录时间
        result.duration_seconds = time.time() - start_time
        result.finished_at = datetime.now().isoformat()

        # 添加到历史
        self._training_history.append(result)

        return result

    def _evaluate(
        self,
        module: Any,
        dataset: list,
        metric: Callable,
    ) -> float:
        """评估模块"""
        if not dataset:
            return 0.0

        total_score = 0.0
        for example in dataset:
            try:
                prediction = module(
                    question=example.question,
                    schema=getattr(example, 'db_schema', ''),
                    dialect=getattr(example, 'dialect', 'sqlite'),
                )
                score = metric(example, prediction)
                total_score += score
            except Exception:
                pass

        return total_score / len(dataset)

    def _save_module(
        self,
        module: Any,
        module_name: str,
        result: TrainingResult,
    ) -> str:
        """保存模块"""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{module_name}_{self.config.optimizer_type}_{timestamp}.json"
        save_path = save_dir / filename

        # 保存模块
        module.save(str(save_path))

        # 添加元数据
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.update({
            "metadata": {
                "module_name": module_name,
                "optimizer": self.config.optimizer_type,
                "accuracy": result.best_accuracy,
                "train_accuracy": result.train_accuracy,
                "val_accuracy": result.val_accuracy,
                "train_size": result.train_size,
                "val_size": result.val_size,
                "created_at": result.finished_at,
                "duration_seconds": result.duration_seconds,
            }
        })

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(save_path)

    def get_history(self) -> list[TrainingResult]:
        """获取训练历史"""
        return self._training_history


class DSPyOptimizer:
    """
    DSPy 优化器 (保持向后兼容)

    用于优化 SQL 生成模块的 Prompt
    """

    def __init__(
        self,
        module: Any,
        metric: Callable,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
    ):
        """
        Args:
            module: 要优化的 DSPy 模块
            metric: 评估指标
            max_bootstrapped_demos: 最大 bootstrap 示例数
            max_labeled_demos: 最大标注示例数
        """
        if not HAS_DSPY:
            raise ImportError("DSPy is required. Install with: pip install dspy-ai")

        self.module = module
        self.metric = metric
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos

    def optimize(
        self,
        trainset: list,
        valset: Optional[list] = None,
        num_threads: int = 4,
    ) -> Any:
        """
        优化模块

        Args:
            trainset: 训练集
            valset: 验证集
            num_threads: 并行线程数

        Returns:
            优化后的模块
        """
        optimizer = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
        )

        optimized_module = optimizer.compile(
            self.module,
            trainset=trainset,
        )

        return optimized_module

    def optimize_with_search(
        self,
        trainset: list,
        valset: Optional[list] = None,
        num_candidate_programs: int = 10,
        num_threads: int = 4,
    ) -> Any:
        """
        使用随机搜索优化

        Args:
            trainset: 训练集
            valset: 验证集
            num_candidate_programs: 候选程序数
            num_threads: 并行线程数

        Returns:
            优化后的模块
        """
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=self.metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            num_candidate_programs=num_candidate_programs,
            num_threads=num_threads,
        )

        optimized_module = optimizer.compile(
            self.module,
            trainset=trainset,
            valset=valset,
        )

        return optimized_module


def load_training_data(file_path: str) -> list:
    """
    加载训练数据

    Args:
        file_path: JSON 文件路径

    Returns:
        DSPy Example 列表
    """
    if not HAS_DSPY:
        raise ImportError("DSPy is required. Install with: pip install dspy-ai")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = dspy.Example(
            question=item['question'],
            db_schema=item['schema'],
            dialect=item.get('dialect', 'sqlite'),
            gold_sql=item['sql'],
        ).with_inputs('question', 'db_schema', 'dialect')
        examples.append(example)

    return examples


def create_training_data(
    questions: list[str],
    schemas: list[str],
    sqls: list[str],
    dialect: str = "sqlite",
) -> list:
    """
    创建训练数据 (P2 新增)

    Args:
        questions: 问题列表
        schemas: Schema 列表
        sqls: SQL 列表
        dialect: SQL 方言

    Returns:
        DSPy Example 列表
    """
    if not HAS_DSPY:
        raise ImportError("DSPy is required. Install with: pip install dspy-ai")

    if len(questions) != len(schemas) or len(questions) != len(sqls):
        raise ValueError("questions, schemas, sqls must have same length")

    examples = []
    for q, s, sql in zip(questions, schemas, sqls):
        example = dspy.Example(
            question=q,
            db_schema=s,
            dialect=dialect,
            gold_sql=sql,
        ).with_inputs('question', 'db_schema', 'dialect')
        examples.append(example)

    return examples


def save_optimized_module(module: Any, save_path: str) -> None:
    """
    保存优化后的模块

    Args:
        module: 优化后的模块
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    module.save(save_path)


def load_optimized_module(module_class: type, load_path: str) -> Any:
    """
    加载优化后的模块

    Args:
        module_class: 模块类
        load_path: 加载路径

    Returns:
        加载的模块
    """
    module = module_class()
    module.load(load_path)
    return module


__all__ = [
    # 配置
    "TrainingConfig",
    "TrainingResult",
    # 评估
    "SQLGenerationMetric",
    # 训练
    "DSPyTrainer",
    "DSPyOptimizer",
    # 数据
    "load_training_data",
    "create_training_data",
    # 模块管理
    "save_optimized_module",
    "load_optimized_module",
]
