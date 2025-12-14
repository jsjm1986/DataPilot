# -*- coding: utf-8 -*-
"""DSPy Modules"""

from .signatures import (
    Text2SQLSignature,
    SQLDecomposeSignature,
    SQLRefineSignature,
    SchemaSelectionSignature,
    ValueMappingSignature,
    AmbiguityDetectionSignature,
    ChartRecommendSignature,
)

from .sql_generator import (
    Text2SQLModule,
    DecomposeAndGenerateModule,
    SQLRefineModule,
    SchemaSelectionModule,
    MultiPathSQLGenerator,
)

from .optimizer import (
    # 配置
    TrainingConfig,
    TrainingResult,
    # 评估
    SQLGenerationMetric,
    # 训练
    DSPyTrainer,
    DSPyOptimizer,
    # 数据
    load_training_data,
    create_training_data,
    # 模块管理
    save_optimized_module,
    load_optimized_module,
)

from .self_evolution import (
    SelfEvolutionEngine,
    TrainingDataCollector,
    ModuleVersionManager,
    EvolutionConfig,
    EvolutionResult,
    EvolutionStatus,
    TrainingSample,
    ModuleVersion,
    get_evolution_engine,
)

__all__ = [
    # Signatures
    "Text2SQLSignature",
    "SQLDecomposeSignature",
    "SQLRefineSignature",
    "SchemaSelectionSignature",
    "ValueMappingSignature",
    "AmbiguityDetectionSignature",
    "ChartRecommendSignature",
    # Modules
    "Text2SQLModule",
    "DecomposeAndGenerateModule",
    "SQLRefineModule",
    "SchemaSelectionModule",
    "MultiPathSQLGenerator",
    # Optimizer - Config
    "TrainingConfig",
    "TrainingResult",
    # Optimizer - Training
    "SQLGenerationMetric",
    "DSPyTrainer",
    "DSPyOptimizer",
    "load_training_data",
    "create_training_data",
    "save_optimized_module",
    "load_optimized_module",
    # Self Evolution
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
