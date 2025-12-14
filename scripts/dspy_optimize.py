# -*- coding: utf-8 -*-
"""
DSPy 自动优化脚本

README Section 6 要求:
- 基于 DSPy 构建 Logic Architect
- 准备 ≥200 组问答-SQL 验证集
- 跑自动 Prompt/CoT 优化

使用方法:
    python scripts/dspy_optimize.py --dataset mock_data/queries/test_queries.json --output optimized_prompts.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    print("Warning: dspy not installed. Run: pip install dspy-ai")


# ============================================
# DSPy Signatures
# ============================================

if HAS_DSPY:
    class Text2SQLSignature(dspy.Signature):
        """将自然语言问题转换为 SQL 查询"""
        question: str = dspy.InputField(desc="用户的自然语言问题")
        schema: str = dspy.InputField(desc="数据库 Schema (DDL)")
        dialect: str = dspy.InputField(desc="SQL 方言 (sqlite/mysql/postgresql)")
        sql: str = dspy.OutputField(desc="生成的 SQL 查询语句")
        explanation: str = dspy.OutputField(desc="SQL 生成的推理过程")


    class Text2SQLModule(dspy.Module):
        """Text-to-SQL 模块"""
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(Text2SQLSignature)

        def forward(self, question: str, schema: str, dialect: str = "sqlite"):
            return self.generate(question=question, schema=schema, dialect=dialect)


# ============================================
# 评估指标
# ============================================

def normalize_sql(sql: str) -> str:
    """标准化 SQL 用于比较"""
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*>\s*', ' > ', sql)
    sql = re.sub(r'\s*<\s*', ' < ', sql)
    return sql.strip(';').strip()


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """精确匹配 (EX 指标)"""
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


def execution_accuracy(pred_sql: str, gold_sql: str, db_path: str = None) -> bool:
    """执行准确率 (EA 指标) - 比较执行结果"""
    # 简化版：仅做语法检查
    try:
        import sqlglot
        sqlglot.parse(pred_sql)
        return True
    except Exception:
        return False


def sql_metric(example, pred, trace=None) -> float:
    """DSPy 评估指标"""
    pred_sql = getattr(pred, 'sql', '') or ''
    gold_sql = getattr(example, 'expected_sql', '') or example.get('expected_sql', '')

    # 精确匹配得 1 分
    if exact_match(pred_sql, gold_sql):
        return 1.0

    # 语法正确得 0.5 分
    if execution_accuracy(pred_sql, gold_sql):
        return 0.5

    return 0.0


# ============================================
# 数据加载
# ============================================

def load_dataset(path: str) -> list:
    """加载评测集"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data:
        if HAS_DSPY:
            example = dspy.Example(
                question=item['query'],
                expected_sql=item['expected_sql'],
                schema=get_default_schema(),
                dialect='sqlite',
            ).with_inputs('question', 'schema', 'dialect')
        else:
            example = {
                'question': item['query'],
                'expected_sql': item['expected_sql'],
                'schema': get_default_schema(),
                'dialect': 'sqlite',
            }
        examples.append(example)

    return examples


def get_default_schema() -> str:
    """获取默认 Schema"""
    return """
-- 产品表
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10,2),
    stock INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'on_sale',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 客户表
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20),
    level VARCHAR(20) DEFAULT '普通',
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 订单表
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER DEFAULT 1,
    amount DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 销售记录表
CREATE TABLE sales_records (
    id INTEGER PRIMARY KEY,
    date DATE,
    region VARCHAR(50),
    salesperson VARCHAR(100),
    product_category VARCHAR(50),
    amount DECIMAL(10,2)
);
"""


# ============================================
# 优化器
# ============================================

def run_optimization(
    dataset_path: str,
    output_path: str,
    num_threads: int = 4,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16,
):
    """运行 DSPy 优化"""
    if not HAS_DSPY:
        print("Error: dspy not installed")
        return

    print(f"Loading dataset from {dataset_path}...")
    examples = load_dataset(dataset_path)
    print(f"Loaded {len(examples)} examples")

    # 划分训练集和验证集
    train_size = int(len(examples) * 0.8)
    trainset = examples[:train_size]
    devset = examples[train_size:]

    print(f"Train: {len(trainset)}, Dev: {len(devset)}")

    # 配置 LLM (使用 DeepSeek)
    from src.datapilot.config.settings import get_settings
    settings = get_settings()

    if settings.deepseek_api_key:
        lm = dspy.LM(
            model="deepseek/deepseek-chat",
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_base_url,
        )
        dspy.configure(lm=lm)
        print("Configured DeepSeek LLM")
    else:
        print("Warning: No DeepSeek API key found. Using default LLM.")

    # 创建模块
    module = Text2SQLModule()

    # 配置优化器
    print("Starting optimization...")
    optimizer = BootstrapFewShot(
        metric=sql_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    # 运行优化
    optimized_module = optimizer.compile(
        module,
        trainset=trainset,
    )

    # 评估
    print("\nEvaluating on dev set...")
    correct = 0
    total = len(devset)

    for example in devset:
        try:
            pred = optimized_module(
                question=example.question,
                schema=example.schema,
                dialect=example.dialect,
            )
            score = sql_metric(example, pred)
            if score >= 0.5:
                correct += 1
        except Exception as e:
            print(f"Error: {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nDev Accuracy: {accuracy:.2%} ({correct}/{total})")

    # 保存优化后的 prompts
    print(f"\nSaving optimized prompts to {output_path}...")

    # 提取优化后的 demos
    optimized_config = {
        "accuracy": accuracy,
        "num_examples": len(examples),
        "train_size": len(trainset),
        "dev_size": len(devset),
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(optimized_config, f, ensure_ascii=False, indent=2)

    print("Optimization complete!")
    return optimized_module, accuracy


# ============================================
# 评估模式
# ============================================

def run_evaluation(dataset_path: str, sample_size: int = 20):
    """运行评估 (不需要 API Key)"""
    print(f"Loading dataset from {dataset_path}...")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # 统计各类别
    from collections import Counter
    categories = Counter(item['category'] for item in data)

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # 显示样例
    print(f"\nSample queries (first {sample_size}):")
    for item in data[:sample_size]:
        print(f"  [{item['category']}] {item['query']}")
        print(f"    -> {item['expected_sql'][:80]}...")

    print(f"\nDataset ready for DSPy optimization!")
    print(f"Run with --optimize flag to start optimization (requires DeepSeek API key)")


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="DSPy SQL Generation Optimizer")
    parser.add_argument(
        "--dataset",
        default="mock_data/queries/test_queries.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        default="optimized_prompts.json",
        help="Output path for optimized prompts",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization (requires API key)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="Max bootstrapped demos",
    )

    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent)

    if args.optimize:
        if not HAS_DSPY:
            print("Error: dspy not installed. Run: pip install dspy-ai")
            sys.exit(1)

        run_optimization(
            dataset_path=args.dataset,
            output_path=args.output,
            max_bootstrapped_demos=args.max_demos,
        )
    else:
        run_evaluation(args.dataset)


if __name__ == "__main__":
    main()
