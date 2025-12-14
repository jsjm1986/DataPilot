# -*- coding: utf-8 -*-
"""
Logic Architect Agent (LLM-Native 版)
基于 DSPy + MAC-SQL 分解策略生成多路 SQL

LLM-Native 设计哲学:
- 使用 SchemaIntrospector 提供的丰富元信息
- 时间字段、枚举值、样本值帮助 LLM 生成准确 SQL
- 无硬编码规则，一切由 LLM 根据真实数据决定

增强功能:
- DSPy teleprompter 优化模块加载
- SQL 方言转换
- **LLM-Native 上下文支持** - 使用时间/枚举/样本值上下文
"""

from typing import Any, Optional
from pathlib import Path
import os
import json
import re
from datetime import datetime

from ..core.state import DataPilotState, SQLCandidate
from ..db.connector import get_db_manager
from ..llm.deepseek import get_deepseek_client
from ..config.settings import get_settings

# DSPy 可选导入
try:
    import dspy
    from ..llm.dspy_modules.sql_generator import (
        Text2SQLModule,
        DecomposeAndGenerateModule,
        SQLRefineModule,
        MultiPathSQLGenerator,
    )
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False


# 优化模块路径
OPTIMIZED_MODULES_DIR = Path("data/dspy_optimized")


class OptimizedModuleLoader:
    """
    DSPy 优化模块加载器

    支持加载 teleprompter 优化后的模块:
    - BootstrapFewShot 优化
    - MIPRO 优化
    - 自定义优化
    """

    def __init__(self, modules_dir: Path = OPTIMIZED_MODULES_DIR):
        self.modules_dir = modules_dir
        self._loaded_modules: dict[str, Any] = {}
        self._module_metadata: dict[str, dict] = {}

    def get_available_modules(self) -> list[dict]:
        """获取可用的优化模块列表"""
        modules = []

        if not self.modules_dir.exists():
            return modules

        for file_path in self.modules_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                modules.append({
                    "name": file_path.stem,
                    "path": str(file_path),
                    "optimizer": metadata.get("optimizer", "unknown"),
                    "accuracy": metadata.get("accuracy", 0),
                    "created_at": metadata.get("created_at", ""),
                    "description": metadata.get("description", ""),
                })
            except Exception:
                continue

        # 按准确率排序
        modules.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
        return modules

    def load_module(self, module_name: str) -> Optional[Any]:
        """
        加载优化后的 DSPy 模块

        Args:
            module_name: 模块名称 (不含扩展名)

        Returns:
            加载的模块，如果失败返回 None
        """
        if not HAS_DSPY:
            return None

        # 检查缓存
        if module_name in self._loaded_modules:
            return self._loaded_modules[module_name]

        module_path = self.modules_dir / f"{module_name}.json"
        if not module_path.exists():
            return None

        try:
            # 加载模块
            module = dspy.load(str(module_path))
            self._loaded_modules[module_name] = module

            # 加载元数据
            with open(module_path, "r", encoding="utf-8") as f:
                self._module_metadata[module_name] = json.load(f)

            return module
        except Exception as e:
            print(f"Failed to load optimized module {module_name}: {e}")
            return None

    def get_best_module(self, task_type: str = "text2sql") -> Optional[Any]:
        """
        获取指定任务类型的最佳优化模块

        Args:
            task_type: 任务类型 (text2sql, decompose, refine)

        Returns:
            最佳模块，如果没有返回 None
        """
        modules = self.get_available_modules()

        # 过滤指定任务类型
        task_modules = [m for m in modules if task_type in m["name"].lower()]

        if not task_modules:
            return None

        # 返回准确率最高的
        best = task_modules[0]
        return self.load_module(best["name"])

    def save_optimized_module(
        self,
        module: Any,
        name: str,
        optimizer: str,
        accuracy: float,
        description: str = "",
    ):
        """
        保存优化后的模块

        Args:
            module: DSPy 模块
            name: 模块名称
            optimizer: 优化器名称
            accuracy: 准确率
            description: 描述
        """
        if not HAS_DSPY:
            return

        self.modules_dir.mkdir(parents=True, exist_ok=True)

        module_path = self.modules_dir / f"{name}.json"

        try:
            # 保存模块
            module.save(str(module_path))

            # 更新元数据
            with open(module_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data.update({
                "optimizer": optimizer,
                "accuracy": accuracy,
                "description": description,
                "created_at": __import__("datetime").datetime.now().isoformat(),
            })

            with open(module_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Failed to save optimized module: {e}")


class LogicArchitect:
    """
    Logic Architect Agent - SQL 构建者 (LLM-Native 版)

    职责:
    1. 基于 DSPy 进行 SQL 生成 (Chain-of-Thought)
    2. MAC-SQL 分解策略处理复杂问题
    3. 多路候选生成 (direct + decompose)
    4. 处理错误反馈进行 SQL 修正 (Self-Correction)

    LLM-Native 增强:
    - 使用 SchemaIntrospector 提供的时间字段、枚举值、样本值上下文
    - LLM 根据真实数据库信息生成 SQL
    - 无硬编码的时间解析或值映射规则
    """

    # LLM-Native SQL 生成 System Prompt
    NATIVE_SQL_SYSTEM_PROMPT = """你是一个专业的 SQL 生成专家。你的任务是根据用户的自然语言查询和提供的数据库信息，生成准确的 SQL 语句。

## 关键原则

### 1. 基于真实数据库结构
- 只使用提供的表和列，不要假设不存在的结构
- 注意列的数据类型，特别是时间字段
- 利用样本值来理解数据的实际格式

### 2. 时间处理
- 查看提供的时间字段列表，选择合适的字段
- 根据今天的日期计算相对时间（如"最近7天"、"上个月"）
- 注意时间字段的实际格式（DATE、DATETIME、TIMESTAMP）

### 3. 值匹配
- 使用提供的枚举值/样本值来匹配用户的表述
- 如果用户说"苹果手机"而样本值中有"iPhone"，使用"iPhone"
- 对于模糊匹配，使用 LIKE 并说明原因

### 4. 不确定时说明
- 如果查询有歧义，在解释中说明你的假设
- 如果找不到完全匹配的值，说明你使用的模糊匹配策略

### 5. 安全性
- 不生成 DROP、DELETE、TRUNCATE 等危险语句
- 大表查询要考虑添加 LIMIT

## 输出格式

返回 JSON:
```json
{
    "sql": "生成的 SQL 语句",
    "explanation": "SQL 解释和生成理由",
    "confidence": 0.0-1.0,
    "warnings": ["警告信息"]
}
```
"""

    def __init__(self, use_optimized_modules: Optional[bool] = None):
        self.llm = get_deepseek_client()
        self.db_manager = get_db_manager()
        self.settings = get_settings()
        self._dspy_configured = False
        # 使用 settings 作为默认值，允许参数覆盖
        self.use_optimized_modules = use_optimized_modules if use_optimized_modules is not None else self.settings.logic_architect_use_optimized_modules
        # 新增: 从 settings 读取其他配置
        self.sql_candidates_count = self.settings.logic_architect_sql_candidates_count
        self.enable_self_correction = self.settings.logic_architect_enable_self_correction
        self.max_correction_rounds = self.settings.logic_architect_max_correction_rounds

        # 优化模块加载器
        self.module_loader = OptimizedModuleLoader()

        # 初始化 DSPy 模块
        if HAS_DSPY:
            self._setup_dspy()
            self._init_dspy_modules()

    def _setup_dspy(self):
        """配置 DSPy 使用 DeepSeek"""
        if self._dspy_configured:
            return

        try:
            # 配置 DeepSeek 作为 DSPy 的 LLM
            api_key = self.settings.deepseek_api_key
            base_url = self.settings.deepseek_base_url

            if api_key:
                lm = dspy.LM(
                    model="deepseek/deepseek-chat",
                    api_key=api_key,
                    api_base=base_url,
                )
                dspy.configure(lm=lm)
                self._dspy_configured = True
        except Exception:
            # DSPy 配置失败，回退到直接 LLM 调用
            pass

    def _init_dspy_modules(self):
        """
        初始化 DSPy 模块

        优先加载优化后的模块，如果没有则使用默认模块
        """
        # 尝试加载优化模块
        if self.use_optimized_modules:
            optimized_direct = self.module_loader.get_best_module("text2sql")
            optimized_decompose = self.module_loader.get_best_module("decompose")
            optimized_refine = self.module_loader.get_best_module("refine")

            if optimized_direct:
                self.direct_generator = optimized_direct
                print("Loaded optimized text2sql module")
            else:
                self.direct_generator = Text2SQLModule()

            if optimized_decompose:
                self.decompose_generator = optimized_decompose
                print("Loaded optimized decompose module")
            else:
                self.decompose_generator = DecomposeAndGenerateModule()

            if optimized_refine:
                self.refine_module = optimized_refine
                print("Loaded optimized refine module")
            else:
                self.refine_module = SQLRefineModule()
        else:
            # 使用默认模块
            self.direct_generator = Text2SQLModule()
            self.decompose_generator = DecomposeAndGenerateModule()
            self.refine_module = SQLRefineModule()

        self.multi_path_generator = MultiPathSQLGenerator()

    def get_module_info(self) -> dict:
        """获取当前使用的模块信息"""
        available = self.module_loader.get_available_modules()
        return {
            "dspy_available": HAS_DSPY,
            "dspy_configured": self._dspy_configured,
            "use_optimized": self.use_optimized_modules,
            "available_optimized_modules": available,
            "modules_dir": str(self.module_loader.modules_dir),
        }

    def _is_complex_query(self, query: str) -> bool:
        """
        判断是否为复杂查询，需要使用 MAC-SQL 分解策略

        复杂查询特征:
        - 包含多个条件 (AND/OR)
        - 包含子查询关键词
        - 包含多表关联
        - 包含聚合 + 过滤
        """
        query_lower = query.lower()

        # 复杂度指标
        complexity_indicators = [
            # 多条件
            "并且" in query or "而且" in query or "同时" in query,
            "或者" in query or "或" in query,
            # 子查询
            "其中" in query or "满足" in query,
            # 多表
            query.count("的") >= 3,
            # 聚合 + 过滤
            any(agg in query_lower for agg in ["最高", "最低", "平均", "总", "排名"]) and
            any(flt in query_lower for flt in ["大于", "小于", "超过", "不足"]),
            # 时间范围 + 分组
            any(time in query_lower for time in ["每日", "每月", "每周", "趋势"]) and
            any(grp in query_lower for grp in ["各", "分别", "按"]),
        ]

        return sum(complexity_indicators) >= 2

    async def generate_sql_with_dspy(
        self,
        query: str,
        schema: str,
        dialect: str = "sqlite",
        use_decompose: bool = False,
    ) -> list[SQLCandidate]:
        """
        使用 DSPy 生成 SQL (多路候选)

        Args:
            query: 用户问题
            schema: 数据库 Schema
            dialect: SQL 方言
            use_decompose: 是否使用分解策略

        Returns:
            SQL 候选列表
        """
        candidates = []

        try:
            # 策略 1: 直接生成
            direct_result = self.direct_generator(
                question=query,
                schema=schema,
                dialect=dialect,
            )
            candidates.append(SQLCandidate(
                id="dspy_direct",
                sql=direct_result.sql,
                explanation=getattr(direct_result, 'explanation', 'Direct generation'),
                confidence=0.85,
                strategy="direct",
            ))

            # 策略 2: MAC-SQL 分解生成 (复杂查询)
            if use_decompose:
                decompose_result = self.decompose_generator(
                    question=query,
                    schema=schema,
                    dialect=dialect,
                )
                candidates.append(SQLCandidate(
                    id="dspy_decompose",
                    sql=decompose_result.sql,
                    explanation=getattr(decompose_result, 'explanation', 'Decompose generation'),
                    confidence=0.80,
                    strategy="decompose",
                ))

        except Exception as e:
            # DSPy 失败，回退到直接 LLM 调用
            fallback = await self._generate_sql_fallback(query, schema, dialect)
            candidates.append(fallback)

        return candidates

    async def _generate_sql_fallback(
        self,
        query: str,
        schema: str,
        dialect: str,
    ) -> SQLCandidate:
        """回退方案：直接调用 LLM"""
        result = await self.llm.generate_sql(
            query=query,
            schema=schema,
            dialect=dialect,
        )
        return SQLCandidate(
            id="llm_direct",
            sql=result.get("sql", ""),
            explanation=result.get("explanation", ""),
            confidence=0.75,
            strategy="fallback",
        )

    async def generate_sql(
        self,
        query: str,
        schema: str,
        database: str = "default",
        value_mappings: Optional[dict] = None,
        time_context: Optional[str] = None,
        enum_context: Optional[str] = None,
        schema_context: Optional[str] = None,
    ) -> list[SQLCandidate]:
        """
        生成 SQL 查询 (LLM-Native 多路候选)

        Args:
            query: 用户自然语言问题
            schema: 数据库 Schema (DDL)
            database: 数据库名称
            value_mappings: 值映射
            time_context: 时间字段上下文 (来自 SchemaIntrospector)
            enum_context: 枚举字段上下文 (来自 SchemaIntrospector)
            schema_context: 完整 Schema 上下文 (来自 SchemaIntrospector)

        Returns:
            SQL 候选列表
        """
        # 确定 SQL 方言
        connector = self.db_manager.get(database)
        dialect = connector.db_type

        # 使用 LLM-Native 上下文 (如果提供)
        if schema_context or time_context or enum_context:
            candidates = await self._generate_sql_native(
                query=query,
                schema=schema,
                dialect=dialect,
                value_mappings=value_mappings,
                time_context=time_context,
                enum_context=enum_context,
                schema_context=schema_context,
            )
            return candidates

        # 回退到传统方式
        # 构建值映射提示
        schema_with_mappings = schema
        if value_mappings:
            mapping_lines = []
            for term, mapping in value_mappings.items():
                if isinstance(mapping, dict):
                    mapping_lines.append(
                        f"- '{term}' 对应数据库值 '{mapping.get('db_value', '')}' "
                        f"(表: {mapping.get('table_name', '')}, 列: {mapping.get('column_name', '')})"
                    )
            if mapping_lines:
                schema_with_mappings = schema + "\n\n## 值映射\n" + "\n".join(mapping_lines)

        # 判断是否需要分解策略
        use_decompose = self._is_complex_query(query)

        # 使用 DSPy 生成 (如果可用)
        if HAS_DSPY and self._dspy_configured:
            candidates = await self.generate_sql_with_dspy(
                query=query,
                schema=schema_with_mappings,
                dialect=dialect,
                use_decompose=use_decompose,
            )
        else:
            # 回退到直接 LLM 调用
            fallback = await self._generate_sql_fallback(
                query=query,
                schema=schema_with_mappings,
                dialect=dialect,
            )
            candidates = [fallback]

            # 如果是复杂查询，尝试第二种策略
            if use_decompose:
                second = await self._generate_sql_with_decompose_hint(
                    query=query,
                    schema=schema_with_mappings,
                    dialect=dialect,
                )
                candidates.append(second)

        return candidates

    async def _generate_sql_native(
        self,
        query: str,
        schema: str,
        dialect: str,
        value_mappings: Optional[dict] = None,
        time_context: Optional[str] = None,
        enum_context: Optional[str] = None,
        schema_context: Optional[str] = None,
    ) -> list[SQLCandidate]:
        """
        LLM-Native SQL 生成

        核心方法: 使用 SchemaIntrospector 提供的丰富上下文
        让 LLM 根据真实数据库信息生成 SQL
        """
        # 获取当前日期
        current_date = datetime.now().strftime("%Y-%m-%d")

        # 构建用户消息
        user_message_parts = [f"## 当前日期\n{current_date}"]

        # 优先使用完整的 schema_context
        if schema_context:
            user_message_parts.append(f"## 数据库结构\n{schema_context}")
        else:
            user_message_parts.append(f"## 数据库结构\n{schema}")

        if time_context:
            user_message_parts.append(f"## 时间字段\n{time_context}")

        if enum_context:
            user_message_parts.append(f"## 枚举/状态字段\n{enum_context}")

        if value_mappings:
            mapping_lines = []
            for term, mapping in value_mappings.items():
                if isinstance(mapping, dict):
                    mapping_lines.append(
                        f"- '{term}' → '{mapping.get('db_value', '')}' "
                        f"({mapping.get('table', '')}.{mapping.get('column', '')})"
                    )
            if mapping_lines:
                user_message_parts.append(f"## 值映射\n" + "\n".join(mapping_lines))

        user_message_parts.append(f"## SQL 方言\n{dialect.upper()}")
        user_message_parts.append(f"## 用户查询\n{query}")
        user_message_parts.append("请根据以上信息生成 SQL。")

        user_message = "\n\n".join(user_message_parts)

        messages = [
            {"role": "system", "content": self.NATIVE_SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        candidates = []

        try:
            response = await self.llm.chat(messages)
            result = self._parse_native_response(response)

            candidates.append(SQLCandidate(
                id="native_primary",
                sql=result.get("sql", ""),
                explanation=result.get("explanation", "LLM-Native generation"),
                confidence=result.get("confidence", 0.85),
                strategy="native",
            ))

            # 如果是复杂查询，生成第二个候选
            if self._is_complex_query(query):
                second = await self._generate_sql_with_decompose_hint(
                    query=query,
                    schema=schema_context or schema,
                    dialect=dialect,
                )
                candidates.append(second)

        except Exception as e:
            # 失败时回退
            fallback = await self._generate_sql_fallback(query, schema, dialect)
            candidates.append(fallback)

        return candidates

    def _parse_native_response(self, response: str) -> dict:
        """解析 LLM-Native 响应"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 尝试直接提取 SQL
        sql_match = re.search(r'(?:```sql\s*)?(SELECT[\s\S]*?)(?:```|$)', response, re.IGNORECASE)
        if sql_match:
            return {
                "sql": sql_match.group(1).strip(),
                "explanation": "从响应中直接提取 SQL",
                "confidence": 0.7,
            }

        return {"sql": "", "explanation": "无法解析响应", "confidence": 0.0}

    async def _generate_sql_with_decompose_hint(
        self,
        query: str,
        schema: str,
        dialect: str,
    ) -> SQLCandidate:
        """使用分解提示生成 SQL"""
        decompose_prompt = f"""这是一个复杂查询，请按以下步骤处理：
1. 将问题分解为多个简单子问题
2. 为每个子问题生成 SQL
3. 使用 CTE (WITH 子句) 或子查询合并结果

用户问题: {query}
"""
        result = await self.llm.generate_sql(
            query=decompose_prompt,
            schema=schema,
            dialect=dialect,
        )
        return SQLCandidate(
            id="llm_decompose",
            sql=result.get("sql", ""),
            explanation=result.get("explanation", "Decompose strategy"),
            confidence=0.70,
            strategy="decompose",
        )

    async def refine_sql(
        self,
        original_sql: str,
        error_message: str,
        schema: str,
        database: str = "default",
        hint: str = "",
    ) -> SQLCandidate:
        """
        根据错误反馈修正 SQL (Self-Correction)

        Args:
            original_sql: 原始 SQL
            error_message: 错误信息
            schema: 数据库 Schema
            database: 数据库名称
            hint: 额外的修正提示

        Returns:
            修正后的 SQL 候选
        """
        connector = self.db_manager.get(database)
        dialect = connector.db_type

        # 尝试使用 DSPy 修正
        if HAS_DSPY and self._dspy_configured:
            try:
                result = self.refine_module(
                    original_sql=original_sql,
                    error_message=error_message,
                    schema=schema,
                )
                return SQLCandidate(
                    id="dspy_refined",
                    sql=result.sql,
                    explanation=getattr(result, 'fix_reason', 'DSPy refinement'),
                    confidence=0.70,
                    strategy="refinement",
                )
            except Exception:
                pass

        # 回退到直接 LLM 调用
        messages = [
            {
                "role": "system",
                "content": f"""你是一个 SQL 专家。请根据错误信息修正 SQL 查询。

## 数据库 Schema
{schema}

## SQL 方言
{dialect.upper()}

## 要求
1. 分析错误原因
2. 修正 SQL 语法或逻辑错误
3. 确保修正后的 SQL 语法正确

## 输出格式
```json
{{
    "sql": "修正后的 SQL",
    "explanation": "修正说明",
    "fix_reason": "错误原因分析"
}}
```
""",
            },
            {
                "role": "user",
                "content": f"""原始 SQL:
```sql
{original_sql}
```

错误信息:
{error_message}
{f'''
额外提示:
{hint}
''' if hint else ''}
请修正这个 SQL。""",
            },
        ]

        response = await self.llm.chat(messages)
        result = self.llm._parse_sql_response(response)

        return SQLCandidate(
            id="llm_refined",
            sql=result.get("sql", original_sql),
            explanation=result.get("explanation", "SQL 修正"),
            confidence=0.65,
            strategy="refinement",
        )

    async def run(self, state: DataPilotState) -> dict[str, Any]:
        """
        执行 Logic Architect Agent

        Args:
            state: 当前工作流状态

        Returns:
            状态更新 (包含多路候选 SQL)
        """
        query = state["query"]
        schema = state.get("schema_context", "")
        database = state.get("database", "default")
        value_mappings = state.get("value_mappings", {})

        # 如果没有 Schema，先获取
        if not schema:
            connector = self.db_manager.get(database)
            schema = await connector.get_schema()

        # 检查是否是重试（有错误上下文）
        error_context = state.get("error_context")
        existing_candidates = state.get("candidates", [])

        if error_context and existing_candidates:
            # 修正模式 (Self-Correction)
            last_sql = existing_candidates[-1]["sql"] if existing_candidates else ""
            candidate = await self.refine_sql(
                original_sql=last_sql,
                error_message=error_context,
                schema=schema,
                database=database,
            )
            new_candidates = existing_candidates + [candidate]
        else:
            # 正常生成模式 (多路候选)
            candidates = await self.generate_sql(
                query=query,
                schema=schema,
                database=database,
                value_mappings=value_mappings,
            )
            new_candidates = existing_candidates + candidates

        # 选择最佳候选 (按置信度排序)
        best_candidate = max(new_candidates, key=lambda c: c.get("confidence", 0) if isinstance(c, dict) else c.confidence)
        winner_sql = best_candidate.get("sql") if isinstance(best_candidate, dict) else best_candidate.sql

        return {
            "candidates": new_candidates,
            "winner_sql": winner_sql,
            "schema_context": schema,
            "current_agent": "logic_architect",
            "next_agent": "judge",
            "error_context": None,  # 清除错误上下文
        }


# 创建 LangGraph 节点函数
async def logic_architect_node(state: DataPilotState) -> dict[str, Any]:
    """LangGraph 节点：Logic Architect"""
    architect = LogicArchitect()
    return await architect.run(state)


# 导出
__all__ = [
    "LogicArchitect",
    "OptimizedModuleLoader",
    "logic_architect_node",
]
