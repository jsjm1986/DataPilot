# -*- coding: utf-8 -*-
"""
分步配置生成器

核心理念:
1. 智能检测（不需要 LLM）：时间字段、枚举字段
2. LLM 生成（分步）：业务词汇、实体类型、歧义规则
3. 每步独立，支持增量生成
4. 实时进度反馈
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from .metadata_config import (
    TimeFieldConfig,
    BusinessTermConfig,
    EnumFieldConfig,
    EntityTypeConfig,
    AmbiguityRuleConfig,
    DatabaseMetadataConfig,
    get_config_store,
)
from .schema_introspector import SchemaIntrospector, SchemaMetadata, TableInfo
from ..llm.deepseek import get_deepseek_client


class GenerateStep(str, Enum):
    """生成步骤"""
    INIT = "init"
    TIME_FIELDS = "time_fields"
    ENUM_FIELDS = "enum_fields"
    BUSINESS_TERMS = "business_terms"
    ENTITY_TYPES = "entity_types"
    AMBIGUITY_RULES = "ambiguity_rules"
    COMPLETE = "complete"


@dataclass
class StepResult:
    """步骤结果"""
    step: str
    success: bool
    count: int = 0
    data: list = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class GenerateTask:
    """生成任务"""
    task_id: str
    database: str
    status: str = "pending"  # pending, running, completed, failed
    current_step: str = GenerateStep.INIT
    progress: int = 0  # 0-100
    steps: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now


# 任务存储（内存级，生产环境可改为 Redis）
_tasks: dict[str, GenerateTask] = {}


def get_task(task_id: str) -> Optional[GenerateTask]:
    """获取任务"""
    return _tasks.get(task_id)


def create_task(database: str) -> GenerateTask:
    """创建任务"""
    task_id = str(uuid.uuid4())[:8]
    task = GenerateTask(task_id=task_id, database=database)
    _tasks[task_id] = task
    return task


def update_task(task: GenerateTask):
    """更新任务"""
    task.updated_at = datetime.utcnow().isoformat()
    _tasks[task.task_id] = task


class SmartDetector:
    """
    智能检测器 - 不需要 LLM

    基于数据类型和模式识别自动检测：
    - 时间字段
    - 枚举字段
    """

    # 时间类型
    TIME_TYPES = ['date', 'datetime', 'timestamp', 'time', 'timestamptz']

    # 时间字段名模式
    TIME_PATTERNS = [
        'created_at', 'updated_at', 'deleted_at', 'modified_at',
        'create_time', 'update_time', 'delete_time', 'modify_time',
        'created', 'updated', 'deleted', 'modified',
        '_at', '_time', '_date', 'date', 'time',
    ]

    def detect_time_fields(self, metadata: SchemaMetadata) -> list[TimeFieldConfig]:
        """检测时间字段"""
        time_fields = []

        for table in metadata.tables:
            for col in table.columns:
                is_time = False

                # 1. 根据数据类型判断
                type_lower = col.data_type.lower()
                if any(t in type_lower for t in self.TIME_TYPES):
                    is_time = True

                # 2. 根据列名模式判断
                name_lower = col.name.lower()
                if any(p in name_lower for p in self.TIME_PATTERNS):
                    is_time = True

                if is_time:
                    # 判断是否是主时间字段
                    is_primary = name_lower in ['created_at', 'create_time', 'created', 'date']

                    # 确定字段类型
                    if 'timestamp' in type_lower:
                        field_type = 'timestamp'
                    elif 'datetime' in type_lower:
                        field_type = 'datetime'
                    else:
                        field_type = 'date'

                    # 生成描述
                    description = self._generate_time_description(col.name, table.name)

                    time_fields.append(TimeFieldConfig(
                        table_name=table.name,
                        column_name=col.name,
                        field_type=field_type,
                        description=description,
                        format_hint="YYYY-MM-DD HH:mm:ss" if field_type != 'date' else "YYYY-MM-DD",
                        is_primary_time=is_primary,
                        auto_detected=True,
                        user_confirmed=False,
                    ))

        return time_fields

    def _generate_time_description(self, col_name: str, table_name: str) -> str:
        """生成时间字段描述"""
        name_lower = col_name.lower()

        if 'created' in name_lower or 'create' in name_lower:
            return f"{table_name} 创建时间"
        elif 'updated' in name_lower or 'update' in name_lower or 'modified' in name_lower:
            return f"{table_name} 更新时间"
        elif 'deleted' in name_lower or 'delete' in name_lower:
            return f"{table_name} 删除时间"
        elif 'paid' in name_lower:
            return "支付时间"
        elif 'shipped' in name_lower:
            return "发货时间"
        elif 'completed' in name_lower:
            return "完成时间"
        elif name_lower == 'date':
            return f"{table_name} 日期"
        else:
            return f"{table_name} {col_name}"

    def detect_enum_fields(self, metadata: SchemaMetadata) -> list[EnumFieldConfig]:
        """检测枚举字段"""
        enum_fields = []

        for table in metadata.tables:
            for col in table.columns:
                # 跳过时间字段和主键
                if col.is_time_field or col.is_primary_key:
                    continue

                # 跳过数值类型（除非是小范围整数）
                type_lower = col.data_type.lower()
                if any(t in type_lower for t in ['int', 'float', 'decimal', 'numeric', 'double']):
                    # 只有当 distinct count 很小时才考虑
                    if col.distinct_count and col.distinct_count > 10:
                        continue

                # 检查是否像枚举（distinct count <= 50）
                if col.is_enum_like and col.sample_values:
                    # 生成值描述
                    value_descriptions = self._generate_enum_descriptions(
                        col.name, col.sample_values
                    )

                    enum_fields.append(EnumFieldConfig(
                        table_name=table.name,
                        column_name=col.name,
                        values=col.sample_values,
                        value_descriptions=value_descriptions,
                        display_names={},
                        auto_detected=True,
                        user_confirmed=False,
                    ))

        return enum_fields

    def _generate_enum_descriptions(self, col_name: str, values: list) -> dict[str, str]:
        """生成枚举值描述"""
        descriptions = {}

        # 常见状态映射
        status_map = {
            'pending': '待处理',
            'processing': '处理中',
            'completed': '已完成',
            'cancelled': '已取消',
            'failed': '失败',
            'success': '成功',
            'active': '活跃',
            'inactive': '不活跃',
            'enabled': '启用',
            'disabled': '禁用',
            'paid': '已支付',
            'unpaid': '未支付',
            'shipped': '已发货',
            'delivered': '已送达',
            'refunded': '已退款',
            'normal': '普通',
            'vip': 'VIP',
            'gold': '黄金',
            'silver': '白银',
            'platinum': '铂金',
            'IN': '入库',
            'OUT': '出库',
            'ADJUST': '调整',
            'view': '浏览',
            'cart': '加购',
            'favorite': '收藏',
            'purchase': '购买',
            'wechat': '微信支付',
            'alipay': '支付宝',
            'credit_card': '信用卡',
            'debit_card': '借记卡',
        }

        for val in values:
            if val is None:
                continue
            val_str = str(val).lower()
            if val_str in status_map:
                descriptions[str(val)] = status_map[val_str]

        return descriptions


class LLMConfigGenerator:
    """
    LLM 配置生成器 - 分步生成

    每个配置类型使用专门的提示词，提高质量
    """

    def __init__(self):
        self.llm = get_deepseek_client()

    async def generate_business_terms(
        self,
        schema_info: str,
        existing_time_fields: list[TimeFieldConfig],
        existing_enum_fields: list[EnumFieldConfig],
    ) -> list[BusinessTermConfig]:
        """生成业务词汇"""

        # 构建上下文
        context = f"""## 数据库结构
{schema_info}

## 已识别的时间字段
{json.dumps([{"table": tf.table_name, "column": tf.column_name} for tf in existing_time_fields], ensure_ascii=False)}

## 已识别的枚举字段
{json.dumps([{"table": ef.table_name, "column": ef.column_name, "values": ef.values} for ef in existing_enum_fields], ensure_ascii=False)}
"""

        prompt = f"""你是一个数据库分析专家。根据数据库结构，提取业务词汇配置。

{context}

## 任务
分析数据库结构，提取业务词汇（如"销量"、"销售额"、"客户数"等）。

## 输出格式
返回 JSON 数组：
```json
[
    {{
        "term": "销量",
        "synonyms": ["销售量", "卖出数量"],
        "mapped_table": "order_items",
        "mapped_column": "quantity",
        "aggregation": "SUM",
        "description": "商品销售数量"
    }}
]
```

## 要求
- 只返回 JSON 数组，不要其他内容
- 业务词汇要符合中文习惯
- aggregation 可选值: SUM, COUNT, AVG, MAX, MIN
- 每个词汇必须有明确的表和列映射
"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请提取业务词汇配置。"},
        ]

        try:
            response = await self.llm.chat(messages)
            data = self._parse_json_array(response)

            terms = []
            for item in data:
                if isinstance(item, dict) and "term" in item:
                    item.setdefault("synonyms", [])
                    item.setdefault("mapped_table", "")
                    item.setdefault("mapped_column", "")
                    item.setdefault("aggregation", "")
                    item.setdefault("description", "")
                    item.setdefault("auto_detected", True)
                    item.setdefault("user_confirmed", False)
                    terms.append(BusinessTermConfig(**item))

            return terms
        except Exception as e:
            print(f"Generate business terms error: {e}")
            return []

    async def generate_entity_types(
        self,
        schema_info: str,
    ) -> list[EntityTypeConfig]:
        """生成实体类型"""

        prompt = f"""你是一个数据库分析专家。根据数据库结构，识别主要的业务实体类型。

## 数据库结构
{schema_info}

## 任务
识别数据库中的主要业务实体（如产品、客户、订单等）。

## 输出格式
返回 JSON 数组：
```json
[
    {{
        "entity_type": "product",
        "display_name": "产品",
        "primary_table": "products",
        "name_column": "name",
        "id_column": "id",
        "search_columns": ["name", "brand"]
    }}
]
```

## 要求
- 只返回 JSON 数组，不要其他内容
- 每个实体必须有主表和名称列
- search_columns 用于搜索时匹配的列
"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请识别业务实体类型。"},
        ]

        try:
            response = await self.llm.chat(messages)
            data = self._parse_json_array(response)

            entities = []
            for item in data:
                if isinstance(item, dict) and "entity_type" in item:
                    item.setdefault("display_name", item["entity_type"])
                    item.setdefault("primary_table", "")
                    item.setdefault("name_column", "name")
                    item.setdefault("id_column", "id")
                    item.setdefault("search_columns", [])
                    item.setdefault("description", "")
                    item.setdefault("auto_detected", True)
                    item.setdefault("user_confirmed", False)
                    entities.append(EntityTypeConfig(**item))

            return entities
        except Exception as e:
            print(f"Generate entity types error: {e}")
            return []

    async def generate_ambiguity_rules(
        self,
        existing_time_fields: list[TimeFieldConfig],
        existing_enum_fields: list[EnumFieldConfig],
    ) -> list[AmbiguityRuleConfig]:
        """生成歧义规则"""

        # 基于已有配置生成规则
        rules = []

        # 1. 时间范围歧义规则（如果有时间字段）
        if existing_time_fields:
            rules.append(AmbiguityRuleConfig(
                rule_id="time_range",
                rule_type="time",
                trigger_keywords=["最近", "近期", "过去", "之前", "本月", "本周", "今年"],
                question_template="您希望查询哪个时间段的数据？",
                options_source="static",
                static_options=["今天", "昨天", "最近7天", "最近30天", "本月", "本季度", "今年", "全部历史"],
                enabled=True,
                auto_detected=True,
                user_confirmed=False,
            ))

        # 2. 基于枚举字段生成规则
        for ef in existing_enum_fields:
            if ef.column_name.lower() in ['status', 'type', 'category']:
                rules.append(AmbiguityRuleConfig(
                    rule_id=f"{ef.table_name}_{ef.column_name}",
                    rule_type="filter",
                    trigger_keywords=[ef.column_name],
                    question_template=f"您想筛选哪种 {ef.column_name}？",
                    options_source="static",
                    static_options=ef.values[:10],  # 最多 10 个选项
                    enabled=True,
                    auto_detected=True,
                    user_confirmed=False,
                ))

        return rules

    def _parse_json_array(self, response: str) -> list:
        """解析 JSON 数组"""
        import re

        if not response:
            return []

        # 移除 markdown 代码块
        cleaned = response
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)

        cleaned = cleaned.strip()

        # 尝试提取 JSON 数组
        array_match = re.search(r'\[[\s\S]*\]', cleaned)
        if array_match:
            try:
                result = json.loads(array_match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return []


class StepByStepGenerator:
    """
    分步配置生成器

    协调智能检测和 LLM 生成，提供进度回调
    """

    def __init__(self):
        self.detector = SmartDetector()
        self.llm_generator = LLMConfigGenerator()

    async def generate(
        self,
        task: GenerateTask,
        connector,
        progress_callback: Optional[Callable[[GenerateTask], None]] = None,
    ) -> DatabaseMetadataConfig:
        """
        分步生成配置

        Args:
            task: 生成任务
            connector: 数据库连接器
            progress_callback: 进度回调函数

        Returns:
            生成的配置
        """
        task.status = "running"
        update_task(task)

        if progress_callback:
            progress_callback(task)

        try:
            # Step 1: 获取 Schema 信息
            task.current_step = GenerateStep.INIT
            task.progress = 5
            update_task(task)
            if progress_callback:
                progress_callback(task)

            introspector = SchemaIntrospector(connector)
            metadata = await introspector.introspect(
                include_samples=True,
                sample_limit=20,
                include_row_counts=True,
            )
            schema_info = introspector.generate_llm_context(metadata)

            # Step 2: 检测时间字段（不需要 LLM）
            task.current_step = GenerateStep.TIME_FIELDS
            task.progress = 20
            update_task(task)
            if progress_callback:
                progress_callback(task)

            start_time = datetime.utcnow()
            time_fields = self.detector.detect_time_fields(metadata)
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            task.steps[GenerateStep.TIME_FIELDS] = asdict(StepResult(
                step=GenerateStep.TIME_FIELDS,
                success=True,
                count=len(time_fields),
                data=[asdict(tf) for tf in time_fields],
                duration_ms=duration,
            ))
            update_task(task)
            if progress_callback:
                progress_callback(task)

            # Step 3: 检测枚举字段（不需要 LLM）
            task.current_step = GenerateStep.ENUM_FIELDS
            task.progress = 35
            update_task(task)
            if progress_callback:
                progress_callback(task)

            start_time = datetime.utcnow()
            enum_fields = self.detector.detect_enum_fields(metadata)
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            task.steps[GenerateStep.ENUM_FIELDS] = asdict(StepResult(
                step=GenerateStep.ENUM_FIELDS,
                success=True,
                count=len(enum_fields),
                data=[asdict(ef) for ef in enum_fields],
                duration_ms=duration,
            ))
            update_task(task)
            if progress_callback:
                progress_callback(task)

            # Step 4: 生成业务词汇（需要 LLM）
            task.current_step = GenerateStep.BUSINESS_TERMS
            task.progress = 50
            update_task(task)
            if progress_callback:
                progress_callback(task)

            start_time = datetime.utcnow()
            try:
                business_terms = await self.llm_generator.generate_business_terms(
                    schema_info, time_fields, enum_fields
                )
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                task.steps[GenerateStep.BUSINESS_TERMS] = asdict(StepResult(
                    step=GenerateStep.BUSINESS_TERMS,
                    success=True,
                    count=len(business_terms),
                    data=[asdict(bt) for bt in business_terms],
                    duration_ms=duration,
                ))
            except Exception as e:
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                task.steps[GenerateStep.BUSINESS_TERMS] = asdict(StepResult(
                    step=GenerateStep.BUSINESS_TERMS,
                    success=False,
                    error=str(e),
                    duration_ms=duration,
                ))
                business_terms = []

            update_task(task)
            if progress_callback:
                progress_callback(task)

            # Step 5: 生成实体类型（需要 LLM）
            task.current_step = GenerateStep.ENTITY_TYPES
            task.progress = 70
            update_task(task)
            if progress_callback:
                progress_callback(task)

            start_time = datetime.utcnow()
            try:
                entity_types = await self.llm_generator.generate_entity_types(schema_info)
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                task.steps[GenerateStep.ENTITY_TYPES] = asdict(StepResult(
                    step=GenerateStep.ENTITY_TYPES,
                    success=True,
                    count=len(entity_types),
                    data=[asdict(et) for et in entity_types],
                    duration_ms=duration,
                ))
            except Exception as e:
                duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                task.steps[GenerateStep.ENTITY_TYPES] = asdict(StepResult(
                    step=GenerateStep.ENTITY_TYPES,
                    success=False,
                    error=str(e),
                    duration_ms=duration,
                ))
                entity_types = []

            update_task(task)
            if progress_callback:
                progress_callback(task)

            # Step 6: 生成歧义规则（基于已有配置，不需要 LLM）
            task.current_step = GenerateStep.AMBIGUITY_RULES
            task.progress = 90
            update_task(task)
            if progress_callback:
                progress_callback(task)

            start_time = datetime.utcnow()
            ambiguity_rules = await self.llm_generator.generate_ambiguity_rules(
                time_fields, enum_fields
            )
            duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            task.steps[GenerateStep.AMBIGUITY_RULES] = asdict(StepResult(
                step=GenerateStep.AMBIGUITY_RULES,
                success=True,
                count=len(ambiguity_rules),
                data=[asdict(ar) for ar in ambiguity_rules],
                duration_ms=duration,
            ))

            # 完成
            task.current_step = GenerateStep.COMPLETE
            task.progress = 100
            task.status = "completed"
            update_task(task)
            if progress_callback:
                progress_callback(task)

            # 构建配置（不保存，等用户确认后再保存）
            config = DatabaseMetadataConfig(
                database_name=task.database,
                database_type=metadata.dialect,
                description=f"AI 自动生成的配置 ({datetime.utcnow().strftime('%Y-%m-%d %H:%M')})",
                time_fields=time_fields,
                business_terms=business_terms,
                enum_fields=enum_fields,
                entity_types=entity_types,
                ambiguity_rules=ambiguity_rules,
                auto_generated=True,
            )

            # 注意：这里不保存配置，等用户在预览页面确认后再保存
            # 用户点击"确认使用"时会调用 /api/v1/config/generate/save 保存

            return config

        except Exception as e:
            import traceback
            print(f"Step-by-step generation error: {e}")
            print(traceback.format_exc())

            task.status = "failed"
            task.error = str(e)
            update_task(task)
            if progress_callback:
                progress_callback(task)

            raise


# 导出
__all__ = [
    "GenerateStep",
    "StepResult",
    "GenerateTask",
    "SmartDetector",
    "LLMConfigGenerator",
    "StepByStepGenerator",
    "get_task",
    "create_task",
    "update_task",
]
