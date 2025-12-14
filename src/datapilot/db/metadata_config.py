# -*- coding: utf-8 -*-
"""
元数据配置系统 - 可配置的数据库元信息

核心理念:
1. 所有配置存储在数据库中，而非硬编码
2. AI 可以自动分析数据库并生成初始配置
3. 用户可以查看和修改配置
4. 不同数据库有完全不同的配置

配置类型:
- 时间字段配置: 哪些字段是时间字段，格式是什么
- 业务词汇配置: 业务术语到数据库字段的映射
- 枚举字段配置: 哪些字段是枚举，有哪些可选值
- 实体类型配置: 数据库中有哪些实体类型
- 歧义规则配置: 针对该数据库的歧义检测规则
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime
from pathlib import Path

from ..llm.deepseek import get_deepseek_client


# ============================================
# 配置数据模型
# ============================================

@dataclass
class TimeFieldConfig:
    """时间字段配置"""
    table_name: str
    column_name: str
    field_type: str  # date, datetime, timestamp
    description: str  # 字段用途描述
    format_hint: str = ""  # 格式提示，如 "YYYY-MM-DD"
    is_primary_time: bool = False  # 是否是主要时间字段（用于默认时间过滤）
    auto_detected: bool = True  # 是否是自动检测的
    user_confirmed: bool = False  # 用户是否确认过


@dataclass
class BusinessTermConfig:
    """业务词汇配置"""
    term: str  # 业务术语，如 "销量"
    synonyms: list[str] = field(default_factory=list)  # 同义词，如 ["销售量", "卖出数量"]
    mapped_table: str = ""  # 映射到的表
    mapped_column: str = ""  # 映射到的列
    aggregation: str = ""  # 聚合方式，如 "SUM", "COUNT"
    description: str = ""  # 描述
    auto_detected: bool = True
    user_confirmed: bool = False


@dataclass
class EnumFieldConfig:
    """枚举字段配置"""
    table_name: str
    column_name: str
    values: list[str] = field(default_factory=list)  # 可选值列表
    value_descriptions: dict[str, str] = field(default_factory=dict)  # 值的描述
    display_names: dict[str, str] = field(default_factory=dict)  # 显示名称映射
    auto_detected: bool = True
    user_confirmed: bool = False


@dataclass
class EntityTypeConfig:
    """实体类型配置"""
    entity_type: str  # 实体类型，如 "product", "customer"
    display_name: str  # 显示名称，如 "产品", "客户"
    primary_table: str  # 主表
    name_column: str  # 名称列
    id_column: str = "id"  # ID 列
    search_columns: list[str] = field(default_factory=list)  # 搜索列
    description: str = ""
    auto_detected: bool = True
    user_confirmed: bool = False


@dataclass
class AmbiguityRuleConfig:
    """歧义规则配置"""
    rule_id: str
    rule_type: str  # time, metric, scope, granularity, entity
    trigger_keywords: list[str] = field(default_factory=list)  # 触发关键词
    question_template: str = ""  # 澄清问题模板
    options_source: str = ""  # 选项来源，如 "static", "dynamic", "llm"
    static_options: list[str] = field(default_factory=list)  # 静态选项
    dynamic_query: str = ""  # 动态查询 SQL
    enabled: bool = True
    auto_detected: bool = True
    user_confirmed: bool = False


@dataclass
class DatabaseMetadataConfig:
    """数据库完整元数据配置"""
    database_name: str
    database_type: str  # mysql, postgresql, sqlite
    description: str = ""
    time_fields: list[TimeFieldConfig] = field(default_factory=list)
    business_terms: list[BusinessTermConfig] = field(default_factory=list)
    enum_fields: list[EnumFieldConfig] = field(default_factory=list)
    entity_types: list[EntityTypeConfig] = field(default_factory=list)
    ambiguity_rules: list[AmbiguityRuleConfig] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    auto_generated: bool = True

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now


# ============================================
# 配置存储管理器
# ============================================

class MetadataConfigStore:
    """
    元数据配置存储

    使用 SQLite 存储配置，支持:
    - 保存/加载配置
    - 版本历史
    - 导入/导出
    """

    def __init__(self, db_path: str = "data/metadata_config.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 主配置表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS database_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                database_name TEXT UNIQUE NOT NULL,
                database_type TEXT NOT NULL,
                description TEXT,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1
            )
        """)

        # 配置历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                database_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                changed_by TEXT,
                change_reason TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def save_config(
        self,
        config: DatabaseMetadataConfig,
        changed_by: str = "system",
        change_reason: str = "",
    ):
        """保存配置"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        config.updated_at = datetime.utcnow().isoformat()
        config_json = json.dumps(asdict(config), ensure_ascii=False, default=str)

        # 检查是否存在
        cursor.execute(
            "SELECT id, version FROM database_configs WHERE database_name = ?",
            (config.database_name,)
        )
        existing = cursor.fetchone()

        if existing:
            # 更新
            new_version = existing[1] + 1
            cursor.execute("""
                UPDATE database_configs
                SET config_json = ?, updated_at = ?, version = ?
                WHERE database_name = ?
            """, (config_json, config.updated_at, new_version, config.database_name))
        else:
            # 插入
            cursor.execute("""
                INSERT INTO database_configs
                (database_name, database_type, description, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                config.database_name,
                config.database_type,
                config.description,
                config_json,
                config.created_at,
                config.updated_at,
            ))

        # 保存历史
        cursor.execute("""
            INSERT INTO config_history
            (database_name, config_json, changed_by, change_reason, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            config.database_name,
            config_json,
            changed_by,
            change_reason,
            datetime.utcnow().isoformat(),
        ))

        conn.commit()
        conn.close()

    def load_config(self, database_name: str) -> Optional[DatabaseMetadataConfig]:
        """加载配置"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT config_json FROM database_configs WHERE database_name = ?",
            (database_name,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            data = json.loads(row[0])
            return self._dict_to_config(data)
        return None

    def _dict_to_config(self, data: dict) -> DatabaseMetadataConfig:
        """将字典转换为配置对象"""
        return DatabaseMetadataConfig(
            database_name=data.get("database_name", ""),
            database_type=data.get("database_type", ""),
            description=data.get("description", ""),
            time_fields=[TimeFieldConfig(**tf) for tf in data.get("time_fields", [])],
            business_terms=[BusinessTermConfig(**bt) for bt in data.get("business_terms", [])],
            enum_fields=[EnumFieldConfig(**ef) for ef in data.get("enum_fields", [])],
            entity_types=[EntityTypeConfig(**et) for et in data.get("entity_types", [])],
            ambiguity_rules=[AmbiguityRuleConfig(**ar) for ar in data.get("ambiguity_rules", [])],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            auto_generated=data.get("auto_generated", True),
        )

    def list_databases(self) -> list[dict]:
        """列出所有已配置的数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT database_name, database_type, description, updated_at, version
            FROM database_configs
            ORDER BY updated_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "database_name": row[0],
                "database_type": row[1],
                "description": row[2],
                "updated_at": row[3],
                "version": row[4],
            }
            for row in rows
        ]

    def get_history(self, database_name: str, limit: int = 10) -> list[dict]:
        """获取配置历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT config_json, changed_by, change_reason, created_at
            FROM config_history
            WHERE database_name = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (database_name, limit))
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "config": json.loads(row[0]),
                "changed_by": row[1],
                "change_reason": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    def export_config(self, database_name: str) -> str:
        """导出配置为 JSON"""
        config = self.load_config(database_name)
        if config:
            return json.dumps(asdict(config), ensure_ascii=False, indent=2)
        return ""

    def import_config(self, json_str: str, changed_by: str = "import"):
        """从 JSON 导入配置"""
        data = json.loads(json_str)
        config = self._dict_to_config(data)
        self.save_config(config, changed_by=changed_by, change_reason="Imported from JSON")


# ============================================
# AI 配置生成器
# ============================================

CONFIG_GENERATION_PROMPT = """你是一个数据库分析专家。根据提供的数据库 Schema 信息，生成元数据配置。

## 数据库信息
{schema_info}

## 任务
分析数据库结构，生成以下配置:

1. **时间字段配置**: 识别所有时间相关的字段
2. **业务词汇配置**: 根据表名和列名推断业务术语
3. **枚举字段配置**: 识别状态、类型等枚举字段
4. **实体类型配置**: 识别主要的业务实体
5. **歧义规则配置**: 针对该数据库可能出现的查询歧义

## 输出格式
返回 JSON:
```json
{{
    "time_fields": [
        {{
            "table_name": "表名",
            "column_name": "列名",
            "field_type": "datetime",
            "description": "字段用途",
            "format_hint": "YYYY-MM-DD HH:mm:ss",
            "is_primary_time": true
        }}
    ],
    "business_terms": [
        {{
            "term": "销量",
            "synonyms": ["销售量", "卖出数量"],
            "mapped_table": "order_items",
            "mapped_column": "quantity",
            "aggregation": "SUM",
            "description": "商品销售数量"
        }}
    ],
    "enum_fields": [
        {{
            "table_name": "orders",
            "column_name": "status",
            "values": ["pending", "paid", "shipped", "completed"],
            "value_descriptions": {{
                "pending": "待支付",
                "paid": "已支付",
                "shipped": "已发货",
                "completed": "已完成"
            }}
        }}
    ],
    "entity_types": [
        {{
            "entity_type": "product",
            "display_name": "产品",
            "primary_table": "products",
            "name_column": "name",
            "id_column": "id",
            "search_columns": ["name", "brand"]
        }}
    ],
    "ambiguity_rules": [
        {{
            "rule_id": "time_range",
            "rule_type": "time",
            "trigger_keywords": ["最近", "近期", "之前"],
            "question_template": "您希望查询哪个时间段的数据？",
            "options_source": "static",
            "static_options": ["今天", "最近7天", "最近30天", "本月", "全部历史"]
        }}
    ]
}}
```

## 注意
- 根据实际的表结构和列名生成配置
- 业务词汇要符合中文习惯
- 枚举值要包含实际的数据库值
- 实体类型要覆盖主要的业务对象
"""


class AIConfigGenerator:
    """
    AI 配置生成器

    使用 LLM 分析数据库结构，自动生成初始配置
    """

    def __init__(self):
        self.llm = get_deepseek_client()

    async def generate_config(
        self,
        database_name: str,
        database_type: str,
        schema_info: str,
        sample_data: dict = None,
    ) -> DatabaseMetadataConfig:
        """
        根据 Schema 信息生成配置

        Args:
            database_name: 数据库名称
            database_type: 数据库类型
            schema_info: Schema 信息（DDL 或结构描述）
            sample_data: 样本数据（可选，帮助识别枚举值）

        Returns:
            生成的配置
        """
        print(f"[AIConfigGenerator] ========== START generate_config ==========")
        print(f"[AIConfigGenerator] database_name: {database_name}")
        print(f"[AIConfigGenerator] database_type: {database_type}")
        print(f"[AIConfigGenerator] schema_info length: {len(schema_info) if schema_info else 0}")
        print(f"[AIConfigGenerator] sample_data keys: {list(sample_data.keys()) if sample_data else 'None'}")

        # 构建提示
        full_schema = schema_info
        if sample_data:
            full_schema += "\n\n## 样本数据\n"
            for table, samples in sample_data.items():
                full_schema += f"\n### {table}\n"
                full_schema += json.dumps(samples[:3], ensure_ascii=False, indent=2)

        prompt = CONFIG_GENERATION_PROMPT.format(schema_info=full_schema)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请分析数据库结构并生成配置。"},
        ]

        try:
            print(f"[AIConfigGenerator] Calling LLM for config generation...")
            response = await self.llm.chat(messages)
            print(f"[AIConfigGenerator] LLM response length: {len(response) if response else 0}")
            if response:
                print(f"[AIConfigGenerator] LLM response preview: {response[:500]}...")
            config_data = self._parse_response(response)
            print(f"[AIConfigGenerator] Parsed config_data keys: {list(config_data.keys()) if config_data else 'None'}")

            # 验证并过滤有效的配置项
            time_fields = []
            for tf in config_data.get("time_fields", []):
                if isinstance(tf, dict) and "table_name" in tf and "column_name" in tf:
                    try:
                        # 设置默认值，避免重复参数
                        tf.setdefault("auto_detected", True)
                        tf.setdefault("user_confirmed", False)
                        time_fields.append(TimeFieldConfig(**tf))
                    except Exception as e:
                        print(f"Skip invalid time_field: {tf}, error: {e}")

            business_terms = []
            for bt in config_data.get("business_terms", []):
                if isinstance(bt, dict) and "term" in bt:
                    try:
                        bt.setdefault("auto_detected", True)
                        bt.setdefault("user_confirmed", False)
                        business_terms.append(BusinessTermConfig(**bt))
                    except Exception as e:
                        print(f"Skip invalid business_term: {bt}, error: {e}")

            enum_fields = []
            for ef in config_data.get("enum_fields", []):
                if isinstance(ef, dict) and "table_name" in ef and "column_name" in ef:
                    try:
                        ef.setdefault("auto_detected", True)
                        ef.setdefault("user_confirmed", False)
                        enum_fields.append(EnumFieldConfig(**ef))
                    except Exception as e:
                        print(f"Skip invalid enum_field: {ef}, error: {e}")

            entity_types = []
            for et in config_data.get("entity_types", []):
                if isinstance(et, dict) and "entity_type" in et:
                    try:
                        et.setdefault("auto_detected", True)
                        et.setdefault("user_confirmed", False)
                        entity_types.append(EntityTypeConfig(**et))
                    except Exception as e:
                        print(f"Skip invalid entity_type: {et}, error: {e}")

            ambiguity_rules = []
            for ar in config_data.get("ambiguity_rules", []):
                if isinstance(ar, dict) and "rule_id" in ar:
                    try:
                        ar.setdefault("auto_detected", True)
                        ar.setdefault("user_confirmed", False)
                        ar.setdefault("enabled", True)
                        ambiguity_rules.append(AmbiguityRuleConfig(**ar))
                    except Exception as e:
                        print(f"Skip invalid ambiguity_rule: {ar}, error: {e}")

            return DatabaseMetadataConfig(
                database_name=database_name,
                database_type=database_type,
                description=f"AI 自动生成的配置 ({datetime.utcnow().strftime('%Y-%m-%d')})",
                time_fields=time_fields,
                business_terms=business_terms,
                enum_fields=enum_fields,
                entity_types=entity_types,
                ambiguity_rules=ambiguity_rules,
                auto_generated=True,
            )
        except Exception as e:
            import traceback
            print(f"AI config generation failed: {e}")
            print(traceback.format_exc())
            # 返回空配置
            return DatabaseMetadataConfig(
                database_name=database_name,
                database_type=database_type,
                description="配置生成失败，请手动配置",
                auto_generated=False,
            )

    def _parse_response(self, response: str) -> dict:
        """解析 LLM 响应"""
        import re

        if not response:
            print("[_parse_response] Empty response")
            return {}

        print(f"[_parse_response] Response length: {len(response)}")

        # 先移除 markdown 代码块标记
        cleaned = response
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)

        # 清理可能的前导/尾随空白
        cleaned = cleaned.strip()

        # 尝试提取 JSON 对象
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            json_str = json_match.group()
            print(f"[_parse_response] Found JSON, length: {len(json_str)}")
            try:
                result = json.loads(json_str)
                # 验证结果是字典
                if isinstance(result, dict):
                    print(f"[_parse_response] Parsed successfully, keys: {list(result.keys())}")
                    return result
                else:
                    print(f"[_parse_response] Result is not a dict: {type(result)}")
            except json.JSONDecodeError as e:
                print(f"[_parse_response] JSON parse error: {e}")
                print(f"[_parse_response] JSON string preview: {json_str[:300]}...")
        else:
            print(f"[_parse_response] No JSON found in response")
            print(f"[_parse_response] Response preview: {cleaned[:300]}...")

        return {}


# ============================================
# 全局实例
# ============================================

_config_store: Optional[MetadataConfigStore] = None
_ai_generator: Optional[AIConfigGenerator] = None


def get_config_store() -> MetadataConfigStore:
    """获取配置存储实例"""
    global _config_store
    if _config_store is None:
        _config_store = MetadataConfigStore()
    return _config_store


def get_ai_generator() -> AIConfigGenerator:
    """获取 AI 生成器实例"""
    global _ai_generator
    if _ai_generator is None:
        _ai_generator = AIConfigGenerator()
    return _ai_generator


__all__ = [
    # 数据模型
    "TimeFieldConfig",
    "BusinessTermConfig",
    "EnumFieldConfig",
    "EntityTypeConfig",
    "AmbiguityRuleConfig",
    "DatabaseMetadataConfig",
    # 存储和生成
    "MetadataConfigStore",
    "AIConfigGenerator",
    # 全局实例
    "get_config_store",
    "get_ai_generator",
]
