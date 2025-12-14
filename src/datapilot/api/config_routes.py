# -*- coding: utf-8 -*-
"""
元数据配置 API 路由

提供配置管理接口:
- 查看/编辑时间字段配置
- 查看/编辑业务词汇配置
- 查看/编辑枚举字段配置
- AI 自动生成配置
- 导入/导出配置
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db.metadata_config import (
    get_config_store,
    get_ai_generator,
    DatabaseMetadataConfig,
    TimeFieldConfig,
    BusinessTermConfig,
    EnumFieldConfig,
    EntityTypeConfig,
    AmbiguityRuleConfig,
)
from ..db.connector import get_db_manager
from ..db.schema_introspector import SchemaIntrospector
from ..db.config_generator import (
    StepByStepGenerator,
    create_task,
    get_task,
    update_task,
    GenerateStep,
)

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])


# ============================================
# 请求/响应模型
# ============================================

class GenerateConfigRequest(BaseModel):
    """生成配置请求"""
    database: str = Field(..., description="数据库名称")
    include_samples: bool = Field(default=True, description="是否包含样本数据")


class UpdateTimeFieldRequest(BaseModel):
    """更新时间字段请求"""
    database: str
    table_name: str
    column_name: str
    field_type: str = "datetime"
    description: str = ""
    format_hint: str = ""
    is_primary_time: bool = False


class UpdateBusinessTermRequest(BaseModel):
    """更新业务词汇请求"""
    database: str
    term: str
    synonyms: list[str] = []
    mapped_table: str = ""
    mapped_column: str = ""
    aggregation: str = ""
    description: str = ""


class UpdateEnumFieldRequest(BaseModel):
    """更新枚举字段请求"""
    database: str
    table_name: str
    column_name: str
    values: list[str] = []
    value_descriptions: dict[str, str] = {}
    display_names: dict[str, str] = {}


class ImportConfigRequest(BaseModel):
    """导入配置请求"""
    config_json: str


class StartGenerateRequest(BaseModel):
    """启动分步生成请求"""
    database: str = Field(..., description="数据库名称")


# ============================================
# 路由定义
# ============================================

@router.get("/databases")
async def list_configured_databases():
    """列出所有已配置的数据库"""
    store = get_config_store()
    databases = store.list_databases()
    return {"databases": databases}


@router.get("/{database}/schema")
async def get_database_schema(database: str):
    """
    获取数据库的表和列信息

    用于前端配置时选择表名和列名
    """
    try:
        db_manager = get_db_manager()
        connector = db_manager.get(database)
        tables = await connector.get_tables()

        return {
            "database": database,
            "tables": [
                {
                    "name": t["name"],
                    "comment": t.get("comment", ""),
                    "columns": [
                        {
                            "name": c["name"],
                            "type": c.get("type", ""),
                            "comment": c.get("comment", ""),
                        }
                        for c in t.get("columns", [])
                    ],
                }
                for t in tables
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 Schema 失败: {str(e)}")


@router.get("/{database}")
async def get_database_config(database: str):
    """获取数据库配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        raise HTTPException(status_code=404, detail=f"配置不存在: {database}")

    return {
        "database": database,
        "config": {
            "database_name": config.database_name,
            "database_type": config.database_type,
            "description": config.description,
            "time_fields": [vars(tf) for tf in config.time_fields],
            "business_terms": [vars(bt) for bt in config.business_terms],
            "enum_fields": [vars(ef) for ef in config.enum_fields],
            "entity_types": [vars(et) for et in config.entity_types],
            "ambiguity_rules": [vars(ar) for ar in config.ambiguity_rules],
            "created_at": config.created_at,
            "updated_at": config.updated_at,
            "auto_generated": config.auto_generated,
        }
    }


@router.post("/generate")
async def generate_config(request: GenerateConfigRequest):
    """
    AI 自动生成数据库配置

    根据数据库 Schema 自动分析并生成:
    - 时间字段配置
    - 业务词汇配置
    - 枚举字段配置
    - 实体类型配置
    - 歧义规则配置
    """
    try:
        # 获取数据库连接
        db_manager = get_db_manager()
        connector = db_manager.get(request.database)

        # 获取 Schema 信息
        introspector = SchemaIntrospector(connector)
        metadata = await introspector.introspect(
            include_samples=request.include_samples,
            sample_limit=20,
            include_row_counts=True,
        )

        # 生成 Schema 描述
        schema_info = introspector.generate_llm_context(metadata)

        # 收集样本数据
        sample_data = {}
        if request.include_samples:
            for table in metadata.tables:
                samples = []
                for col in table.columns:
                    if col.sample_values:
                        samples.append({
                            "column": col.name,
                            "type": col.data_type,
                            "samples": col.sample_values[:5],
                        })
                if samples:
                    sample_data[table.name] = samples

        # AI 生成配置
        generator = get_ai_generator()
        try:
            config = await generator.generate_config(
                database_name=request.database,
                database_type=metadata.dialect,
                schema_info=schema_info,
                sample_data=sample_data,
            )
        except Exception as gen_error:
            import traceback
            error_str = str(gen_error)
            print(f"AI config generation error: {gen_error}")
            print(traceback.format_exc())

            # 检查是否是 API Key 问题
            if "401" in error_str or "authentication" in error_str.lower() or "api_key" in error_str.lower():
                description = "AI 生成失败: DeepSeek API Key 无效或未配置，请检查 .env 文件中的 DEEPSEEK_API_KEY"
            else:
                description = f"AI 生成失败: {error_str[:100]}，请手动配置"

            # 返回空配置而不是失败
            config = DatabaseMetadataConfig(
                database_name=request.database,
                database_type=metadata.dialect,
                description=description,
            )

        # 保存配置
        store = get_config_store()
        store.save_config(config, changed_by="ai_generator", change_reason="AI 自动生成")

        return {
            "success": True,
            "message": "配置生成成功" if config.auto_generated else "配置生成部分成功，请手动补充",
            "config": {
                "time_fields_count": len(config.time_fields),
                "business_terms_count": len(config.business_terms),
                "enum_fields_count": len(config.enum_fields),
                "entity_types_count": len(config.entity_types),
                "ambiguity_rules_count": len(config.ambiguity_rules),
            }
        }

    except Exception as e:
        import traceback
        print(f"Config generation route error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"配置生成失败: {str(e)}")


# ============================================
# 分步生成 API（新版，更稳定）
# ============================================

@router.post("/generate/start")
async def start_generate(request: StartGenerateRequest):
    """
    启动分步配置生成任务

    返回任务 ID，前端可以通过轮询或 WebSocket 获取进度
    """
    try:
        # 创建任务
        task = create_task(request.database)

        # 获取数据库连接
        db_manager = get_db_manager()
        connector = db_manager.get(request.database)

        # 启动异步生成（不等待完成）
        import asyncio

        async def run_generate():
            try:
                generator = StepByStepGenerator()
                await generator.generate(task, connector)
            except Exception as e:
                import traceback
                print(f"Background generate error: {e}")
                print(traceback.format_exc())
                task.status = "failed"
                task.error = str(e)
                update_task(task)

        # 在后台运行
        asyncio.create_task(run_generate())

        return {
            "success": True,
            "task_id": task.task_id,
            "message": "生成任务已启动",
        }

    except Exception as e:
        import traceback
        print(f"Start generate error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"启动生成任务失败: {str(e)}")


@router.get("/generate/status/{task_id}")
async def get_generate_status(task_id: str):
    """
    获取生成任务状态

    前端可以轮询此接口获取进度
    """
    task = get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return {
        "task_id": task.task_id,
        "database": task.database,
        "status": task.status,
        "current_step": task.current_step,
        "progress": task.progress,
        "steps": task.steps,
        "error": task.error,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


@router.post("/generate/sync")
async def generate_sync(request: StartGenerateRequest):
    """
    同步分步配置生成（等待完成）

    适用于前端需要等待结果的场景
    超时时间较长，适合复杂数据库
    """
    try:
        # 创建任务
        task = create_task(request.database)

        # 获取数据库连接
        db_manager = get_db_manager()
        connector = db_manager.get(request.database)

        # 同步执行生成
        generator = StepByStepGenerator()
        config = await generator.generate(task, connector)

        return {
            "success": True,
            "task_id": task.task_id,
            "message": "配置生成成功",
            "progress": task.progress,
            "steps": task.steps,
            "config": {
                "time_fields_count": len(config.time_fields),
                "business_terms_count": len(config.business_terms),
                "enum_fields_count": len(config.enum_fields),
                "entity_types_count": len(config.entity_types),
                "ambiguity_rules_count": len(config.ambiguity_rules),
            }
        }

    except Exception as e:
        import traceback
        print(f"Sync generate error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"配置生成失败: {str(e)}")


class SaveGeneratedConfigRequest(BaseModel):
    """保存生成的配置请求"""
    database: str
    time_fields: list[dict] = []
    enum_fields: list[dict] = []
    business_terms: list[dict] = []
    entity_types: list[dict] = []
    ambiguity_rules: list[dict] = []


@router.post("/generate/save")
async def save_generated_config(request: SaveGeneratedConfigRequest):
    """
    保存用户修改后的生成配置

    用户在预览页面可以删除不需要的项，然后保存
    """
    try:
        from datetime import datetime

        # 构建配置对象
        time_fields = []
        for tf in request.time_fields:
            time_fields.append(TimeFieldConfig(
                table_name=tf.get("table_name", ""),
                column_name=tf.get("column_name", ""),
                field_type=tf.get("field_type", "datetime"),
                description=tf.get("description", ""),
                format_hint=tf.get("format_hint", ""),
                is_primary_time=tf.get("is_primary_time", False),
                auto_detected=tf.get("auto_detected", True),
                user_confirmed=True,
            ))

        enum_fields = []
        for ef in request.enum_fields:
            enum_fields.append(EnumFieldConfig(
                table_name=ef.get("table_name", ""),
                column_name=ef.get("column_name", ""),
                values=ef.get("values", []),
                value_descriptions=ef.get("value_descriptions", {}),
                display_names=ef.get("display_names", {}),
                auto_detected=ef.get("auto_detected", True),
                user_confirmed=True,
            ))

        business_terms = []
        for bt in request.business_terms:
            business_terms.append(BusinessTermConfig(
                term=bt.get("term", ""),
                synonyms=bt.get("synonyms", []),
                mapped_table=bt.get("mapped_table", ""),
                mapped_column=bt.get("mapped_column", ""),
                aggregation=bt.get("aggregation", ""),
                description=bt.get("description", ""),
                auto_detected=bt.get("auto_detected", True),
                user_confirmed=True,
            ))

        entity_types = []
        for et in request.entity_types:
            entity_types.append(EntityTypeConfig(
                entity_type=et.get("entity_type", ""),
                display_name=et.get("display_name", ""),
                primary_table=et.get("primary_table", ""),
                name_column=et.get("name_column", ""),
                id_column=et.get("id_column", "id"),
                search_columns=et.get("search_columns", []),
                description=et.get("description", ""),
                auto_detected=et.get("auto_detected", True),
                user_confirmed=True,
            ))

        ambiguity_rules = []
        for ar in request.ambiguity_rules:
            ambiguity_rules.append(AmbiguityRuleConfig(
                rule_id=ar.get("rule_id", ""),
                rule_type=ar.get("rule_type", ""),
                trigger_keywords=ar.get("trigger_keywords", []),
                question_template=ar.get("question_template", ""),
                options_source=ar.get("options_source", "static"),
                static_options=ar.get("static_options", []),
                dynamic_query=ar.get("dynamic_query", ""),
                enabled=ar.get("enabled", True),
                auto_detected=ar.get("auto_detected", True),
                user_confirmed=True,
            ))

        # 获取现有配置的数据库类型
        store = get_config_store()
        existing_config = store.load_config(request.database)
        db_type = existing_config.database_type if existing_config else "unknown"

        config = DatabaseMetadataConfig(
            database_name=request.database,
            database_type=db_type,
            description=f"AI 生成配置（用户已确认）- {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            time_fields=time_fields,
            business_terms=business_terms,
            enum_fields=enum_fields,
            entity_types=entity_types,
            ambiguity_rules=ambiguity_rules,
            auto_generated=True,
        )

        # 保存配置
        store.save_config(config, changed_by="user", change_reason="用户确认 AI 生成的配置")

        return {
            "success": True,
            "message": "配置保存成功",
            "config": {
                "time_fields_count": len(time_fields),
                "business_terms_count": len(business_terms),
                "enum_fields_count": len(enum_fields),
                "entity_types_count": len(entity_types),
                "ambiguity_rules_count": len(ambiguity_rules),
            }
        }

    except Exception as e:
        import traceback
        print(f"Save generated config error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


@router.get("/{database}/time-fields")
async def get_time_fields(database: str):
    """获取时间字段配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        return {"database": database, "time_fields": []}

    return {
        "database": database,
        "time_fields": [vars(tf) for tf in config.time_fields],
    }


@router.post("/{database}/time-fields")
async def update_time_field(database: str, request: UpdateTimeFieldRequest):
    """添加或更新时间字段配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        # 创建新配置
        config = DatabaseMetadataConfig(
            database_name=database,
            database_type="unknown",
        )

    # 查找是否已存在
    existing_idx = None
    for i, tf in enumerate(config.time_fields):
        if tf.table_name == request.table_name and tf.column_name == request.column_name:
            existing_idx = i
            break

    new_field = TimeFieldConfig(
        table_name=request.table_name,
        column_name=request.column_name,
        field_type=request.field_type,
        description=request.description,
        format_hint=request.format_hint,
        is_primary_time=request.is_primary_time,
        auto_detected=False,
        user_confirmed=True,
    )

    if existing_idx is not None:
        config.time_fields[existing_idx] = new_field
    else:
        config.time_fields.append(new_field)

    store.save_config(config, changed_by="user", change_reason="更新时间字段配置")

    return {"success": True, "message": "时间字段配置已更新"}


@router.delete("/{database}/time-fields/{table_name}/{column_name}")
async def delete_time_field(database: str, table_name: str, column_name: str):
    """删除时间字段配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        raise HTTPException(status_code=404, detail="配置不存在")

    config.time_fields = [
        tf for tf in config.time_fields
        if not (tf.table_name == table_name and tf.column_name == column_name)
    ]

    store.save_config(config, changed_by="user", change_reason="删除时间字段配置")

    return {"success": True, "message": "时间字段配置已删除"}


@router.get("/{database}/business-terms")
async def get_business_terms(database: str):
    """获取业务词汇配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        return {"database": database, "business_terms": []}

    return {
        "database": database,
        "business_terms": [vars(bt) for bt in config.business_terms],
    }


@router.post("/{database}/business-terms")
async def update_business_term(database: str, request: UpdateBusinessTermRequest):
    """添加或更新业务词汇配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        config = DatabaseMetadataConfig(
            database_name=database,
            database_type="unknown",
        )

    # 查找是否已存在
    existing_idx = None
    for i, bt in enumerate(config.business_terms):
        if bt.term == request.term:
            existing_idx = i
            break

    new_term = BusinessTermConfig(
        term=request.term,
        synonyms=request.synonyms,
        mapped_table=request.mapped_table,
        mapped_column=request.mapped_column,
        aggregation=request.aggregation,
        description=request.description,
        auto_detected=False,
        user_confirmed=True,
    )

    if existing_idx is not None:
        config.business_terms[existing_idx] = new_term
    else:
        config.business_terms.append(new_term)

    store.save_config(config, changed_by="user", change_reason="更新业务词汇配置")

    return {"success": True, "message": "业务词汇配置已更新"}


@router.get("/{database}/enum-fields")
async def get_enum_fields(database: str):
    """获取枚举字段配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        return {"database": database, "enum_fields": []}

    return {
        "database": database,
        "enum_fields": [vars(ef) for ef in config.enum_fields],
    }


@router.post("/{database}/enum-fields")
async def update_enum_field(database: str, request: UpdateEnumFieldRequest):
    """添加或更新枚举字段配置"""
    store = get_config_store()
    config = store.load_config(database)

    if not config:
        config = DatabaseMetadataConfig(
            database_name=database,
            database_type="unknown",
        )

    # 查找是否已存在
    existing_idx = None
    for i, ef in enumerate(config.enum_fields):
        if ef.table_name == request.table_name and ef.column_name == request.column_name:
            existing_idx = i
            break

    new_field = EnumFieldConfig(
        table_name=request.table_name,
        column_name=request.column_name,
        values=request.values,
        value_descriptions=request.value_descriptions,
        display_names=request.display_names,
        auto_detected=False,
        user_confirmed=True,
    )

    if existing_idx is not None:
        config.enum_fields[existing_idx] = new_field
    else:
        config.enum_fields.append(new_field)

    store.save_config(config, changed_by="user", change_reason="更新枚举字段配置")

    return {"success": True, "message": "枚举字段配置已更新"}


@router.get("/{database}/history")
async def get_config_history(database: str, limit: int = 10):
    """获取配置修改历史"""
    store = get_config_store()
    history = store.get_history(database, limit)

    return {
        "database": database,
        "history": history,
    }


@router.get("/{database}/export")
async def export_config(database: str):
    """导出配置为 JSON"""
    store = get_config_store()
    config_json = store.export_config(database)

    if not config_json:
        raise HTTPException(status_code=404, detail="配置不存在")

    return {
        "database": database,
        "config_json": config_json,
    }


@router.post("/{database}/import")
async def import_config(database: str, request: ImportConfigRequest):
    """从 JSON 导入配置"""
    try:
        store = get_config_store()
        store.import_config(request.config_json, changed_by="user_import")

        return {"success": True, "message": "配置导入成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"配置导入失败: {str(e)}")


__all__ = ["router"]
