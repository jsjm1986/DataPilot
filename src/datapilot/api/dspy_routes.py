# -*- coding: utf-8 -*-
"""
DSPy 训练 API 路由

提供 DSPy 训练样本管理、训练任务、模块版本管理功能
"""

import asyncio
import json
from typing import Optional, Any
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..observability.audit import get_audit_store
from ..llm.dspy_modules.self_evolution import (
    get_evolution_engine,
    TrainingSample,
    EvolutionStatus,
)
from ..db.connector import get_db_manager

router = APIRouter(prefix="/api/v1/dspy", tags=["DSPy Training"])


# ============================================
# 请求/响应模型
# ============================================

class SampleFromExecutionRequest(BaseModel):
    """从执行记录创建样本的请求"""
    trace_ids: list[str] = Field(..., description="执行记录的 trace_id 列表")


class TrainRequest(BaseModel):
    """训练请求"""
    module_type: str = Field(default="text2sql", description="模块类型: text2sql, decompose, refine")
    optimizer_type: str = Field(default="bootstrap", description="优化器类型: bootstrap, random_search")
    max_bootstrapped_demos: int = Field(default=4, description="最大 bootstrap 示例数")
    max_labeled_demos: int = Field(default=16, description="最大标注示例数")
    validation_split: float = Field(default=0.2, description="验证集比例")


class ActivateVersionRequest(BaseModel):
    """激活版本请求"""
    version: int = Field(..., description="要激活的版本号")


class VerifySampleRequest(BaseModel):
    """验证样本请求"""
    verified: bool = Field(default=True, description="是否验证")


class ApiResponse(BaseModel):
    """通用 API 响应"""
    success: bool = True
    message: str = ""
    data: Any = None


# ============================================
# 全局训练状态
# ============================================

_training_status = {
    "is_training": False,
    "progress": 0,
    "stage": "",
    "logs": [],
    "result": None,
    "error": None,
    "started_at": None,
}


# ============================================
# 样本采集 API
# ============================================

@router.get("/candidates")
async def get_candidates(
    status: str = Query(default="success", description="状态过滤"),
    database: Optional[str] = Query(default=None, description="数据库过滤"),
    start_time: Optional[str] = Query(default=None, description="开始时间"),
    end_time: Optional[str] = Query(default=None, description="结束时间"),
    search: Optional[str] = Query(default=None, description="搜索关键词"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
):
    """
    获取可用于训练的执行历史记录

    只返回有 SQL 的成功执行记录
    """
    audit_store = get_audit_store()

    # 计算偏移量
    offset = (page - 1) * page_size

    # 查询审计日志
    records = await audit_store.query(
        status=status,
        database=database,
        start_time=start_time,
        end_time=end_time,
        search_query=search,
        limit=page_size,
        offset=offset,
        order_by="timestamp",
        order_dir="DESC",
    )

    # 过滤掉没有 SQL 的记录
    records = [r for r in records if r.get("sql")]

    # 获取总数
    total = await audit_store.count(
        status=status,
        start_time=start_time,
        end_time=end_time,
    )

    return {
        "success": True,
        "data": records,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": offset + len(records) < total,
    }


@router.get("/candidates/{trace_id}")
async def get_candidate_detail(trace_id: str):
    """
    获取单条执行记录的完整信息，包含数据库 Schema
    """
    audit_store = get_audit_store()

    # 获取执行记录
    record = await audit_store.get_by_trace_id(trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="执行记录不存在")

    # 获取数据库 Schema
    schema = ""
    try:
        db_manager = get_db_manager()
        database = record.get("database", "")
        if database:
            # 尝试获取 Schema
            schema_info = await db_manager.get_schema(database)
            if schema_info:
                schema = schema_info
    except Exception as e:
        # Schema 获取失败不影响返回
        schema = f"-- Schema 获取失败: {str(e)}"

    return {
        "success": True,
        "data": {
            **record,
            "schema": schema,
        },
    }


@router.post("/samples/from-execution")
async def add_samples_from_execution(request: SampleFromExecutionRequest):
    """
    将执行记录添加为训练样本
    """
    audit_store = get_audit_store()
    engine = get_evolution_engine()

    added = []
    errors = []

    for trace_id in request.trace_ids:
        try:
            # 获取执行记录
            record = await audit_store.get_by_trace_id(trace_id)
            if not record:
                errors.append({"trace_id": trace_id, "error": "记录不存在"})
                continue

            if not record.get("sql"):
                errors.append({"trace_id": trace_id, "error": "记录没有 SQL"})
                continue

            # 获取数据库 Schema
            schema = ""
            try:
                db_manager = get_db_manager()
                database = record.get("database", "")
                if database:
                    schema_info = await db_manager.get_schema(database)
                    if schema_info:
                        schema = schema_info
            except Exception:
                schema = ""

            # 确定 dialect
            dialect = "sqlite"  # 默认
            database = record.get("database", "").lower()
            if "mysql" in database:
                dialect = "mysql"
            elif "postgres" in database or "pg" in database:
                dialect = "postgresql"

            # 添加训练样本
            sample = engine.collector.add_sample(
                question=record["query"],
                schema=schema,
                gold_sql=record["sql"],
                dialect=dialect,
                execution_result=None,
                source="execution",
                verified=True,  # 执行成功的自动验证
            )

            added.append({
                "trace_id": trace_id,
                "sample_id": sample.id,
                "question": record["query"][:50] + "..." if len(record["query"]) > 50 else record["query"],
            })

        except Exception as e:
            errors.append({"trace_id": trace_id, "error": str(e)})

    return {
        "success": True,
        "message": f"成功添加 {len(added)} 条样本",
        "data": {
            "added": added,
            "errors": errors,
        },
    }


# ============================================
# 训练样本管理 API
# ============================================

@router.get("/samples")
async def get_samples(
    verified: Optional[bool] = Query(default=None, description="验证状态过滤"),
    source: Optional[str] = Query(default=None, description="来源过滤"),
    search: Optional[str] = Query(default=None, description="搜索关键词"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
):
    """
    获取训练样本列表
    """
    engine = get_evolution_engine()

    # 获取所有样本
    all_samples = engine.collector.get_samples()

    # 过滤
    samples = all_samples
    if verified is not None:
        samples = [s for s in samples if s.verified == verified]
    if source:
        samples = [s for s in samples if s.source == source]
    if search:
        search_lower = search.lower()
        samples = [s for s in samples if search_lower in s.question.lower() or search_lower in s.gold_sql.lower()]

    # 分页
    total = len(samples)
    offset = (page - 1) * page_size
    samples = samples[offset:offset + page_size]

    # 转换为字典
    data = [
        {
            "id": s.id,
            "question": s.question,
            "schema": s.schema[:200] + "..." if len(s.schema) > 200 else s.schema,
            "dialect": s.dialect,
            "gold_sql": s.gold_sql,
            "source": s.source,
            "verified": s.verified,
            "created_at": s.created_at,
        }
        for s in samples
    ]

    return {
        "success": True,
        "data": data,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": offset + len(data) < total,
    }


@router.get("/samples/stats")
async def get_samples_stats():
    """
    获取训练样本统计
    """
    engine = get_evolution_engine()
    stats = engine.collector.get_stats()

    return {
        "success": True,
        "data": stats,
    }


@router.get("/samples/{sample_id}")
async def get_sample_detail(sample_id: str):
    """
    获取单条训练样本详情
    """
    engine = get_evolution_engine()
    samples = engine.collector.get_samples()

    for s in samples:
        if s.id == sample_id:
            return {
                "success": True,
                "data": {
                    "id": s.id,
                    "question": s.question,
                    "schema": s.schema,
                    "dialect": s.dialect,
                    "gold_sql": s.gold_sql,
                    "execution_result": s.execution_result,
                    "source": s.source,
                    "verified": s.verified,
                    "created_at": s.created_at,
                },
            }

    raise HTTPException(status_code=404, detail="样本不存在")


@router.delete("/samples/{sample_id}")
async def delete_sample(sample_id: str):
    """
    删除训练样本
    """
    engine = get_evolution_engine()
    engine.collector.delete_sample(sample_id)

    return {
        "success": True,
        "message": "样本已删除",
    }


@router.put("/samples/{sample_id}/verify")
async def verify_sample(sample_id: str, request: VerifySampleRequest):
    """
    验证/取消验证训练样本
    """
    engine = get_evolution_engine()
    engine.collector.verify_sample(sample_id, request.verified)

    return {
        "success": True,
        "message": "验证状态已更新",
    }


@router.post("/samples/export")
async def export_samples(
    verified_only: bool = Query(default=True, description="只导出已验证样本"),
):
    """
    导出训练样本为 JSON
    """
    engine = get_evolution_engine()
    samples = engine.collector.get_samples(verified_only=verified_only)

    data = [
        {
            "id": s.id,
            "question": s.question,
            "schema": s.schema,
            "dialect": s.dialect,
            "sql": s.gold_sql,
            "source": s.source,
            "verified": s.verified,
            "created_at": s.created_at,
        }
        for s in samples
    ]

    return {
        "success": True,
        "data": data,
        "count": len(data),
    }


# ============================================
# 训练任务 API
# ============================================

@router.post("/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    启动 DSPy 训练任务
    """
    global _training_status

    if _training_status["is_training"]:
        raise HTTPException(status_code=400, detail="训练任务正在进行中")

    # 检查样本数量
    engine = get_evolution_engine()
    samples = engine.collector.get_samples(verified_only=True)

    if len(samples) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"训练样本不足，当前只有 {len(samples)} 条已验证样本，至少需要 5 条"
        )

    # 重置状态
    _training_status = {
        "is_training": True,
        "progress": 0,
        "stage": "初始化",
        "logs": [f"[{datetime.now().strftime('%H:%M:%S')}] 开始训练任务..."],
        "result": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
    }

    # 在后台执行训练
    background_tasks.add_task(
        _run_training,
        request.module_type,
        request.optimizer_type,
        request.max_bootstrapped_demos,
        request.max_labeled_demos,
        request.validation_split,
    )

    return {
        "success": True,
        "message": "训练任务已启动",
    }


async def _run_training(
    module_type: str,
    optimizer_type: str,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
    validation_split: float,
):
    """后台训练任务"""
    global _training_status

    try:
        engine = get_evolution_engine()

        # 更新配置
        engine.config.max_bootstrapped_demos = max_bootstrapped_demos
        engine.config.max_labeled_demos = max_labeled_demos
        engine.config.train_split = 1 - validation_split

        # 更新状态
        _training_status["stage"] = "加载数据"
        _training_status["progress"] = 10
        _training_status["logs"].append(
            f"[{datetime.now().strftime('%H:%M:%S')}] 加载训练数据..."
        )

        await asyncio.sleep(0.5)  # 给前端时间获取状态

        # 更新状态
        _training_status["stage"] = "优化中"
        _training_status["progress"] = 30
        _training_status["logs"].append(
            f"[{datetime.now().strftime('%H:%M:%S')}] 开始 DSPy 优化..."
        )

        # 执行进化
        result = await engine.evolve(module_type=module_type, force=True)

        # 更新状态
        _training_status["progress"] = 100
        _training_status["stage"] = "完成"
        _training_status["result"] = {
            "success": result.success,
            "module_type": result.module_type,
            "old_version": result.old_version,
            "new_version": result.new_version,
            "old_accuracy": result.old_accuracy,
            "new_accuracy": result.new_accuracy,
            "train_samples": result.train_samples,
            "eval_samples": result.eval_samples,
            "duration_seconds": result.duration_seconds,
            "deployed": result.deployed,
            "error": result.error,
        }
        _training_status["logs"].append(
            f"[{datetime.now().strftime('%H:%M:%S')}] 训练完成! 准确率: {result.new_accuracy:.1%}"
        )

    except Exception as e:
        _training_status["error"] = str(e)
        _training_status["stage"] = "失败"
        _training_status["logs"].append(
            f"[{datetime.now().strftime('%H:%M:%S')}] 训练失败: {str(e)}"
        )

    finally:
        _training_status["is_training"] = False


@router.get("/train/status")
async def get_training_status():
    """
    获取训练状态
    """
    return {
        "success": True,
        "data": _training_status,
    }


@router.post("/train/cancel")
async def cancel_training():
    """
    取消训练任务
    """
    global _training_status

    if not _training_status["is_training"]:
        raise HTTPException(status_code=400, detail="没有正在进行的训练任务")

    # 标记取消（实际取消需要在训练循环中检查）
    _training_status["is_training"] = False
    _training_status["stage"] = "已取消"
    _training_status["logs"].append(
        f"[{datetime.now().strftime('%H:%M:%S')}] 训练已取消"
    )

    return {
        "success": True,
        "message": "训练任务已取消",
    }


@router.get("/train/stream")
async def train_status_stream():
    """
    训练状态 SSE 流
    """
    async def event_generator():
        last_log_count = 0
        while True:
            status = _training_status.copy()

            # 只发送新日志
            new_logs = status["logs"][last_log_count:]
            last_log_count = len(status["logs"])

            yield f"data: {json.dumps({'status': status, 'new_logs': new_logs}, ensure_ascii=False)}\n\n"

            if not status["is_training"]:
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================
# 模块管理 API
# ============================================

@router.get("/modules")
async def get_modules():
    """
    获取所有模块类型及其版本信息
    """
    engine = get_evolution_engine()

    module_types = ["text2sql", "decompose", "refine"]
    modules = {}

    for mt in module_types:
        versions = engine.version_manager.get_all_versions(mt)
        active = engine.version_manager.get_active_version(mt)

        modules[mt] = {
            "versions": [
                {
                    "version": v.version,
                    "accuracy": v.accuracy,
                    "created_at": v.created_at,
                    "optimizer": v.optimizer,
                    "train_samples": v.train_samples,
                    "is_active": v.is_active,
                }
                for v in versions
            ],
            "active_version": active.version if active else None,
            "total_versions": len(versions),
        }

    return {
        "success": True,
        "data": modules,
    }


@router.get("/modules/{module_type}/versions")
async def get_module_versions(module_type: str):
    """
    获取指定模块的所有版本
    """
    engine = get_evolution_engine()
    versions = engine.version_manager.get_all_versions(module_type)

    return {
        "success": True,
        "data": [
            {
                "version": v.version,
                "accuracy": v.accuracy,
                "created_at": v.created_at,
                "optimizer": v.optimizer,
                "train_samples": v.train_samples,
                "is_active": v.is_active,
                "path": v.path,
                "metadata": v.metadata,
            }
            for v in sorted(versions, key=lambda x: x.version, reverse=True)
        ],
    }


@router.post("/modules/{module_type}/activate")
async def activate_module_version(module_type: str, request: ActivateVersionRequest):
    """
    激活指定版本
    """
    engine = get_evolution_engine()
    success = engine.version_manager.activate_version(module_type, request.version)

    if not success:
        raise HTTPException(status_code=404, detail="版本不存在")

    return {
        "success": True,
        "message": f"已激活 {module_type} v{request.version}",
    }


@router.post("/modules/{module_type}/rollback")
async def rollback_module(module_type: str):
    """
    回滚到上一个版本
    """
    engine = get_evolution_engine()
    prev_version = engine.version_manager.rollback(module_type)

    if not prev_version:
        raise HTTPException(status_code=400, detail="没有可回滚的版本")

    return {
        "success": True,
        "message": f"已回滚到 {module_type} v{prev_version.version}",
        "data": {
            "version": prev_version.version,
            "accuracy": prev_version.accuracy,
        },
    }


@router.delete("/modules/{module_type}/{version}")
async def delete_module_version(module_type: str, version: int):
    """
    删除指定版本（不能删除激活的版本）
    """
    engine = get_evolution_engine()

    # 检查是否是激活版本
    active = engine.version_manager.get_active_version(module_type)
    if active and active.version == version:
        raise HTTPException(status_code=400, detail="不能删除激活的版本")

    # 获取版本信息
    versions = engine.version_manager.get_all_versions(module_type)
    target = None
    for v in versions:
        if v.version == version:
            target = v
            break

    if not target:
        raise HTTPException(status_code=404, detail="版本不存在")

    # 删除文件
    try:
        path = Path(target.path)
        if path.exists():
            path.unlink()
    except Exception:
        pass

    # 从版本列表中移除
    engine.version_manager._versions[module_type] = [
        v for v in versions if v.version != version
    ]
    engine.version_manager._save_versions()

    return {
        "success": True,
        "message": f"已删除 {module_type} v{version}",
    }


# ============================================
# 状态 API
# ============================================

@router.get("/status")
async def get_dspy_status():
    """
    获取 DSPy 系统整体状态
    """
    engine = get_evolution_engine()

    return {
        "success": True,
        "data": {
            "evolution_status": engine.status.value,
            "training_status": _training_status,
            "collector_stats": engine.collector.get_stats(),
            "config": {
                "min_samples_for_evolution": engine.config.min_samples_for_evolution,
                "auto_evolution_enabled": engine.config.auto_evolution_enabled,
                "accuracy_threshold": engine.config.accuracy_threshold,
            },
        },
    }
