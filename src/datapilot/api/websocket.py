# -*- coding: utf-8 -*-
"""
WebSocket Handler for Real-time Query Processing

使用 LangGraph 1.x 新特性:
1. astream_events() - 实时事件流
2. interrupt() - Human-in-the-loop 原生支持
3. Command(resume=...) - 恢复中断的工作流
"""

import uuid
from typing import Optional, Any
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from ..core.graph import (
    get_compiled_graph,
    run_with_streaming,
    resume_after_interrupt,
)
from ..core.state import create_initial_state
from ..observability.audit import log_audit, AuditTimer


class ConnectionManager:
    """WebSocket Connection Manager"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        # 存储每个客户端的 thread_id，用于 interrupt/resume
        self.client_threads: dict[str, str] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        # 为每个客户端创建唯一的 thread_id
        self.client_threads[client_id] = f"ws-{client_id}-{uuid.uuid4().hex[:8]}"

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_threads:
            del self.client_threads[client_id]

    def get_thread_id(self, client_id: str) -> str:
        """获取客户端的 thread_id"""
        return self.client_threads.get(client_id, f"ws-{client_id}")

    def set_thread_id(self, client_id: str, thread_id: str):
        """设置客户端的 thread_id"""
        self.client_threads[client_id] = thread_id

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                # 连接已关闭，静默处理
                print(f"[WebSocket] Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()

# 存储待处理的 interrupt 状态
_pending_interrupts: dict[str, dict] = {}


async def process_query_with_langgraph(
    client_id: str,
    query: str,
    database: str = "default",
    user_id: str = "anonymous",
    tenant_id: str = "default",
):
    """
    使用 LangGraph astream_events() 处理查询

    利用 LangGraph 1.x 的原生流式事件 API，
    自动处理 interrupt() 暂停和恢复。
    """
    thread_id = manager.get_thread_id(client_id)
    timer = AuditTimer()
    timer.__enter__()

    final_sql = None
    final_data = None
    final_row_count = 0

    try:
        # 使用 run_with_streaming 获取实时事件流
        async for event in run_with_streaming(
            query=query,
            database=database,
            user_id=user_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
        ):
            event_type = event.get("type")

            if event_type == "agent_start":
                # Agent 开始执行
                agent_name = event.get("agent", "unknown")
                await manager.send_message(client_id, {
                    "type": "progress",
                    "stage": agent_name,
                    "message": f"Starting {agent_name}...",
                    "timestamp": event.get("timestamp"),
                })

            elif event_type == "agent_end":
                # Agent 执行完成
                agent_name = event.get("agent", "unknown")
                output = event.get("output", {})

                # 检查是否需要澄清 (通过 clarify_needed 标志或 clarify_options)
                clarify_options = output.get("clarify_options") if isinstance(output, dict) else None
                if isinstance(output, dict) and (output.get("clarify_needed") or clarify_options):
                    # 保存 interrupt 状态
                    _pending_interrupts[client_id] = {
                        "thread_id": thread_id,
                        "query": query,
                        "database": database,
                        "user_id": user_id,
                        "tenant_id": tenant_id,
                    }
                    # 从 clarify_options 或直接从 output 获取问题和选项
                    question = clarify_options.get("question") if clarify_options else output.get("question")
                    options = clarify_options.get("options") if clarify_options else output.get("options")
                    await manager.send_message(client_id, {
                        "type": "clarify_needed",
                        "question": question,
                        "options": options,
                        "hint": output.get("hint", "请选择一个选项以继续查询"),
                        "thread_id": thread_id,
                    })
                    timer.__exit__(None, None, None)
                    log_audit(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        trace_id=thread_id,
                        session_id=client_id,
                        database=database,
                        query=query,
                        sql=None,
                        status="clarify_needed",
                        duration_ms=getattr(timer, "elapsed_ms", None),
                    )
                    return

                # 检查是否需要人工介入 (human_handoff)
                if isinstance(output, dict) and output.get("type") == "human_handoff":
                    _pending_interrupts[client_id] = {
                        "thread_id": thread_id,
                        "query": query,
                        "database": database,
                        "user_id": user_id,
                        "tenant_id": tenant_id,
                        "is_handoff": True,
                    }
                    await manager.send_message(client_id, {
                        "type": "human_handoff",
                        "reason": output.get("reason"),
                        "last_error": output.get("last_error"),
                        "sql_attempts": output.get("sql_attempts"),
                        "thread_id": thread_id,
                    })
                    timer.__exit__(None, None, None)
                    log_audit(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        trace_id=thread_id,
                        session_id=client_id,
                        database=database,
                        query=query,
                        sql=None,
                        status="human_handoff",
                        duration_ms=getattr(timer, "elapsed_ms", None),
                    )
                    return

                # 提取关键数据
                if isinstance(output, dict):
                    if "winner_sql" in output:
                        final_sql = output["winner_sql"]
                    if "execution_result" in output:
                        exec_result = output["execution_result"]
                        if exec_result and exec_result.get("success"):
                            final_data = exec_result.get("data", [])
                            final_row_count = exec_result.get("row_count", 0)

                await manager.send_message(client_id, {
                    "type": "progress",
                    "stage": agent_name,
                    "message": f"Completed {agent_name}",
                    "data": _safe_output(output),
                    "timestamp": event.get("timestamp"),
                })

            elif event_type == "stream":
                # 流式数据
                await manager.send_message(client_id, {
                    "type": "stream",
                    "data": event.get("data"),
                    "timestamp": event.get("timestamp"),
                })

        # 工作流完成，获取最终状态
        graph = get_compiled_graph()
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await graph.aget_state(config)

        chart_config = {}
        if final_state and final_state.values:
            state_values = final_state.values
            final_sql = state_values.get("winner_sql", final_sql)
            exec_result = state_values.get("execution_result") or {}
            if exec_result.get("success"):
                final_data = exec_result.get("data", final_data)
                final_row_count = exec_result.get("row_count", final_row_count)
            chart_config = state_values.get("chart_config", {})

        # 无论如何都发送最终结果
        await manager.send_message(client_id, {
            "type": "result",
            "thread_id": thread_id,
            "sql": final_sql,
            "data": final_data,
            "row_count": final_row_count,
            "chart_config": chart_config,
            "status": "success",
        })

        timer.__exit__(None, None, None)
        log_audit(
            user_id=user_id,
            tenant_id=tenant_id,
            trace_id=thread_id,
            session_id=client_id,
            database=database,
            query=query,
            sql=final_sql,
            status="success",
            row_count=final_row_count,
            duration_ms=getattr(timer, "elapsed_ms", None),
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[WebSocket Error] {e}")
        print(f"[WebSocket Traceback]\n{error_traceback}")
        timer.__exit__(None, None, None)
        log_audit(
            user_id=user_id,
            tenant_id=tenant_id,
            trace_id=thread_id,
            session_id=client_id,
            database=database,
            query=query,
            sql=final_sql,
            status="error",
            error=str(e),
            duration_ms=getattr(timer, "elapsed_ms", None),
        )
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e),
            "thread_id": thread_id,
        })


async def handle_clarify_response(
    client_id: str,
    selected_option: Any,
):
    """
    处理用户的澄清选择，重新执行查询流程（跳过歧义检测）
    """
    pending = _pending_interrupts.pop(client_id, None)
    if not pending:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "No pending clarification",
        })
        return

    thread_id = pending["thread_id"]
    original_query = pending.get("query", "")
    database = pending.get("database", "default")
    user_id = pending.get("user_id", "anonymous")
    tenant_id = pending.get("tenant_id", "default")

    # 将用户选择融入查询
    resolved_query = f"{original_query} ({selected_option})"
    print(f"[Clarify] Resolved query: {resolved_query}")

    timer = AuditTimer()
    timer.__enter__()

    final_sql = None
    final_data = None
    final_row_count = 0

    try:
        # 重新执行查询流程，使用解析后的查询
        # 创建新的 thread_id 用于新的执行
        new_thread_id = f"{thread_id}-clarified"

        # 跳过歧义检测，因为用户已经选择了澄清选项
        async for event in run_with_streaming(
            query=resolved_query,
            database=database,
            user_id=user_id,
            tenant_id=tenant_id,
            thread_id=new_thread_id,
            skip_ambi_resolver=True,
        ):
            event_type = event.get("type")

            if event_type == "agent_start":
                agent_name = event.get("agent", "unknown")
                await manager.send_message(client_id, {
                    "type": "progress",
                    "stage": agent_name,
                    "message": f"Starting {agent_name}...",
                    "timestamp": event.get("timestamp"),
                })

            elif event_type == "agent_end":
                agent_name = event.get("agent", "unknown")
                output = event.get("output", {})

                if isinstance(output, dict):
                    if "winner_sql" in output:
                        final_sql = output["winner_sql"]
                    if "execution_result" in output:
                        exec_result = output["execution_result"]
                        if exec_result and exec_result.get("success"):
                            final_data = exec_result.get("data", [])
                            final_row_count = exec_result.get("row_count", 0)

                await manager.send_message(client_id, {
                    "type": "progress",
                    "stage": agent_name,
                    "message": f"Completed {agent_name}",
                    "timestamp": event.get("timestamp"),
                })

        # 获取最终状态 - 使用相同的图配置
        graph = get_compiled_graph(include_ambi_resolver=False)
        config = {"configurable": {"thread_id": new_thread_id}}
        final_state = await graph.aget_state(config)

        chart_config = {}
        if final_state and final_state.values:
            state_values = final_state.values
            final_sql = state_values.get("winner_sql", final_sql)
            exec_result = state_values.get("execution_result") or {}
            if exec_result.get("success"):
                final_data = exec_result.get("data", final_data)
                final_row_count = exec_result.get("row_count", final_row_count)
            chart_config = state_values.get("chart_config", {})

        # 无论如何都发送结果
        await manager.send_message(client_id, {
            "type": "result",
            "thread_id": new_thread_id,
            "sql": final_sql,
            "data": final_data,
            "row_count": final_row_count,
            "chart_config": chart_config,
            "status": "success",
        })

        timer.__exit__(None, None, None)
        log_audit(
            user_id=user_id,
            tenant_id=tenant_id,
            trace_id=new_thread_id,
            session_id=client_id,
            database=database,
            query=resolved_query,
            sql=final_sql,
            status="success",
            row_count=final_row_count,
            duration_ms=getattr(timer, "elapsed_ms", None),
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[Clarify Error] {e}")
        print(f"[Clarify Traceback]\n{error_traceback}")
        timer.__exit__(None, None, None)
        await manager.send_message(client_id, {
            "type": "error",
            "message": str(e),
            "thread_id": thread_id,
        })


async def handle_human_handoff_response(
    client_id: str,
    action: str,
    hint: Optional[str] = None,
):
    """
    处理人工介入响应

    Args:
        action: "retry" 重试, "abort" 放弃
        hint: 人工提供的提示信息
    """
    pending = _pending_interrupts.pop(client_id, None)
    if not pending or not pending.get("is_handoff"):
        await manager.send_message(client_id, {
            "type": "error",
            "message": "No pending human handoff",
        })
        return

    thread_id = pending["thread_id"]

    if action == "abort":
        await manager.send_message(client_id, {
            "type": "result",
            "thread_id": thread_id,
            "status": "aborted",
            "message": "Query aborted by user",
        })
        return

    if action == "retry":
        try:
            # 使用 Command 恢复，传递人工提示
            final_state = await resume_after_interrupt(
                thread_id=thread_id,
                user_input={"action": "retry", "hint": hint},
            )

            if final_state:
                final_sql = final_state.get("winner_sql")
                exec_result = final_state.get("execution_result") or {}
                final_data = exec_result.get("data", []) if exec_result.get("success") else []
                final_row_count = exec_result.get("row_count", 0)
                chart_config = final_state.get("chart_config", {})

                await manager.send_message(client_id, {
                    "type": "result",
                    "thread_id": thread_id,
                    "sql": final_sql,
                    "data": final_data,
                    "row_count": final_row_count,
                    "chart_config": chart_config,
                    "status": "success",
                })

        except Exception as e:
            await manager.send_message(client_id, {
                "type": "error",
                "message": str(e),
                "thread_id": thread_id,
            })


def _safe_output(output: Any) -> Any:
    """安全地序列化输出，移除不可序列化的内容"""
    if output is None:
        return None
    if isinstance(output, dict):
        result = {}
        for k, v in output.items():
            try:
                # 跳过大数据字段
                if k in ("data", "execution_result") and isinstance(v, (list, dict)):
                    if isinstance(v, list) and len(v) > 10:
                        result[k] = f"[{len(v)} items]"
                    elif isinstance(v, dict) and "data" in v:
                        result[k] = {**v, "data": f"[{len(v.get('data', []))} items]"}
                    else:
                        result[k] = v
                else:
                    result[k] = v
            except Exception:
                result[k] = str(v)
        return result
    return output


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint handler

    支持的消息类型:
    - query: 执行查询
    - clarify: 响应澄清请求
    - handoff: 响应人工介入请求
    - ping: 心跳检测
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "query":
                query = data.get("query", "")
                database = data.get("database", "default")
                user_id = data.get("user_id", "anonymous")
                tenant_id = data.get("tenant_id", "default")

                if query:
                    # 为新查询创建新的 thread_id
                    new_thread_id = f"ws-{client_id}-{uuid.uuid4().hex[:8]}"
                    manager.set_thread_id(client_id, new_thread_id)

                    await process_query_with_langgraph(
                        client_id=client_id,
                        query=query,
                        database=database,
                        user_id=user_id,
                        tenant_id=tenant_id,
                    )
                else:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Query is required",
                    })

            elif msg_type == "clarify":
                selected_option = data.get("selected_option")
                if selected_option:
                    await handle_clarify_response(client_id, selected_option)
                else:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "selected_option is required",
                    })

            elif msg_type == "handoff":
                action = data.get("action", "abort")
                hint = data.get("hint")
                await handle_human_handoff_response(client_id, action, hint)

            elif msg_type == "ping":
                await manager.send_message(client_id, {"type": "pong"})

            else:
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"WebSocket error: {str(e)}",
        })
        manager.disconnect(client_id)


# 保留旧的函数名以保持向后兼容
process_query_streaming = process_query_with_langgraph


__all__ = [
    "manager",
    "websocket_endpoint",
    "process_query_streaming",
    "process_query_with_langgraph",
    "handle_clarify_response",
    "handle_human_handoff_response",
]
