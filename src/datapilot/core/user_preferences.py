# -*- coding: utf-8 -*-
"""
用户偏好持久化模块

实现用户偏好的存储、读取和管理

功能:
1. 用户偏好存储 (文件/数据库)
2. 查询历史记录
3. 可视化偏好
4. 默认设置管理
5. 多租户支持
"""

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from enum import Enum
import hashlib

from prometheus_client import Counter, Gauge


# ============================================
# Prometheus 指标
# ============================================

PREFERENCE_OPERATIONS = Counter(
    "datapilot_preference_operations_total",
    "Total preference operations",
    labelnames=["operation", "status"],  # get, set, delete; success, failed
)

ACTIVE_USERS = Gauge(
    "datapilot_active_users",
    "Number of users with stored preferences",
)


# ============================================
# 数据结构
# ============================================

class ChartPreference(Enum):
    """图表偏好"""
    AUTO = "auto"  # 自动推荐
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    TABLE = "table"


class ThemePreference(Enum):
    """主题偏好"""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


@dataclass
class VisualizationPreferences:
    """可视化偏好"""
    default_chart_type: str = "auto"
    color_scheme: str = "default"
    show_data_labels: bool = True
    animation_enabled: bool = True
    max_rows_display: int = 100
    decimal_places: int = 2


@dataclass
class QueryPreferences:
    """查询偏好"""
    default_database: Optional[str] = None
    default_limit: int = 100
    auto_execute: bool = False
    show_sql: bool = True
    save_history: bool = True
    max_history_items: int = 100


@dataclass
class UIPreferences:
    """UI 偏好"""
    theme: str = "dark"
    language: str = "zh-CN"
    sidebar_collapsed: bool = False
    trace_panel_visible: bool = True
    editor_font_size: int = 14
    result_panel_height: int = 300


@dataclass
class QueryHistoryItem:
    """查询历史项"""
    id: str
    query: str
    sql: str
    database: str
    timestamp: str
    success: bool
    row_count: int = 0
    execution_time_ms: float = 0
    chart_type: Optional[str] = None
    is_favorite: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.query}:{self.timestamp}".encode()
            ).hexdigest()[:12]


@dataclass
class UserPreferences:
    """用户偏好"""
    user_id: str
    visualization: VisualizationPreferences = field(default_factory=VisualizationPreferences)
    query: QueryPreferences = field(default_factory=QueryPreferences)
    ui: UIPreferences = field(default_factory=UIPreferences)
    custom_settings: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# ============================================
# 存储后端接口
# ============================================

class PreferenceStorageBackend:
    """偏好存储后端接口"""

    async def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """获取用户偏好"""
        raise NotImplementedError

    async def save_preferences(self, preferences: UserPreferences) -> bool:
        """保存用户偏好"""
        raise NotImplementedError

    async def delete_preferences(self, user_id: str) -> bool:
        """删除用户偏好"""
        raise NotImplementedError

    async def get_query_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[QueryHistoryItem]:
        """获取查询历史"""
        raise NotImplementedError

    async def add_query_history(
        self,
        user_id: str,
        item: QueryHistoryItem,
    ) -> bool:
        """添加查询历史"""
        raise NotImplementedError

    async def delete_query_history(
        self,
        user_id: str,
        item_id: Optional[str] = None,
    ) -> bool:
        """删除查询历史"""
        raise NotImplementedError

    async def list_users(self) -> list[str]:
        """列出所有用户"""
        raise NotImplementedError


# ============================================
# 文件存储后端
# ============================================

class FileStorageBackend(PreferenceStorageBackend):
    """
    文件存储后端

    将用户偏好存储为 JSON 文件
    """

    def __init__(self, data_dir: str = "data/user_preferences"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_user_dir(self, user_id: str) -> Path:
        """获取用户目录"""
        # 使用 hash 避免特殊字符问题
        safe_id = hashlib.md5(user_id.encode()).hexdigest()[:16]
        user_dir = self.data_dir / safe_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_preferences_file(self, user_id: str) -> Path:
        """获取偏好文件路径"""
        return self._get_user_dir(user_id) / "preferences.json"

    def _get_history_file(self, user_id: str) -> Path:
        """获取历史文件路径"""
        return self._get_user_dir(user_id) / "history.json"

    async def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """获取用户偏好"""
        try:
            file_path = self._get_preferences_file(user_id)
            if not file_path.exists():
                return None

            with self._lock:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # 重建嵌套对象
            prefs = UserPreferences(
                user_id=data.get("user_id", user_id),
                visualization=VisualizationPreferences(**data.get("visualization", {})),
                query=QueryPreferences(**data.get("query", {})),
                ui=UIPreferences(**data.get("ui", {})),
                custom_settings=data.get("custom_settings", {}),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
            )

            PREFERENCE_OPERATIONS.labels(operation="get", status="success").inc()
            return prefs

        except Exception as e:
            PREFERENCE_OPERATIONS.labels(operation="get", status="failed").inc()
            print(f"Failed to get preferences for {user_id}: {e}")
            return None

    async def save_preferences(self, preferences: UserPreferences) -> bool:
        """保存用户偏好"""
        try:
            file_path = self._get_preferences_file(preferences.user_id)

            # 更新时间戳
            preferences.updated_at = datetime.utcnow().isoformat()

            # 转换为可序列化的字典
            data = {
                "user_id": preferences.user_id,
                "visualization": asdict(preferences.visualization),
                "query": asdict(preferences.query),
                "ui": asdict(preferences.ui),
                "custom_settings": preferences.custom_settings,
                "created_at": preferences.created_at,
                "updated_at": preferences.updated_at,
            }

            with self._lock:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            PREFERENCE_OPERATIONS.labels(operation="set", status="success").inc()
            self._update_user_count()
            return True

        except Exception as e:
            PREFERENCE_OPERATIONS.labels(operation="set", status="failed").inc()
            print(f"Failed to save preferences for {preferences.user_id}: {e}")
            return False

    async def delete_preferences(self, user_id: str) -> bool:
        """删除用户偏好"""
        try:
            user_dir = self._get_user_dir(user_id)
            if user_dir.exists():
                import shutil
                shutil.rmtree(user_dir)

            PREFERENCE_OPERATIONS.labels(operation="delete", status="success").inc()
            self._update_user_count()
            return True

        except Exception as e:
            PREFERENCE_OPERATIONS.labels(operation="delete", status="failed").inc()
            print(f"Failed to delete preferences for {user_id}: {e}")
            return False

    async def get_query_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[QueryHistoryItem]:
        """获取查询历史"""
        try:
            file_path = self._get_history_file(user_id)
            if not file_path.exists():
                return []

            with self._lock:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            items = [QueryHistoryItem(**item) for item in data]

            # 按时间倒序排列
            items.sort(key=lambda x: x.timestamp, reverse=True)

            # 分页
            return items[offset:offset + limit]

        except Exception as e:
            print(f"Failed to get query history for {user_id}: {e}")
            return []

    async def add_query_history(
        self,
        user_id: str,
        item: QueryHistoryItem,
    ) -> bool:
        """添加查询历史"""
        try:
            file_path = self._get_history_file(user_id)

            # 读取现有历史
            history = []
            if file_path.exists():
                with self._lock:
                    with open(file_path, "r", encoding="utf-8") as f:
                        history = json.load(f)

            # 添加新项
            history.append(asdict(item))

            # 获取用户偏好中的最大历史数
            prefs = await self.get_preferences(user_id)
            max_items = prefs.query.max_history_items if prefs else 100

            # 限制历史数量
            if len(history) > max_items:
                # 保留收藏的项
                favorites = [h for h in history if h.get("is_favorite")]
                non_favorites = [h for h in history if not h.get("is_favorite")]

                # 按时间排序，保留最新的
                non_favorites.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                non_favorites = non_favorites[:max_items - len(favorites)]

                history = favorites + non_favorites

            with self._lock:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"Failed to add query history for {user_id}: {e}")
            return False

    async def delete_query_history(
        self,
        user_id: str,
        item_id: Optional[str] = None,
    ) -> bool:
        """删除查询历史"""
        try:
            file_path = self._get_history_file(user_id)

            if item_id is None:
                # 删除所有历史
                if file_path.exists():
                    os.remove(file_path)
            else:
                # 删除指定项
                if file_path.exists():
                    with self._lock:
                        with open(file_path, "r", encoding="utf-8") as f:
                            history = json.load(f)

                    history = [h for h in history if h.get("id") != item_id]

                    with self._lock:
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(history, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"Failed to delete query history for {user_id}: {e}")
            return False

    async def list_users(self) -> list[str]:
        """列出所有用户"""
        users = []
        for user_dir in self.data_dir.iterdir():
            if user_dir.is_dir():
                prefs_file = user_dir / "preferences.json"
                if prefs_file.exists():
                    try:
                        with open(prefs_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        users.append(data.get("user_id", user_dir.name))
                    except Exception:
                        pass
        return users

    def _update_user_count(self):
        """更新用户计数"""
        try:
            count = len([d for d in self.data_dir.iterdir() if d.is_dir()])
            ACTIVE_USERS.set(count)
        except Exception:
            pass


# ============================================
# 用户偏好管理器
# ============================================

class UserPreferenceManager:
    """
    用户偏好管理器

    提供统一的偏好管理接口
    """

    def __init__(
        self,
        backend: Optional[PreferenceStorageBackend] = None,
        data_dir: str = "data/user_preferences",
    ):
        self.backend = backend or FileStorageBackend(data_dir)
        self._cache: dict[str, UserPreferences] = {}
        self._cache_ttl = 300  # 5 分钟缓存
        self._cache_timestamps: dict[str, datetime] = {}

    async def get_preferences(
        self,
        user_id: str,
        create_if_missing: bool = True,
    ) -> UserPreferences:
        """
        获取用户偏好

        Args:
            user_id: 用户 ID
            create_if_missing: 如果不存在是否创建默认偏好

        Returns:
            用户偏好
        """
        # 检查缓存
        if user_id in self._cache:
            cache_time = self._cache_timestamps.get(user_id)
            if cache_time and (datetime.utcnow() - cache_time).seconds < self._cache_ttl:
                return self._cache[user_id]

        # 从后端获取
        prefs = await self.backend.get_preferences(user_id)

        if prefs is None and create_if_missing:
            # 创建默认偏好
            prefs = UserPreferences(user_id=user_id)
            await self.backend.save_preferences(prefs)

        if prefs:
            # 更新缓存
            self._cache[user_id] = prefs
            self._cache_timestamps[user_id] = datetime.utcnow()

        return prefs or UserPreferences(user_id=user_id)

    async def update_preferences(
        self,
        user_id: str,
        updates: dict[str, Any],
    ) -> UserPreferences:
        """
        更新用户偏好

        Args:
            user_id: 用户 ID
            updates: 更新内容

        Returns:
            更新后的偏好
        """
        prefs = await self.get_preferences(user_id)

        # 应用更新
        for key, value in updates.items():
            if key == "visualization" and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(prefs.visualization, k):
                        setattr(prefs.visualization, k, v)
            elif key == "query" and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(prefs.query, k):
                        setattr(prefs.query, k, v)
            elif key == "ui" and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(prefs.ui, k):
                        setattr(prefs.ui, k, v)
            elif key == "custom_settings" and isinstance(value, dict):
                prefs.custom_settings.update(value)

        # 保存
        await self.backend.save_preferences(prefs)

        # 更新缓存
        self._cache[user_id] = prefs
        self._cache_timestamps[user_id] = datetime.utcnow()

        return prefs

    async def reset_preferences(self, user_id: str) -> UserPreferences:
        """重置用户偏好为默认值"""
        prefs = UserPreferences(user_id=user_id)
        await self.backend.save_preferences(prefs)

        # 更新缓存
        self._cache[user_id] = prefs
        self._cache_timestamps[user_id] = datetime.utcnow()

        return prefs

    async def delete_user(self, user_id: str) -> bool:
        """删除用户所有数据"""
        # 清除缓存
        if user_id in self._cache:
            del self._cache[user_id]
        if user_id in self._cache_timestamps:
            del self._cache_timestamps[user_id]

        return await self.backend.delete_preferences(user_id)

    async def record_query(
        self,
        user_id: str,
        query: str,
        sql: str,
        database: str,
        success: bool,
        row_count: int = 0,
        execution_time_ms: float = 0,
        chart_type: Optional[str] = None,
    ) -> bool:
        """
        记录查询历史

        Args:
            user_id: 用户 ID
            query: 自然语言查询
            sql: 生成的 SQL
            database: 数据库名称
            success: 是否成功
            row_count: 返回行数
            execution_time_ms: 执行时间
            chart_type: 图表类型

        Returns:
            是否成功
        """
        # 检查用户是否启用历史记录
        prefs = await self.get_preferences(user_id)
        if not prefs.query.save_history:
            return True

        item = QueryHistoryItem(
            id="",
            query=query,
            sql=sql,
            database=database,
            timestamp=datetime.utcnow().isoformat(),
            success=success,
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            chart_type=chart_type,
        )

        return await self.backend.add_query_history(user_id, item)

    async def get_query_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[QueryHistoryItem]:
        """获取查询历史"""
        return await self.backend.get_query_history(user_id, limit, offset)

    async def toggle_favorite(
        self,
        user_id: str,
        item_id: str,
    ) -> bool:
        """切换收藏状态"""
        history = await self.backend.get_query_history(user_id, limit=1000)

        for item in history:
            if item.id == item_id:
                item.is_favorite = not item.is_favorite
                # 重新保存
                await self.backend.delete_query_history(user_id)
                for h in history:
                    await self.backend.add_query_history(user_id, h)
                return True

        return False

    async def clear_history(self, user_id: str) -> bool:
        """清除查询历史"""
        return await self.backend.delete_query_history(user_id)

    def clear_cache(self, user_id: Optional[str] = None):
        """清除缓存"""
        if user_id:
            if user_id in self._cache:
                del self._cache[user_id]
            if user_id in self._cache_timestamps:
                del self._cache_timestamps[user_id]
        else:
            self._cache.clear()
            self._cache_timestamps.clear()


# ============================================
# 全局实例
# ============================================

_preference_manager: Optional[UserPreferenceManager] = None


def get_preference_manager() -> UserPreferenceManager:
    """获取偏好管理器单例"""
    global _preference_manager
    if _preference_manager is None:
        _preference_manager = UserPreferenceManager()
    return _preference_manager


# ============================================
# 导出
# ============================================

__all__ = [
    "UserPreferences",
    "VisualizationPreferences",
    "QueryPreferences",
    "UIPreferences",
    "QueryHistoryItem",
    "ChartPreference",
    "ThemePreference",
    "PreferenceStorageBackend",
    "FileStorageBackend",
    "UserPreferenceManager",
    "get_preference_manager",
    "PREFERENCE_OPERATIONS",
    "ACTIVE_USERS",
]
