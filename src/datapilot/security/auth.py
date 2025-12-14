# -*- coding: utf-8 -*-
"""
认证鉴权模块
支持 JWT Token 认证和 API Key 认证
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from functools import wraps

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel
import jwt

try:
    from passlib.context import CryptContext
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

from ..config.settings import get_settings


# ============================================
# 配置
# ============================================

# JWT 配置（改为从 Settings 读取，必须显式配置）
settings = get_settings()
JWT_SECRET_KEY = settings.__dict__.get("jwt_secret_key", "") or settings.__dict__.get("jwt_secret", "") or ""
# Do not set a development fallback secret in the open-source export.
# Production users must set `JWT_SECRET_KEY` via environment or config.
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

# API Key 配置
API_KEY_HEADER = "X-API-Key"

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


# ============================================
# 数据模型
# ============================================

class User(BaseModel):
    """用户模型"""
    user_id: str
    username: str
    tenant_id: str = "default"
    roles: list[str] = ["user"]
    is_active: bool = True


class TokenPayload(BaseModel):
    """Token 载荷"""
    sub: str  # user_id
    username: str
    tenant_id: str
    roles: list[str]
    exp: datetime
    iat: datetime


class AuthResult(BaseModel):
    """认证结果"""
    authenticated: bool
    user: Optional[User] = None
    method: Optional[str] = None  # "jwt" or "api_key"
    error: Optional[str] = None


# ============================================
# 模拟用户存储 (生产环境应使用数据库)
# ============================================

MOCK_USERS: dict[str, dict] = {}

# API Key 存储（优先读取配置，未配置时回退开发用 demo key）
_api_keys: dict[str, dict] = {}
if settings.admin_api_key:
    _api_keys[settings.admin_api_key] = {
        "user_id": "user_admin",
        "username": "admin",
        "tenant_id": "default",
        "roles": ["admin", "user"],
        "is_active": True,
    }
if settings.user_api_key:
    _api_keys[settings.user_api_key] = {
        "user_id": "user_standard",
        "username": "api_user",
        "tenant_id": "default",
        "roles": ["user"],
        "is_active": True,
    }
if not _api_keys:
    # No API keys configured. Do not populate demo keys in the open-source export.
    # Users should set `ADMIN_API_KEY`/`USER_API_KEY` via environment in their deployment.
    _api_keys = {}


# ============================================
# JWT 工具函数
# ============================================

def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建 JWT Access Token

    Args:
        user: 用户对象
        expires_delta: 过期时间增量

    Returns:
        JWT Token 字符串
    """
    now = datetime.utcnow()
    expire = now + (expires_delta or timedelta(hours=JWT_EXPIRE_HOURS))

    payload = {
        "sub": user.user_id,
        "username": user.username,
        "tenant_id": user.tenant_id,
        "roles": user.roles,
        "exp": expire,
        "iat": now,
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[TokenPayload]:
    """
    解码 JWT Token

    Args:
        token: JWT Token 字符串

    Returns:
        Token 载荷，解码失败返回 None
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# 密码哈希上下文 (优先使用 bcrypt)
if BCRYPT_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
else:
    pwd_context = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码

    支持两种哈希格式:
    - bcrypt: 以 $2b$ 开头的哈希 (推荐)
    - SHA256: 64 字符的十六进制字符串 (向后兼容)
    """
    # 检测哈希类型
    if hashed_password.startswith("$2b$") or hashed_password.startswith("$2a$"):
        # bcrypt 哈希
        if pwd_context:
            return pwd_context.verify(plain_password, hashed_password)
        else:
            raise RuntimeError("bcrypt not available, please install passlib[bcrypt]")

    # SHA256 哈希 (向后兼容)
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def hash_password(plain_password: str) -> str:
    """
    哈希密码 (使用 bcrypt)

    Returns:
        bcrypt 哈希字符串
    """
    if pwd_context:
        return pwd_context.hash(plain_password)
    else:
        # 回退到 SHA256 (不推荐)
        return hashlib.sha256(plain_password.encode()).hexdigest()


# ============================================
# 认证函数
# ============================================

def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    用户名密码认证

    Args:
        username: 用户名
        password: 密码

    Returns:
        认证成功返回 User，失败返回 None
    """
    if not settings.allow_mock_users:
        return None
    user_data = MOCK_USERS.get(username)
    if not user_data:
        return None
    if not verify_password(password, user_data["password_hash"]):
        return None
    if not user_data["is_active"]:
        return None
    return User(
        user_id=user_data["user_id"],
        username=user_data["username"],
        tenant_id=user_data["tenant_id"],
        roles=user_data["roles"],
        is_active=user_data["is_active"],
    )


def authenticate_api_key(api_key: str) -> Optional[User]:
    """
    API Key 认证

    Args:
        api_key: API Key

    Returns:
        认证成功返回 User，失败返回 None
    """
    key_data = _api_keys.get(api_key)
    if not key_data:
        return None

    if not key_data["is_active"]:
        return None

    return User(
        user_id=key_data["user_id"],
        username=key_data["username"],
        tenant_id=key_data["tenant_id"],
        roles=key_data["roles"],
        is_active=True,
    )


def authenticate_jwt(token: str) -> Optional[User]:
    """
    JWT Token 认证

    Args:
        token: JWT Token

    Returns:
        认证成功返回 User，失败返回 None
    """
    payload = decode_access_token(token)
    if not payload:
        return None

    return User(
        user_id=payload.sub,
        username=payload.username,
        tenant_id=payload.tenant_id,
        roles=payload.roles,
        is_active=True,
    )


# ============================================
# FastAPI 依赖
# ============================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> AuthResult:
    """
    获取当前用户 (支持 JWT 和 API Key)

    优先级: JWT > API Key
    """
    # 1. 尝试 JWT 认证
    if credentials and credentials.credentials:
        user = authenticate_jwt(credentials.credentials)
        if user:
            return AuthResult(authenticated=True, user=user, method="jwt")
        return AuthResult(authenticated=False, error="Invalid or expired token")

    # 2. 尝试 API Key 认证
    if api_key:
        user = authenticate_api_key(api_key)
        if user:
            return AuthResult(authenticated=True, user=user, method="api_key")
        return AuthResult(authenticated=False, error="Invalid API key")

    # 3. 无认证信息
    return AuthResult(authenticated=False, error="No authentication provided")


async def require_auth(
    auth_result: AuthResult = Depends(get_current_user),
) -> User:
    """
    要求认证 (用于需要登录的接口)
    """
    if not auth_result.authenticated:
        raise HTTPException(
            status_code=401,
            detail=auth_result.error or "Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_result.user


async def optional_auth(
    auth_result: AuthResult = Depends(get_current_user),
) -> Optional[User]:
    """
    可选认证 (用于支持匿名访问的接口)
    """
    if auth_result.authenticated:
        return auth_result.user
    return None


def require_roles(*required_roles: str):
    """
    角色检查装饰器

    Usage:
        @app.get("/admin")
        @require_roles("admin")
        async def admin_endpoint(user: User = Depends(require_auth)):
            ...
    """
    async def role_checker(user: User = Depends(require_auth)) -> User:
        for role in required_roles:
            if role in user.roles:
                return user
        raise HTTPException(
            status_code=403,
            detail=f"Required roles: {', '.join(required_roles)}",
        )
    return role_checker


# ============================================
# 辅助函数
# ============================================

def generate_api_key(prefix: str = "sk-datapilot") -> str:
    """生成新的 API Key"""
    random_part = secrets.token_hex(16)
    return f"{prefix}-{random_part}"


def hash_api_key(api_key: str) -> str:
    """哈希 API Key (用于存储)"""
    return hashlib.sha256(api_key.encode()).hexdigest()


# ============================================
# 导出
# ============================================

__all__ = [
    "User",
    "TokenPayload",
    "AuthResult",
    "create_access_token",
    "decode_access_token",
    "authenticate_user",
    "authenticate_api_key",
    "authenticate_jwt",
    "get_current_user",
    "require_auth",
    "optional_auth",
    "require_roles",
    "generate_api_key",
    "hash_api_key",
    "hash_password",
    "verify_password",
]
