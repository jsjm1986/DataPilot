# -*- coding: utf-8 -*-
"""Security Module"""

from .injection_guard import InjectionGuard, check_sql_injection
from .sanitizer import PIISanitizer, sanitize_output
from .auth import (
    User,
    AuthResult,
    create_access_token,
    authenticate_user,
    authenticate_api_key,
    get_current_user,
    require_auth,
    optional_auth,
    require_roles,
    generate_api_key,
)

__all__ = [
    "InjectionGuard",
    "check_sql_injection",
    "PIISanitizer",
    "sanitize_output",
    "User",
    "AuthResult",
    "create_access_token",
    "authenticate_user",
    "authenticate_api_key",
    "get_current_user",
    "require_auth",
    "optional_auth",
    "require_roles",
    "generate_api_key",
]
