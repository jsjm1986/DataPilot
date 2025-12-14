# -*- coding: utf-8 -*-
"""
Sandbox executor for running Python visualization code.

支持两种执行方式:
1. E2B Cloud Sandbox (需要 API Key)
2. Docker 本地沙箱 (需要 Docker 环境)
3. 子进程沙箱 (最后的回退方案，安全性较低)
"""

import asyncio
import subprocess
import tempfile
import os
from typing import Optional

from ..config.settings import get_settings


async def run_python_in_sandbox(code: str) -> Optional[dict]:
    """
    Run Python code in sandbox.

    优先级:
    1. E2B Cloud Sandbox (如果配置了 API Key)
    2. Docker 本地沙箱 (如果 Docker 可用)
    3. 子进程沙箱 (回退方案)

    Args:
        code: Python 代码

    Returns:
        执行结果字典，包含 success, stdout, stderr, error
    """
    settings = get_settings()

    # 1. 尝试 E2B
    if settings.e2b_api_key:
        result = await _run_in_e2b(code, settings)
        if result.get("success") or "SDK not available" not in result.get("error", ""):
            return result

    # 2. 尝试 Docker
    if _is_docker_available():
        result = await _run_in_docker(code, settings)
        if result.get("success") or "Docker" not in result.get("error", ""):
            return result

    # 3. 回退到子进程 (仅开发环境)
    if settings.is_development:
        return await _run_in_subprocess(code, settings)

    return {"success": False, "error": "No sandbox available. Configure E2B API key or install Docker."}


async def _run_in_e2b(code: str, settings) -> dict:
    """使用 E2B Cloud Sandbox 执行代码"""
    try:
        from e2b_code_interpreter import AsyncSandbox  # type: ignore
    except ImportError as e:
        return {"success": False, "error": f"E2B SDK not available: {e}"}

    sandbox = None
    try:
        # 新版 E2B SDK 使用 AsyncSandbox
        sandbox = await AsyncSandbox.create(
            api_key=settings.e2b_api_key,
            timeout=settings.sandbox_timeout_seconds,
        )

        # 执行代码
        execution = await sandbox.run_code(code)

        # 提取结果
        error_msg = None

        # 处理执行结果
        if execution.error:
            error_msg = str(execution.error)

        # 收集输出 - logs 是 Logs 对象，有 stdout 和 stderr 列表属性
        stdout_str = ""
        stderr_str = ""
        if execution.logs:
            if hasattr(execution.logs, 'stdout') and execution.logs.stdout:
                stdout_str = "".join(execution.logs.stdout)
            if hasattr(execution.logs, 'stderr') and execution.logs.stderr:
                stderr_str = "".join(execution.logs.stderr)

        # 也检查 results (用于 Jupyter 风格的输出)
        if execution.results:
            for result in execution.results:
                if hasattr(result, 'text') and result.text:
                    stdout_str += result.text

        return {
            "success": error_msg is None,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "error": error_msg,
            "executor": "e2b",
        }
    except Exception as e:
        return {"success": False, "error": str(e), "executor": "e2b"}
    finally:
        if sandbox:
            try:
                await sandbox.kill()
            except Exception:
                pass


def _is_docker_available() -> bool:
    """检查 Docker 是否可用"""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


async def _run_in_docker(code: str, settings) -> dict:
    """使用 Docker 本地沙箱执行代码"""
    try:
        # 创建临时文件存放代码
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            code_file = f.name

        try:
            # Docker 运行参数
            docker_cmd = [
                "docker", "run",
                "--rm",                          # 运行后删除容器
                "--network", "none",             # 禁用网络
                "--memory", "512m",              # 内存限制
                "--cpus", "0.5",                 # CPU 限制
                "--read-only",                   # 只读文件系统
                "--tmpfs", "/tmp:size=100m",    # 临时文件系统
                "-v", f"{code_file}:/app/code.py:ro",  # 挂载代码文件
                "-w", "/app",
                "python:3.11-slim",
                "python", "/app/code.py",
            ]

            # 异步执行
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=settings.sandbox_timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": "Execution timeout",
                    "executor": "docker",
                }

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "error": None if process.returncode == 0 else f"Exit code: {process.returncode}",
                "executor": "docker",
            }

        finally:
            # 清理临时文件
            try:
                os.unlink(code_file)
            except Exception:
                pass

    except Exception as e:
        return {"success": False, "error": f"Docker execution failed: {e}", "executor": "docker"}


async def _run_in_subprocess(code: str, settings) -> dict:
    """
    使用子进程执行代码 (仅开发环境)

    警告: 这种方式安全性较低，仅用于开发测试
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            code_file = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                "python", code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=settings.sandbox_timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": "Execution timeout",
                    "executor": "subprocess",
                }

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "error": None if process.returncode == 0 else f"Exit code: {process.returncode}",
                "executor": "subprocess",
            }

        finally:
            try:
                os.unlink(code_file)
            except Exception:
                pass

    except Exception as e:
        return {"success": False, "error": f"Subprocess execution failed: {e}", "executor": "subprocess"}


# ============================================
# AST 安全分析 (替代字符串匹配)
# ============================================

import ast
from dataclasses import dataclass, field


@dataclass
class SecurityAnalysisResult:
    """安全分析结果"""
    is_safe: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    analyzed_imports: list[str] = field(default_factory=list)
    analyzed_calls: list[str] = field(default_factory=list)


# 危险模块黑名单
DANGEROUS_MODULES = {
    # 系统操作
    "os", "sys", "subprocess", "shutil", "pathlib",
    # 网络
    "socket", "urllib", "requests", "httplib", "ftplib",
    # 代码执行
    "code", "codeop", "compile", "importlib",
    # 文件操作
    "io", "tempfile", "fileinput",
    # 进程
    "multiprocessing", "threading", "concurrent",
    # 其他危险模块
    "ctypes", "pickle", "marshal", "shelve",
    "pty", "tty", "termios", "fcntl",
    "resource", "sysconfig", "platform",
}

# 允许的安全模块白名单
ALLOWED_MODULES = {
    # 数据处理
    "json", "csv", "re", "math", "statistics",
    "decimal", "fractions", "random", "itertools",
    "functools", "operator", "collections",
    # 数据分析
    "pandas", "numpy", "scipy",
    # 可视化
    "matplotlib", "plotly", "seaborn", "altair",
    # 日期时间
    "datetime", "time", "calendar",
    # 类型
    "typing", "dataclasses", "enum",
}

# 危险内置函数
DANGEROUS_BUILTINS = {
    "eval", "exec", "compile", "__import__",
    "open", "input", "raw_input",
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit",
}

# 危险属性访问
DANGEROUS_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__",
    "__mro__", "__code__", "__globals__",
    "__builtins__", "__import__", "__loader__",
    "__spec__", "__file__", "__cached__",
}


class ASTSecurityAnalyzer(ast.NodeVisitor):
    """
    AST 安全分析器

    使用 AST 遍历代码，检测危险操作
    比字符串匹配更准确，不易被绕过
    """

    def __init__(self, strict_mode: bool = True):
        """
        初始化分析器

        Args:
            strict_mode: 严格模式 (只允许白名单模块)
        """
        self.strict_mode = strict_mode
        self.violations: list[str] = []
        self.warnings: list[str] = []
        self.imports: list[str] = []
        self.calls: list[str] = []

    def analyze(self, code: str) -> SecurityAnalysisResult:
        """
        分析代码安全性

        Args:
            code: Python 代码

        Returns:
            安全分析结果
        """
        try:
            tree = ast.parse(code)
            self.visit(tree)

            return SecurityAnalysisResult(
                is_safe=len(self.violations) == 0,
                violations=self.violations,
                warnings=self.warnings,
                analyzed_imports=self.imports,
                analyzed_calls=self.calls,
            )
        except SyntaxError as e:
            return SecurityAnalysisResult(
                is_safe=False,
                violations=[f"Syntax error: {e}"],
            )

    def visit_Import(self, node: ast.Import):
        """检查 import 语句"""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self.imports.append(alias.name)

            if module_name in DANGEROUS_MODULES:
                self.violations.append(
                    f"Dangerous module import: {alias.name}"
                )
            elif self.strict_mode and module_name not in ALLOWED_MODULES:
                self.warnings.append(
                    f"Unknown module (not in whitelist): {alias.name}"
                )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """检查 from ... import 语句"""
        if node.module:
            module_name = node.module.split('.')[0]
            self.imports.append(node.module)

            if module_name in DANGEROUS_MODULES:
                self.violations.append(
                    f"Dangerous module import: from {node.module}"
                )
            elif self.strict_mode and module_name not in ALLOWED_MODULES:
                self.warnings.append(
                    f"Unknown module (not in whitelist): {node.module}"
                )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """检查函数调用"""
        func_name = self._get_call_name(node)
        if func_name:
            self.calls.append(func_name)

            # 检查危险内置函数
            if func_name in DANGEROUS_BUILTINS:
                self.violations.append(
                    f"Dangerous builtin call: {func_name}()"
                )

            # 检查 __import__
            if func_name == "__import__":
                self.violations.append(
                    "Dynamic import detected: __import__()"
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """检查属性访问"""
        attr_name = node.attr

        if attr_name in DANGEROUS_ATTRIBUTES:
            self.violations.append(
                f"Dangerous attribute access: .{attr_name}"
            )

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """检查变量名"""
        # 检查直接访问危险内置
        if node.id in DANGEROUS_BUILTINS:
            # 只有在 Load 上下文中才是危险的
            if isinstance(node.ctx, ast.Load):
                self.warnings.append(
                    f"Reference to dangerous builtin: {node.id}"
                )

        self.generic_visit(node)

    def visit_Exec(self, node):
        """检查 exec 语句 (Python 2)"""
        self.violations.append("exec statement detected")
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str:
        """获取函数调用名称"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


def validate_code(code: str, strict_mode: bool = False) -> tuple[bool, str]:
    """
    验证代码安全性 (使用 AST 分析)

    Args:
        code: Python 代码
        strict_mode: 严格模式 (只允许白名单模块)

    Returns:
        (is_safe, reason)
    """
    analyzer = ASTSecurityAnalyzer(strict_mode=strict_mode)
    result = analyzer.analyze(code)

    if result.is_safe:
        if result.warnings:
            return True, f"Code is safe (warnings: {', '.join(result.warnings[:3])})"
        return True, "Code is safe"
    else:
        return False, f"Security violations: {'; '.join(result.violations[:3])}"


def validate_code_detailed(code: str, strict_mode: bool = False) -> SecurityAnalysisResult:
    """
    详细的代码安全验证

    Args:
        code: Python 代码
        strict_mode: 严格模式

    Returns:
        完整的安全分析结果
    """
    analyzer = ASTSecurityAnalyzer(strict_mode=strict_mode)
    return analyzer.analyze(code)


async def run_python_safe(code: str) -> dict:
    """
    安全执行 Python 代码

    先验证代码安全性，再执行
    """
    is_safe, reason = validate_code(code)
    if not is_safe:
        return {"success": False, "error": reason, "executor": "validator"}

    return await run_python_in_sandbox(code)


__all__ = [
    "run_python_in_sandbox",
    "run_python_safe",
    "validate_code",
    "validate_code_detailed",
    "ASTSecurityAnalyzer",
    "SecurityAnalysisResult",
    "DANGEROUS_MODULES",
    "ALLOWED_MODULES",
    "DANGEROUS_BUILTINS",
]
