"""
DataPilot CLI 结果展示模块
使用 rich 库美化终端输出
"""

import json
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ResultDisplay:
    """CLI 结果展示器"""

    def __init__(self):
        if RICH_AVAILABLE:
            # 使用 force_terminal=True 和 no_color=False 确保颜色输出
            # legacy_windows=False 禁用 Windows 旧版渲染避免编码问题
            self.console = Console(force_terminal=True, legacy_windows=False)
        else:
            self.console = None

    def print(self, message: str, style: str = None):
        """打印消息"""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)

    def print_error(self, message: str):
        """打印错误消息"""
        if self.console:
            self.console.print(f"[X] {message}", style="bold red")
        else:
            print(f"[ERROR] {message}")

    def print_success(self, message: str):
        """打印成功消息"""
        if self.console:
            self.console.print(f"[OK] {message}", style="bold green")
        else:
            print(f"[OK] {message}")

    def print_info(self, message: str):
        """打印信息"""
        if self.console:
            self.console.print(f"[*] {message}", style="blue")
        else:
            print(f"[INFO] {message}")

    def print_warning(self, message: str):
        """打印警告"""
        if self.console:
            self.console.print(f"[!] {message}", style="yellow")
        else:
            print(f"[WARN] {message}")

    def print_sql(self, sql: str):
        """打印 SQL 语句（语法高亮）"""
        if not sql:
            return

        if self.console and RICH_AVAILABLE:
            syntax = Syntax(sql, "sql", theme="monokai", line_numbers=False)
            panel = Panel(
                syntax,
                title="[bold blue]生成的 SQL[/bold blue]",
                border_style="blue",
                box=box.ROUNDED
            )
            self.console.print(panel)
        else:
            print("\n=== 生成的 SQL ===")
            print(sql)
            print("==================\n")

    def print_data_table(self, data: list[dict], row_count: int = None):
        """打印数据表格"""
        if not data:
            self.print_warning("查询结果为空")
            return

        if self.console and RICH_AVAILABLE:
            # 获取列名
            columns = list(data[0].keys())

            # 创建表格
            table = Table(
                title=f"[bold]查询结果[/bold] ({row_count or len(data)} 行)",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan"
            )

            # 添加序号列
            table.add_column("#", style="dim", width=4)

            # 添加数据列
            for col in columns:
                table.add_column(col)

            # 添加数据行（最多显示50行）
            display_data = data[:50]
            for i, row in enumerate(display_data, 1):
                values = [str(i)] + [str(v) if v is not None else "" for v in row.values()]
                table.add_row(*values)

            if len(data) > 50:
                table.add_row("...", *["..." for _ in columns])

            self.console.print(table)

            if len(data) > 50:
                self.print_info(f"仅显示前 50 行，共 {len(data)} 行")
        else:
            print(f"\n=== 查询结果 ({row_count or len(data)} 行) ===")
            columns = list(data[0].keys())

            # 打印表头
            header = " | ".join(columns)
            print(header)
            print("-" * len(header))

            # 打印数据行
            for i, row in enumerate(data[:50], 1):
                values = [str(v) if v is not None else "" for v in row.values()]
                print(f"{i}. " + " | ".join(values))

            if len(data) > 50:
                print(f"... (仅显示前 50 行，共 {len(data)} 行)")
            print()

    def print_chart_config(self, chart_config: dict):
        """打印图表配置"""
        if not chart_config:
            return

        if self.console and RICH_AVAILABLE:
            json_str = json.dumps(chart_config, ensure_ascii=False, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            panel = Panel(
                syntax,
                title="[bold green]图表配置 (ECharts)[/bold green]",
                border_style="green",
                box=box.ROUNDED
            )
            self.console.print(panel)
        else:
            print("\n=== 图表配置 ===")
            print(json.dumps(chart_config, ensure_ascii=False, indent=2))
            print("================\n")

    def print_insight(self, insight: str):
        """打印数据洞察"""
        if not insight:
            return

        if self.console and RICH_AVAILABLE:
            panel = Panel(
                insight,
                title="[bold magenta]数据洞察[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            )
            self.console.print(panel)
        else:
            print("\n=== 数据洞察 ===")
            print(insight)
            print("================\n")

    def print_clarify_options(self, clarify_options: dict) -> int:
        """打印澄清选项并获取用户选择"""
        if not clarify_options:
            return -1

        question = clarify_options.get("question", "请选择:")
        options = clarify_options.get("options", [])

        if self.console and RICH_AVAILABLE:
            self.console.print(f"\n[bold yellow]? {question}[/bold yellow]\n")

            for i, opt in enumerate(options, 1):
                # 支持字符串或字典格式的选项
                if isinstance(opt, str):
                    label = opt
                    desc = ""
                else:
                    label = opt.get("label", opt.get("value", f"Option {i}"))
                    desc = opt.get("description", "")
                if desc:
                    self.console.print(f"  [cyan]{i}[/cyan]. {label} - [dim]{desc}[/dim]")
                else:
                    self.console.print(f"  [cyan]{i}[/cyan]. {label}")

            self.console.print()
        else:
            print(f"\n? {question}\n")
            for i, opt in enumerate(options, 1):
                # 支持字符串或字典格式的选项
                if isinstance(opt, str):
                    label = opt
                else:
                    label = opt.get("label", opt.get("value", f"Option {i}"))
                print(f"  {i}. {label}")
            print()

        # 获取用户输入
        while True:
            try:
                choice = input("请输入选项编号: ").strip()
                if not choice:
                    continue
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return choice_num - 1
                else:
                    self.print_error(f"请输入 1-{len(options)} 之间的数字")
            except ValueError:
                self.print_error("请输入有效的数字")
            except KeyboardInterrupt:
                return -1

    def print_timing(self, duration_ms: float):
        """打印耗时"""
        if duration_ms:
            seconds = duration_ms / 1000
            if self.console:
                self.console.print(f"\n[Time] {seconds:.2f} s", style="dim")
            else:
                print(f"\nTime: {seconds:.2f} s")

    def create_progress(self) -> Any:
        """创建进度指示器"""
        if self.console and RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            )
        return None

    def print_welcome(self, database: str = None):
        """打印欢迎信息"""
        if self.console and RICH_AVAILABLE:
            title = "[bold blue]DataPilot CLI[/bold blue] - 自然语言数据查询"
            if database:
                title += f"\n[dim]当前数据库: {database}[/dim]"

            panel = Panel(
                title,
                box=box.DOUBLE,
                border_style="blue"
            )
            self.console.print(panel)
            self.console.print("[dim]输入自然语言问题进行查询，输入 'exit' 或 'quit' 退出[/dim]\n")
        else:
            print("=" * 50)
            print("DataPilot CLI - 自然语言数据查询")
            if database:
                print(f"当前数据库: {database}")
            print("=" * 50)
            print("输入自然语言问题进行查询，输入 'exit' 或 'quit' 退出\n")

    def print_databases(self, databases: list[str]):
        """打印可用数据库列表"""
        if not databases:
            self.print_warning("没有可用的数据库")
            return

        if self.console and RICH_AVAILABLE:
            table = Table(title="可用数据库", box=box.SIMPLE)
            table.add_column("#", style="dim", width=4)
            table.add_column("数据库名称", style="cyan")

            for i, db in enumerate(databases, 1):
                table.add_row(str(i), db)

            self.console.print(table)
        else:
            print("\n可用数据库:")
            for i, db in enumerate(databases, 1):
                print(f"  {i}. {db}")
            print()
