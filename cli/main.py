#!/usr/bin/env python
"""
DataPilot CLI - 命令行自然语言数据查询工具

使用方式:
    # 单次查询
    python -m datapilot.cli "查询上个月销售额最高的前10个产品" --db sales_db

    # 交互式模式
    python -m datapilot.cli --interactive --db sales_db

    # 显示更多信息
    python -m datapilot.cli "..." --db sales_db --show-sql --show-chart
"""

import argparse
import asyncio
import sys
import csv
import json
from pathlib import Path

from .runner import QueryRunner, QueryResult
from .display import ResultDisplay


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DataPilot CLI - 自然语言数据查询工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m datapilot.cli "查询销售额前10的产品" --db sales
  python -m datapilot.cli --interactive --db sales
  python -m datapilot.cli "查询用户数" --db analytics --show-sql
  python -m datapilot.cli "月度趋势" --db sales --output result.csv
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="自然语言查询（交互模式下可省略）"
    )

    parser.add_argument(
        "--db", "-d",
        dest="database",
        default="default",
        help="目标数据库名称（默认: default）"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="进入交互式模式"
    )

    parser.add_argument(
        "--show-sql", "-s",
        action="store_true",
        help="显示生成的 SQL"
    )

    parser.add_argument(
        "--show-chart", "-c",
        action="store_true",
        help="显示图表配置"
    )

    parser.add_argument(
        "--show-insight",
        action="store_true",
        help="显示数据洞察"
    )

    parser.add_argument(
        "--output", "-o",
        dest="output_file",
        help="导出结果到文件（支持 .csv 和 .json）"
    )

    parser.add_argument(
        "--viz-mode",
        choices=["echarts", "codeact"],
        default="echarts",
        help="可视化模式（默认: echarts）"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用语义缓存"
    )

    parser.add_argument(
        "--list-db",
        action="store_true",
        help="列出可用数据库"
    )

    return parser.parse_args()


def export_result(result: QueryResult, output_file: str, display: ResultDisplay):
    """导出结果到文件"""
    if not result.data:
        display.print_warning("没有数据可导出")
        return

    path = Path(output_file)
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                if result.data:
                    writer = csv.DictWriter(f, fieldnames=result.data[0].keys())
                    writer.writeheader()
                    writer.writerows(result.data)
            display.print_success(f"已导出到 {output_file}")

        elif suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "sql": result.sql,
                    "data": result.data,
                    "row_count": result.row_count,
                    "chart_config": result.chart_config,
                }, f, ensure_ascii=False, indent=2)
            display.print_success(f"已导出到 {output_file}")

        else:
            display.print_error(f"不支持的文件格式: {suffix}（支持 .csv 和 .json）")

    except Exception as e:
        display.print_error(f"导出失败: {e}")


def handle_result(
    result: QueryResult,
    display: ResultDisplay,
    show_sql: bool = False,
    show_chart: bool = False,
    show_insight: bool = False,
    output_file: str = None,
):
    """处理并显示查询结果"""
    if not result.success:
        if result.error:
            display.print_error(result.error)
        return

    # 显示 SQL
    if show_sql and result.sql:
        display.print_sql(result.sql)

    # 显示数据表格
    if result.data:
        display.print_data_table(result.data, result.row_count)

    # 显示洞察
    if show_insight and result.insight:
        display.print_insight(result.insight)

    # 显示图表配置
    if show_chart and result.chart_config:
        display.print_chart_config(result.chart_config)

    # 显示耗时
    display.print_timing(result.duration_ms)

    # 导出结果
    if output_file:
        export_result(result, output_file, display)


async def run_single_query(
    query: str,
    runner: QueryRunner,
    display: ResultDisplay,
    show_sql: bool = False,
    show_chart: bool = False,
    show_insight: bool = False,
    output_file: str = None,
):
    """执行单次查询"""
    display.print_info("正在分析查询...")

    result = await runner.run_query(query)

    # 处理澄清请求
    if result.clarify_needed and result.clarify_options:
        choice = display.print_clarify_options(result.clarify_options)
        if choice >= 0:
            display.print_info("正在继续执行...")
            result = await runner.continue_with_clarify(choice)
        else:
            display.print_warning("已取消查询")
            return

    handle_result(
        result,
        display,
        show_sql=show_sql,
        show_chart=show_chart,
        show_insight=show_insight,
        output_file=output_file,
    )


async def run_interactive(
    runner: QueryRunner,
    display: ResultDisplay,
    show_sql: bool = False,
    show_chart: bool = False,
    show_insight: bool = False,
):
    """运行交互式模式"""
    display.print_welcome(runner.database)

    while True:
        try:
            # 获取用户输入
            query = input("\n> ").strip()

            if not query:
                continue

            # 退出命令
            if query.lower() in ("exit", "quit", "q", "退出"):
                display.print_info("再见！")
                break

            # 帮助命令
            if query.lower() in ("help", "h", "帮助", "?"):
                print("""
可用命令:
  exit, quit, q    退出程序
  help, h, ?       显示帮助
  db <name>        切换数据库
  list db          列出可用数据库
  sql on/off       开启/关闭 SQL 显示
  chart on/off     开启/关闭图表配置显示
  其他             自然语言查询
                """)
                continue

            # 切换数据库
            if query.lower().startswith("db "):
                new_db = query[3:].strip()
                runner.set_database(new_db)
                display.print_success(f"已切换到数据库: {new_db}")
                continue

            # 列出数据库
            if query.lower() in ("list db", "list databases", "列出数据库"):
                databases = runner.list_databases()
                display.print_databases(databases)
                continue

            # SQL 显示开关
            if query.lower() == "sql on":
                show_sql = True
                display.print_success("已开启 SQL 显示")
                continue
            if query.lower() == "sql off":
                show_sql = False
                display.print_success("已关闭 SQL 显示")
                continue

            # 图表显示开关
            if query.lower() == "chart on":
                show_chart = True
                display.print_success("已开启图表配置显示")
                continue
            if query.lower() == "chart off":
                show_chart = False
                display.print_success("已关闭图表配置显示")
                continue

            # 执行查询
            await run_single_query(
                query,
                runner,
                display,
                show_sql=show_sql,
                show_chart=show_chart,
                show_insight=show_insight,
            )

        except KeyboardInterrupt:
            print()
            display.print_info("按 Ctrl+C 再次退出，或输入 'exit' 退出")
            try:
                input()
            except KeyboardInterrupt:
                print()
                display.print_info("再见！")
                break
        except EOFError:
            print()
            display.print_info("再见！")
            break


def main():
    """CLI 入口函数"""
    args = parse_args()

    display = ResultDisplay()

    # 创建查询执行器
    runner = QueryRunner(
        database=args.database,
        viz_mode=args.viz_mode,
        include_cache=not args.no_cache,
    )

    # 列出数据库
    if args.list_db:
        databases = runner.list_databases()
        display.print_databases(databases)
        return

    # 交互式模式
    if args.interactive:
        asyncio.run(run_interactive(
            runner,
            display,
            show_sql=args.show_sql,
            show_chart=args.show_chart,
            show_insight=args.show_insight,
        ))
        return

    # 单次查询模式
    if args.query:
        asyncio.run(run_single_query(
            args.query,
            runner,
            display,
            show_sql=args.show_sql,
            show_chart=args.show_chart,
            show_insight=args.show_insight,
            output_file=args.output_file,
        ))
        return

    # 没有查询也没有交互模式，显示帮助
    display.print_error("请提供查询内容或使用 --interactive 进入交互模式")
    display.print_info("使用 --help 查看帮助")


if __name__ == "__main__":
    main()
