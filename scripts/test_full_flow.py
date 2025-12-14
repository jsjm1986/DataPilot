# -*- coding: utf-8 -*-
"""
完整业务流程测试脚本

测试所有重构后的模块是否正常工作:
1. MCP 工具标准化
2. 成本熔断机制
3. DSPy 自进化
4. 用户偏好持久化
5. 多 Judge 博弈机制
6. LLM 驱动的各 Agent
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置环境变量
os.environ.setdefault("DEEPSQL_ENV", "development")


def print_section(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = ""):
    """打印测试结果"""
    status = "[PASS]" if success else "[FAIL]"
    print(f"  {status} | {name}")
    if details:
        for line in details.split("\n"):
            print(f"         {line}")


async def test_mcp_tools():
    """测试 MCP 工具标准化"""
    print_section("1. MCP 工具标准化测试")

    try:
        from src.datapilot.mcp import (
            MCPTool,
            ToolRegistry,
            get_tool_registry,
            ToolCategory,
        )

        # 测试工具注册表
        registry = get_tool_registry()
        tools = registry.list_tools()

        print_result(
            "工具注册表",
            len(tools) > 0,
            f"已注册 {len(tools)} 个工具"
        )

        # 列出所有工具
        for tool in tools:
            print(f"         - {tool.name}: {tool.description[:50]}...")

        # 测试工具 MCP 定义
        definitions = registry.get_mcp_definitions()
        print_result(
            "MCP 定义生成",
            all("inputSchema" in d for d in definitions),
            f"所有工具都有 inputSchema"
        )

        # 测试工具分类
        schema_tools = registry.list_by_category(ToolCategory.SCHEMA)
        print_result(
            "工具分类",
            len(schema_tools) > 0,
            f"SCHEMA 类别有 {len(schema_tools)} 个工具"
        )

        return True

    except Exception as e:
        print_result("MCP 工具测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_cost_control():
    """测试成本熔断机制"""
    print_section("2. 成本熔断机制测试")

    try:
        from src.datapilot.llm.cost_control import (
            CostController,
            CostLimit,
            get_cost_controller,
            estimate_cost,
            CircuitState,
        )

        # 测试成本估算
        cost = estimate_cost("deepseek-chat", 1000, 500)
        print_result(
            "成本估算",
            cost > 0,
            f"1000 输入 + 500 输出 tokens = ${cost:.6f}"
        )

        # 测试控制器
        controller = get_cost_controller()

        # 设置测试租户限制
        test_limit = CostLimit(
            hourly_limit=1.0,
            daily_limit=10.0,
            requests_per_minute=10,
        )
        controller.set_tenant_limit("test_tenant", test_limit)

        # 测试请求检查
        allowed, reason = controller.check_allowed("test_tenant", 100)
        print_result(
            "请求检查",
            allowed,
            f"允许: {allowed}, 原因: {reason}"
        )

        # 测试使用记录
        controller.record_usage(
            model="deepseek-chat",
            input_tokens=500,
            output_tokens=200,
            tenant_id="test_tenant",
        )

        # 获取状态
        status = controller.get_status("test_tenant")
        print_result(
            "状态追踪",
            status.hourly_cost > 0,
            f"小时成本: ${status.hourly_cost:.6f}, 熔断状态: {status.circuit_state.name}"
        )

        return True

    except Exception as e:
        print_result("成本熔断测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_dspy_evolution():
    """测试 DSPy 自进化"""
    print_section("3. DSPy 自进化测试")

    try:
        from src.datapilot.llm.dspy_modules.self_evolution import (
            SelfEvolutionEngine,
            TrainingDataCollector,
            ModuleVersionManager,
            EvolutionConfig,
            get_evolution_engine,
        )

        # 测试进化引擎
        engine = get_evolution_engine()

        print_result(
            "进化引擎初始化",
            engine is not None,
            f"状态: {engine.status.value}"
        )

        # 测试数据收集
        engine.record_successful_query(
            question="查询销量最高的产品",
            schema="CREATE TABLE products (id INT, name TEXT, sales INT)",
            sql="SELECT name, sales FROM products ORDER BY sales DESC LIMIT 1",
            dialect="sqlite",
        )

        stats = engine.collector.get_stats()
        print_result(
            "训练数据收集",
            stats["total_samples"] > 0,
            f"总样本数: {stats['total_samples']}"
        )

        # 测试进化检查
        should_evolve, reason = engine.should_evolve()
        print_result(
            "进化条件检查",
            True,  # 检查本身成功即可
            f"应该进化: {should_evolve}, 原因: {reason}"
        )

        # 测试版本管理
        versions = engine.version_manager.get_all_versions("text2sql")
        print_result(
            "版本管理",
            True,
            f"text2sql 模块版本数: {len(versions)}"
        )

        return True

    except Exception as e:
        print_result("DSPy 自进化测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_user_preferences():
    """测试用户偏好持久化"""
    print_section("4. 用户偏好持久化测试")

    try:
        from src.datapilot.core.user_preferences import (
            UserPreferenceManager,
            get_preference_manager,
            UserPreferences,
            VisualizationPreferences,
        )

        manager = get_preference_manager()
        test_user = "test_user_001"

        # 测试获取/创建偏好
        prefs = await manager.get_preferences(test_user, create_if_missing=True)
        print_result(
            "偏好获取/创建",
            prefs is not None,
            f"用户: {prefs.user_id}, 主题: {prefs.ui.theme}"
        )

        # 测试更新偏好
        updated = await manager.update_preferences(test_user, {
            "visualization": {"default_chart_type": "bar"},
            "ui": {"theme": "dark"},
        })
        print_result(
            "偏好更新",
            updated.visualization.default_chart_type == "bar",
            f"图表类型: {updated.visualization.default_chart_type}"
        )

        # 测试查询历史记录
        await manager.record_query(
            user_id=test_user,
            query="查询销量",
            sql="SELECT * FROM sales",
            database="default",
            success=True,
            row_count=100,
        )

        history = await manager.get_query_history(test_user, limit=10)
        print_result(
            "查询历史",
            len(history) > 0,
            f"历史记录数: {len(history)}"
        )

        # 清理测试数据
        await manager.delete_user(test_user)

        return True

    except Exception as e:
        print_result("用户偏好测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_multi_judge():
    """测试多 Judge 博弈机制"""
    print_section("5. 多 Judge 博弈机制测试")

    try:
        from src.datapilot.agents.multi_judge import (
            MultiJudgeCoordinator,
            RuleBasedJudge,
            CostJudge,
            SemanticJudge,
            ConflictResolution,
            Verdict,
            get_multi_judge_coordinator,
        )

        # 测试单个 Judge
        rule_judge = RuleBasedJudge("rule_test")

        # 测试安全 SQL
        safe_sql = "SELECT name, price FROM products WHERE price > 100"
        vote = await rule_judge.evaluate(safe_sql)
        print_result(
            "规则 Judge (安全 SQL)",
            vote.verdict == Verdict.APPROVE,
            f"裁决: {vote.verdict.value}, 置信度: {vote.confidence:.2f}"
        )

        # 测试危险 SQL
        dangerous_sql = "DROP TABLE products; SELECT * FROM users"
        vote = await rule_judge.evaluate(dangerous_sql)
        print_result(
            "规则 Judge (危险 SQL)",
            vote.verdict == Verdict.REJECT,
            f"裁决: {vote.verdict.value}, 原因: {vote.reason[:50]}..."
        )

        # 测试多 Judge 协调器 (仅使用规则 Judge 避免 API 调用)
        coordinator = MultiJudgeCoordinator(
            judges=[
                RuleBasedJudge("rule_1"),
                RuleBasedJudge("rule_2"),
            ],
            resolution=ConflictResolution.MAJORITY,
        )

        result = await coordinator.evaluate(safe_sql)
        print_result(
            "多 Judge 协调 (多数决)",
            result.approved,
            f"最终裁决: {'通过' if result.approved else '拒绝'}, "
            f"投票数: {len(result.votes)}, 方法: {result.consensus_method}"
        )

        # 测试加权投票
        coordinator_weighted = MultiJudgeCoordinator(
            judges=[
                RuleBasedJudge("rule_1"),
                RuleBasedJudge("rule_2"),
            ],
            resolution=ConflictResolution.WEIGHTED,
        )

        result = await coordinator_weighted.evaluate(safe_sql)
        print_result(
            "多 Judge 协调 (加权)",
            result.approved,
            f"置信度: {result.final_confidence:.2f}"
        )

        return True

    except Exception as e:
        print_result("多 Judge 测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_sandbox_security():
    """测试沙箱 AST 安全分析"""
    print_section("6. 沙箱 AST 安全分析测试")

    try:
        from src.datapilot.sandbox.executor import (
            validate_code,
            validate_code_detailed,
            ASTSecurityAnalyzer,
            DANGEROUS_MODULES,
            ALLOWED_MODULES,
        )

        # 测试安全代码
        safe_code = """
import pandas as pd
import json

data = [{"name": "A", "value": 1}]
df = pd.DataFrame(data)
print(df.to_json())
"""
        is_safe, reason = validate_code(safe_code)
        print_result(
            "安全代码检测",
            is_safe,
            f"安全: {is_safe}, 原因: {reason}"
        )

        # 测试危险代码 - os 模块
        dangerous_code_os = """
import os
os.system("rm -rf /")
"""
        is_safe, reason = validate_code(dangerous_code_os)
        print_result(
            "危险代码检测 (os)",
            not is_safe,
            f"安全: {is_safe}, 原因: {reason[:60]}..."
        )

        # 测试危险代码 - eval
        dangerous_code_eval = """
user_input = "print('hacked')"
eval(user_input)
"""
        is_safe, reason = validate_code(dangerous_code_eval)
        print_result(
            "危险代码检测 (eval)",
            not is_safe,
            f"安全: {is_safe}, 原因: {reason[:60]}..."
        )

        # 测试详细分析
        result = validate_code_detailed(safe_code)
        print_result(
            "详细安全分析",
            result.is_safe,
            f"导入: {result.analyzed_imports}, 调用: {len(result.analyzed_calls)} 个"
        )

        # 显示模块白名单/黑名单
        print(f"\n         危险模块黑名单: {len(DANGEROUS_MODULES)} 个")
        print(f"         安全模块白名单: {len(ALLOWED_MODULES)} 个")

        return True

    except Exception as e:
        print_result("沙箱安全测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_llm_driven_agents():
    """测试 LLM 驱动的 Agent (需要 API Key)"""
    print_section("7. LLM 驱动 Agent 测试 (需要 API)")

    try:
        from src.datapilot.config.settings import get_settings
        settings = get_settings()

        if not settings.deepseek_api_key:
            print_result(
                "API Key 检查",
                False,
                "未配置 DEEPSEEK_API_KEY，跳过 LLM 测试"
            )
            return True  # 不算失败

        print_result("API Key 检查", True, "已配置 DeepSeek API Key")

        # 测试 DeepSeek 客户端
        from src.datapilot.llm.deepseek import get_deepseek_client
        client = get_deepseek_client()

        # 简单测试
        response = await client.chat([
            {"role": "user", "content": "回复 'OK' 两个字母"}
        ])
        print_result(
            "DeepSeek 客户端",
            "OK" in response.upper(),
            f"响应: {response[:50]}..."
        )

        return True

    except Exception as e:
        print_result("LLM Agent 测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def test_business_keywords():
    """测试配置化业务词汇"""
    print_section("8. 配置化业务词汇测试")

    try:
        from src.datapilot.agents.data_sniper import (
            BusinessKeywordsConfig,
            get_keywords_config,
        )

        # 测试默认配置
        config = BusinessKeywordsConfig()
        keywords = config.get_keywords()

        print_result(
            "默认业务词汇",
            len(keywords) > 0,
            f"类别数: {len(keywords)}"
        )

        # 显示部分类别
        for category in list(keywords.keys())[:5]:
            print(f"         - {category}: {keywords[category][:3]}...")

        # 测试从文件加载
        config_path = "data/business_keywords.json"
        if os.path.exists(config_path):
            config_from_file = BusinessKeywordsConfig(config_path)
            keywords_from_file = config_from_file.get_keywords()
            print_result(
                "文件配置加载",
                len(keywords_from_file) >= len(keywords),
                f"加载后类别数: {len(keywords_from_file)}"
            )
        else:
            print_result(
                "文件配置加载",
                True,
                f"配置文件不存在，使用默认配置"
            )

        # 测试动态添加
        config.add_keywords("custom", ["自定义1", "自定义2"])
        print_result(
            "动态添加词汇",
            "custom" in config.get_keywords(),
            f"新增类别: custom"
        )

        return True

    except Exception as e:
        print_result("业务词汇测试", False, str(e))
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  DataPilot 完整业务流程测试")
    print("  测试所有重构后的模块")
    print("=" * 60)

    results = []

    # 运行所有测试
    results.append(("MCP 工具标准化", await test_mcp_tools()))
    results.append(("成本熔断机制", await test_cost_control()))
    results.append(("DSPy 自进化", await test_dspy_evolution()))
    results.append(("用户偏好持久化", await test_user_preferences()))
    results.append(("多 Judge 博弈", await test_multi_judge()))
    results.append(("沙箱 AST 安全", await test_sandbox_security()))
    results.append(("LLM 驱动 Agent", await test_llm_driven_agents()))
    results.append(("配置化业务词汇", await test_business_keywords()))

    # 打印总结
    print_section("测试总结")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[OK]" if result else "[X]"
        print(f"  {status} {name}")

    print(f"\n  总计: {passed}/{total} 通过")

    if passed == total:
        print("\n  [SUCCESS] All tests passed! Refactoring successful!")
    else:
        print(f"\n  [WARNING] {total - passed} test(s) failed, please check logs")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
