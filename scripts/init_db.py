#!/usr/bin/env python3
"""
DataPilot 数据库初始化脚本
初始化 MySQL 和 PostgreSQL 数据库，导入模拟数据
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import aiomysql
import asyncpg
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(PROJECT_ROOT / ".env")

# 数据目录
SCHEMA_DIR = PROJECT_ROOT / "mock_data" / "schemas"
SEEDS_DIR = PROJECT_ROOT / "mock_data" / "seeds"


def load_json(filename: str) -> list[dict]:
    """加载 JSON 数据文件"""
    filepath = SEEDS_DIR / filename
    if not filepath.exists():
        print(f"⚠ 文件不存在: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sql(filename: str) -> str:
    """加载 SQL 文件"""
    filepath = SCHEMA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"SQL 文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ============================================
# MySQL 初始化
# ============================================

async def init_mysql():
    """初始化 MySQL 数据库"""
    print("\n" + "=" * 50)
    print("初始化 MySQL 数据库 (电商)")
    print("=" * 50)

    # 连接配置
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "datapilot"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "db": os.getenv("MYSQL_DATABASE", "ecommerce"),
        "charset": "utf8mb4",
    }

    try:
        # 创建连接
        conn = await aiomysql.connect(**config)
        cursor = await conn.cursor()
        print(f"✓ 已连接到 MySQL: {config['host']}:{config['port']}/{config['db']}")

        # 执行 Schema SQL
        print("\n[1/2] 创建表结构...")
        schema_sql = load_sql("ecommerce.sql")

        # 分割 SQL 语句并逐个执行
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for stmt in statements:
            if stmt and not stmt.startswith("--"):
                try:
                    await cursor.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"  ⚠ SQL 执行警告: {str(e)[:100]}")

        await conn.commit()
        print("  ✓ 表结构创建完成")

        # 导入数据
        print("\n[2/2] 导入模拟数据...")

        # 导入分类
        categories = load_json("categories.json")
        if categories:
            await cursor.execute("DELETE FROM categories")
            for cat in categories:
                await cursor.execute(
                    """INSERT INTO categories (category_id, parent_id, category_name, category_level)
                       VALUES (%s, %s, %s, %s)""",
                    (cat["category_id"], cat.get("parent_id"), cat["category_name"], cat["category_level"])
                )
            print(f"  ✓ 分类: {len(categories)} 条")

        # 导入品牌
        brands = load_json("brands.json")
        if brands:
            await cursor.execute("DELETE FROM brands")
            for brand in brands:
                await cursor.execute(
                    """INSERT INTO brands (brand_id, brand_name, brand_name_en, country)
                       VALUES (%s, %s, %s, %s)""",
                    (brand["brand_id"], brand["brand_name"], brand.get("brand_name_en"), brand.get("country"))
                )
            print(f"  ✓ 品牌: {len(brands)} 条")

        # 导入商品
        products = load_json("products.json")
        if products:
            await cursor.execute("DELETE FROM products")
            for prod in products:
                await cursor.execute(
                    """INSERT INTO products (product_id, product_code, product_name, category_id, brand_id,
                       price, original_price, cost_price, stock, sold_count, status, is_hot, is_new, rating, review_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (prod["product_id"], prod["product_code"], prod["product_name"], prod["category_id"],
                     prod.get("brand_id"), prod["price"], prod.get("original_price"), prod.get("cost_price"),
                     prod.get("stock", 0), prod.get("sold_count", 0), prod.get("status", "on_sale"),
                     prod.get("is_hot", False), prod.get("is_new", False), prod.get("rating", 5.0),
                     prod.get("review_count", 0))
                )
            print(f"  ✓ 商品: {len(products)} 条")

        # 导入用户
        users = load_json("customers.json")
        if users:
            await cursor.execute("DELETE FROM users")
            for user in users:
                await cursor.execute(
                    """INSERT INTO users (user_id, username, email, phone, password_hash, nickname,
                       gender, birthday, status, vip_level, points, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (user["user_id"], user["username"], user["email"], user.get("phone"),
                     "hashed_password", user.get("nickname"), user.get("gender", "unknown"),
                     user.get("birthday"), user.get("status", "active"), user.get("vip_level", 0),
                     user.get("points", 0), user.get("created_at"))
                )
            print(f"  ✓ 用户: {len(users)} 条")

        # 导入订单
        orders = load_json("orders.json")
        if orders:
            await cursor.execute("DELETE FROM order_items")
            await cursor.execute("DELETE FROM orders")
            for order in orders:
                await cursor.execute(
                    """INSERT INTO orders (order_id, order_no, user_id, total_amount, discount_amount,
                       shipping_fee, pay_amount, status, payment_method, payment_time,
                       receiver_name, receiver_phone, receiver_address, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (order["order_id"], order["order_no"], order["user_id"], order["total_amount"],
                     order.get("discount_amount", 0), order.get("shipping_fee", 0), order["pay_amount"],
                     order["status"], order.get("payment_method"), order.get("payment_time"),
                     order["receiver_name"], order["receiver_phone"], order["receiver_address"],
                     order.get("created_at"))
                )
            print(f"  ✓ 订单: {len(orders)} 条")

        # 导入订单明细
        order_items = load_json("order_items.json")
        if order_items:
            for item in order_items:
                await cursor.execute(
                    """INSERT INTO order_items (item_id, order_id, product_id, product_name,
                       price, quantity, total_price)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (item["item_id"], item["order_id"], item["product_id"], item["product_name"],
                     item["price"], item["quantity"], item["total_price"])
                )
            print(f"  ✓ 订单明细: {len(order_items)} 条")

        await conn.commit()
        print("\n✓ MySQL 数据库初始化完成!")

        # 关闭连接
        await cursor.close()
        conn.close()

    except Exception as e:
        print(f"\n✗ MySQL 初始化失败: {e}")
        raise


# ============================================
# PostgreSQL 初始化
# ============================================

async def init_postgres():
    """初始化 PostgreSQL 数据库"""
    print("\n" + "=" * 50)
    print("初始化 PostgreSQL 数据库 (销售)")
    print("=" * 50)

    # 连接配置
    config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "user": os.getenv("POSTGRES_USER", "datapilot"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "database": os.getenv("POSTGRES_DATABASE", "sales"),
    }

    try:
        # 创建连接
        conn = await asyncpg.connect(**config)
        print(f"✓ 已连接到 PostgreSQL: {config['host']}:{config['port']}/{config['database']}")

        # 执行 Schema SQL
        print("\n[1/2] 创建表结构...")
        schema_sql = load_sql("sales.sql")

        # 分割并执行 SQL
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for stmt in statements:
            if stmt and not stmt.startswith("--"):
                try:
                    await conn.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"  ⚠ SQL 执行警告: {str(e)[:100]}")

        print("  ✓ 表结构创建完成")

        # 导入数据
        print("\n[2/2] 导入模拟数据...")

        # 导入部门
        departments = load_json("departments.json")
        if departments:
            await conn.execute("DELETE FROM departments")
            for dept in departments:
                await conn.execute(
                    """INSERT INTO departments (department_id, department_name, parent_id, level)
                       VALUES ($1, $2, $3, $4)""",
                    dept["department_id"], dept["department_name"], dept.get("parent_id"), dept.get("level", 1)
                )
            print(f"  ✓ 部门: {len(departments)} 条")

        # 导入员工
        employees = load_json("employees.json")
        if employees:
            await conn.execute("DELETE FROM salespeople")
            await conn.execute("DELETE FROM employees")
            for emp in employees:
                await conn.execute(
                    """INSERT INTO employees (employee_id, employee_no, name, email, phone,
                       department_id, position, hire_date, base_salary, status)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                    emp["employee_id"], emp["employee_no"], emp["name"], emp.get("email"),
                    emp.get("phone"), emp.get("department_id"), emp.get("position"),
                    emp["hire_date"], emp.get("base_salary"), emp.get("status", "active")
                )
            print(f"  ✓ 员工: {len(employees)} 条")

        # 导入销售团队
        teams = load_json("sales_teams.json")
        if teams:
            await conn.execute("DELETE FROM sales_teams")
            for team in teams:
                await conn.execute(
                    """INSERT INTO sales_teams (team_id, team_name, team_leader_id, department_id,
                       target_amount, region)
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    team["team_id"], team["team_name"], team.get("team_leader_id"),
                    team.get("department_id"), team.get("target_amount", 0), team.get("region")
                )
            print(f"  ✓ 销售团队: {len(teams)} 条")

        # 导入销售人员
        salespeople = load_json("salespeople.json")
        if salespeople:
            for sp in salespeople:
                await conn.execute(
                    """INSERT INTO salespeople (salesperson_id, employee_id, team_id, sales_level,
                       commission_rate, monthly_target, quarterly_target, annual_target)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    sp["salesperson_id"], sp["employee_id"], sp.get("team_id"), sp.get("sales_level", "junior"),
                    sp.get("commission_rate", 0.05), sp.get("monthly_target", 0),
                    sp.get("quarterly_target", 0), sp.get("annual_target", 0)
                )
            print(f"  ✓ 销售人员: {len(salespeople)} 条")

        # 导入客户
        customers = load_json("b2b_customers.json")
        if customers:
            await conn.execute("DELETE FROM customers")
            for cust in customers:
                await conn.execute(
                    """INSERT INTO customers (customer_id, customer_code, company_name, short_name,
                       industry, company_size, contact_name, contact_phone, contact_email,
                       province, city, source, salesperson_id, customer_level, status, first_contact_date)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)""",
                    cust["customer_id"], cust["customer_code"], cust["company_name"], cust.get("short_name"),
                    cust.get("industry"), cust.get("company_size"), cust.get("contact_name"),
                    cust.get("contact_phone"), cust.get("contact_email"), cust.get("province"),
                    cust.get("city"), cust.get("source"), cust.get("salesperson_id"),
                    cust.get("customer_level", "C"), cust.get("status", "potential"),
                    cust.get("first_contact_date")
                )
            print(f"  ✓ 客户: {len(customers)} 条")

        # 导入合同
        contracts = load_json("contracts.json")
        if contracts:
            await conn.execute("DELETE FROM contracts")
            for cont in contracts:
                await conn.execute(
                    """INSERT INTO contracts (contract_id, contract_no, contract_name, customer_id,
                       salesperson_id, contract_type, total_amount, paid_amount, start_date,
                       end_date, sign_date, status)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
                    cont["contract_id"], cont["contract_no"], cont["contract_name"], cont["customer_id"],
                    cont["salesperson_id"], cont.get("contract_type", "new"), cont["total_amount"],
                    cont.get("paid_amount", 0), cont["start_date"], cont.get("end_date"),
                    cont.get("sign_date"), cont.get("status", "active")
                )
            print(f"  ✓ 合同: {len(contracts)} 条")

        # 导入销售业绩
        performance = load_json("sales_performance.json")
        if performance:
            await conn.execute("DELETE FROM sales_performance")
            for perf in performance:
                await conn.execute(
                    """INSERT INTO sales_performance (performance_id, salesperson_id, year, month,
                       target_amount, achieved_amount, contract_count, achievement_rate)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    perf["performance_id"], perf["salesperson_id"], perf["year"], perf["month"],
                    perf.get("target_amount", 0), perf.get("achieved_amount", 0),
                    perf.get("contract_count", 0), perf.get("achievement_rate", 0)
                )
            print(f"  ✓ 销售业绩: {len(performance)} 条")

        print("\n✓ PostgreSQL 数据库初始化完成!")

        # 关闭连接
        await conn.close()

    except Exception as e:
        print(f"\n✗ PostgreSQL 初始化失败: {e}")
        raise


# ============================================
# 主函数
# ============================================

async def main():
    print("=" * 50)
    print("DataPilot 数据库初始化")
    print("=" * 50)

    # 检查数据文件是否存在
    if not SEEDS_DIR.exists():
        print(f"\n⚠ 数据目录不存在: {SEEDS_DIR}")
        print("请先运行 generate_mock_data.py 生成模拟数据")
        return

    # 初始化数据库
    try:
        await init_mysql()
    except Exception as e:
        print(f"MySQL 初始化跳过: {e}")

    try:
        await init_postgres()
    except Exception as e:
        print(f"PostgreSQL 初始化跳过: {e}")

    print("\n" + "=" * 50)
    print("数据库初始化完成!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
