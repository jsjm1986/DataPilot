"""
SQLite 数据库初始化脚本
创建表结构并生成模拟数据
"""

import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 修复 Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 确保 data 目录存在
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "datapilot.db"


def create_tables(conn: sqlite3.Connection):
    """创建表结构"""
    cursor = conn.cursor()

    # 电商数据库表
    cursor.executescript("""
    -- 商品分类表
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        parent_id INTEGER,
        level INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 商品表
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category_id INTEGER,
        brand TEXT,
        price DECIMAL(10, 2) NOT NULL,
        cost DECIMAL(10, 2),
        stock INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (category_id) REFERENCES categories(id)
    );

    -- 客户表
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        city TEXT,
        province TEXT,
        member_level TEXT DEFAULT 'normal',
        total_spent DECIMAL(12, 2) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 订单表
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_no TEXT UNIQUE NOT NULL,
        customer_id INTEGER,
        total_amount DECIMAL(12, 2) NOT NULL,
        discount_amount DECIMAL(10, 2) DEFAULT 0,
        payment_amount DECIMAL(12, 2) NOT NULL,
        status TEXT DEFAULT 'pending',
        payment_method TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        paid_at TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );

    -- 订单明细表
    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL,
        subtotal DECIMAL(12, 2) NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );

    -- 销售记录表 (按日汇总)
    CREATE TABLE IF NOT EXISTS sales_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        product_id INTEGER,
        category_id INTEGER,
        quantity_sold INTEGER DEFAULT 0,
        revenue DECIMAL(12, 2) DEFAULT 0,
        cost DECIMAL(12, 2) DEFAULT 0,
        profit DECIMAL(12, 2) DEFAULT 0,
        FOREIGN KEY (product_id) REFERENCES products(id),
        FOREIGN KEY (category_id) REFERENCES categories(id)
    );

    -- 库存变动表
    CREATE TABLE IF NOT EXISTS inventory_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        change_type TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        before_stock INTEGER,
        after_stock INTEGER,
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (product_id) REFERENCES products(id)
    );

    -- 用户行为表
    CREATE TABLE IF NOT EXISTS user_behaviors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        product_id INTEGER,
        behavior_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    conn.commit()
    print("✓ 表结构创建完成")


def generate_mock_data(conn: sqlite3.Connection):
    """生成模拟数据"""
    cursor = conn.cursor()

    # 1. 生成分类数据
    categories = [
        ("手机数码", None, 1),
        ("电脑办公", None, 1),
        ("家用电器", None, 1),
        ("服装鞋包", None, 1),
        ("食品生鲜", None, 1),
        ("智能手机", 1, 2),
        ("平板电脑", 1, 2),
        ("笔记本电脑", 2, 2),
        ("台式机", 2, 2),
        ("冰箱", 3, 2),
        ("洗衣机", 3, 2),
        ("空调", 3, 2),
        ("男装", 4, 2),
        ("女装", 4, 2),
        ("零食", 5, 2),
        ("水果", 5, 2),
    ]
    cursor.executemany(
        "INSERT INTO categories (name, parent_id, level) VALUES (?, ?, ?)",
        categories
    )
    print(f"✓ 生成 {len(categories)} 个分类")

    # 2. 生成商品数据
    brands = {
        6: ["Apple", "华为", "小米", "OPPO", "vivo", "三星"],
        7: ["Apple", "华为", "小米", "联想"],
        8: ["联想", "戴尔", "华硕", "惠普", "苹果"],
        9: ["联想", "戴尔", "华硕", "惠普"],
        10: ["海尔", "美的", "西门子", "容声"],
        11: ["海尔", "美的", "小天鹅", "西门子"],
        12: ["格力", "美的", "海尔", "奥克斯"],
        13: ["优衣库", "海澜之家", "森马", "太平鸟"],
        14: ["优衣库", "ZARA", "H&M", "UR"],
        15: ["三只松鼠", "良品铺子", "百草味"],
        16: ["佳沛", "都乐", "佳农"],
    }

    products = []
    product_id = 1
    for cat_id in range(6, 17):
        cat_brands = brands.get(cat_id, ["通用品牌"])
        for i in range(random.randint(15, 30)):
            brand = random.choice(cat_brands)
            price = round(random.uniform(50, 10000), 2)
            cost = round(price * random.uniform(0.5, 0.8), 2)
            stock = random.randint(0, 500)
            products.append((
                f"{brand} 商品{product_id}",
                cat_id,
                brand,
                price,
                cost,
                stock,
                random.choice(["active", "active", "active", "inactive"])
            ))
            product_id += 1

    cursor.executemany(
        "INSERT INTO products (name, category_id, brand, price, cost, stock, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        products
    )
    print(f"✓ 生成 {len(products)} 个商品")

    # 3. 生成客户数据
    cities = [
        ("北京", "北京"), ("上海", "上海"), ("广州", "广东"), ("深圳", "广东"),
        ("杭州", "浙江"), ("南京", "江苏"), ("成都", "四川"), ("武汉", "湖北"),
        ("西安", "陕西"), ("重庆", "重庆"), ("苏州", "江苏"), ("天津", "天津"),
    ]
    member_levels = ["normal", "normal", "normal", "silver", "silver", "gold", "platinum"]

    customers = []
    for i in range(1, 501):
        city, province = random.choice(cities)
        customers.append((
            f"客户{i}",
            f"customer{i}@example.com",
            f"138{random.randint(10000000, 99999999)}",
            city,
            province,
            random.choice(member_levels),
            0
        ))

    cursor.executemany(
        "INSERT INTO customers (name, email, phone, city, province, member_level, total_spent) VALUES (?, ?, ?, ?, ?, ?, ?)",
        customers
    )
    print(f"✓ 生成 {len(customers)} 个客户")

    # 4. 生成订单数据
    payment_methods = ["alipay", "wechat", "credit_card", "debit_card"]
    statuses = ["pending", "paid", "shipped", "completed", "completed", "completed", "cancelled"]

    # 获取商品列表
    cursor.execute("SELECT id, price FROM products WHERE status = 'active'")
    active_products = cursor.fetchall()

    orders = []
    order_items = []
    order_id = 1

    # 生成过去 180 天的订单
    base_date = datetime.now()
    for day_offset in range(180, 0, -1):
        order_date = base_date - timedelta(days=day_offset)
        # 每天 10-50 个订单
        daily_orders = random.randint(10, 50)

        for _ in range(daily_orders):
            customer_id = random.randint(1, 500)
            status = random.choice(statuses)
            payment_method = random.choice(payment_methods)

            # 订单包含 1-5 个商品
            num_items = random.randint(1, 5)
            selected_products = random.sample(active_products, min(num_items, len(active_products)))

            total_amount = 0
            for prod_id, prod_price in selected_products:
                quantity = random.randint(1, 3)
                subtotal = round(prod_price * quantity, 2)
                total_amount += subtotal
                order_items.append((order_id, prod_id, quantity, prod_price, subtotal))

            discount = round(total_amount * random.uniform(0, 0.15), 2) if random.random() > 0.7 else 0
            payment_amount = round(total_amount - discount, 2)

            order_time = order_date + timedelta(
                hours=random.randint(8, 22),
                minutes=random.randint(0, 59)
            )
            paid_time = order_time + timedelta(minutes=random.randint(1, 30)) if status != "pending" else None

            orders.append((
                f"ORD{order_date.strftime('%Y%m%d')}{order_id:06d}",
                customer_id,
                total_amount,
                discount,
                payment_amount,
                status,
                payment_method,
                order_time.strftime("%Y-%m-%d %H:%M:%S"),
                paid_time.strftime("%Y-%m-%d %H:%M:%S") if paid_time else None
            ))
            order_id += 1

    cursor.executemany(
        "INSERT INTO orders (order_no, customer_id, total_amount, discount_amount, payment_amount, status, payment_method, created_at, paid_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        orders
    )
    cursor.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal) VALUES (?, ?, ?, ?, ?)",
        order_items
    )
    print(f"✓ 生成 {len(orders)} 个订单, {len(order_items)} 个订单明细")

    # 5. 更新客户消费总额
    cursor.execute("""
        UPDATE customers SET total_spent = (
            SELECT COALESCE(SUM(payment_amount), 0)
            FROM orders
            WHERE orders.customer_id = customers.id AND orders.status != 'cancelled'
        )
    """)

    # 6. 生成销售日报数据
    cursor.execute("""
        INSERT INTO sales_daily (date, product_id, category_id, quantity_sold, revenue, cost, profit)
        SELECT
            DATE(o.created_at) as date,
            oi.product_id,
            p.category_id,
            SUM(oi.quantity) as quantity_sold,
            SUM(oi.subtotal) as revenue,
            SUM(oi.quantity * p.cost) as cost,
            SUM(oi.subtotal - oi.quantity * p.cost) as profit
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        JOIN products p ON oi.product_id = p.id
        WHERE o.status != 'cancelled'
        GROUP BY DATE(o.created_at), oi.product_id, p.category_id
    """)
    print("✓ 生成销售日报数据")

    # 7. 生成用户行为数据
    behaviors = []
    behavior_types = ["view", "view", "view", "cart", "cart", "favorite", "purchase"]
    for _ in range(5000):
        behaviors.append((
            random.randint(1, 500),
            random.randint(1, len(products)),
            random.choice(behavior_types),
            (base_date - timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d %H:%M:%S")
        ))

    cursor.executemany(
        "INSERT INTO user_behaviors (customer_id, product_id, behavior_type, created_at) VALUES (?, ?, ?, ?)",
        behaviors
    )
    print(f"✓ 生成 {len(behaviors)} 条用户行为记录")

    conn.commit()


def show_statistics(conn: sqlite3.Connection):
    """显示数据统计"""
    cursor = conn.cursor()

    print("\n" + "=" * 50)
    print("数据库统计信息")
    print("=" * 50)

    tables = ["categories", "products", "customers", "orders", "order_items", "sales_daily", "user_behaviors"]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} 条记录")

    # 订单金额统计
    cursor.execute("SELECT SUM(payment_amount), AVG(payment_amount) FROM orders WHERE status != 'cancelled'")
    total, avg = cursor.fetchone()
    print(f"\n  订单总金额: ¥{total:,.2f}")
    print(f"  订单平均金额: ¥{avg:,.2f}")

    # 热销商品
    cursor.execute("""
        SELECT p.name, SUM(oi.quantity) as sold
        FROM order_items oi
        JOIN products p ON oi.product_id = p.id
        JOIN orders o ON oi.order_id = o.id
        WHERE o.status != 'cancelled'
        GROUP BY p.id
        ORDER BY sold DESC
        LIMIT 5
    """)
    print("\n  热销商品 TOP 5:")
    for name, sold in cursor.fetchall():
        print(f"    - {name}: {sold} 件")

    print("=" * 50)


def main():
    """主函数"""
    print(f"SQLite 数据库路径: {DB_PATH}")

    # 删除旧数据库
    if DB_PATH.exists():
        os.remove(DB_PATH)
        print("✓ 删除旧数据库")

    # 创建新数据库
    conn = sqlite3.connect(DB_PATH)
    print("✓ 创建新数据库")

    try:
        create_tables(conn)
        generate_mock_data(conn)
        show_statistics(conn)
        print("\n✅ 数据库初始化完成!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
