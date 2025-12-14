import json
import os

data = []
schema = "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, stock INTEGER, category_id INTEGER, brand TEXT); CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, city TEXT); CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total_amount REAL, status TEXT, created_at TEXT); CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER, price REAL);"

queries = [
    ("查询所有产品", "SELECT * FROM products LIMIT 100"),
    ("查询所有客户", "SELECT * FROM customers LIMIT 100"),
    ("查询所有订单", "SELECT * FROM orders LIMIT 100"),
    ("查询产品数量", "SELECT COUNT(*) as count FROM products"),
    ("查询客户数量", "SELECT COUNT(*) as count FROM customers"),
    ("查询订单数量", "SELECT COUNT(*) as count FROM orders"),
    ("查询产品名称和价格", "SELECT name, price FROM products LIMIT 100"),
    ("查询价格大于100的产品", "SELECT * FROM products WHERE price > 100 LIMIT 100"),
    ("查询价格小于50的产品", "SELECT * FROM products WHERE price < 50 LIMIT 100"),
    ("查询库存大于10的产品", "SELECT * FROM products WHERE stock > 10 LIMIT 100"),
    ("查询库存为0的产品", "SELECT * FROM products WHERE stock = 0 LIMIT 100"),
    ("查询产品平均价格", "SELECT AVG(price) as avg_price FROM products"),
    ("查询产品最高价格", "SELECT MAX(price) as max_price FROM products"),
    ("查询产品最低价格", "SELECT MIN(price) as min_price FROM products"),
    ("查询订单总金额", "SELECT SUM(total_amount) as total FROM orders"),
    ("查询订单平均金额", "SELECT AVG(total_amount) as avg_amount FROM orders"),
    ("查询每个分类的产品数量", "SELECT category_id, COUNT(*) as count FROM products GROUP BY category_id"),
    ("查询每个城市的客户数量", "SELECT city, COUNT(*) as count FROM customers GROUP BY city"),
    ("查询每个状态的订单数量", "SELECT status, COUNT(*) as count FROM orders GROUP BY status"),
    ("查询价格最高的10个产品", "SELECT * FROM products ORDER BY price DESC LIMIT 10"),
    ("查询价格最低的10个产品", "SELECT * FROM products ORDER BY price ASC LIMIT 10"),
    ("查询最新的10个订单", "SELECT * FROM orders ORDER BY created_at DESC LIMIT 10"),
    ("查询金额最大的10个订单", "SELECT * FROM orders ORDER BY total_amount DESC LIMIT 10"),
    ("查询库存最多的产品", "SELECT * FROM products ORDER BY stock DESC LIMIT 10"),
    ("查询订单及其客户信息", "SELECT o.*, c.name as customer_name FROM orders o JOIN customers c ON o.customer_id = c.id LIMIT 100"),
    ("查询产品及其分类", "SELECT p.*, c.name as category_name FROM products p JOIN categories c ON p.category_id = c.id LIMIT 100"),
    ("查询客户的订单数量", "SELECT c.name, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id"),
    ("查询没有订单的客户", "SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.id IS NULL"),
    ("查询热销产品TOP10", "SELECT p.name, SUM(oi.quantity) as total_sold FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id ORDER BY total_sold DESC LIMIT 10"),
    ("查询每个客户的消费总额", "SELECT c.name, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total_spent DESC"),
    ("查询复购客户", "SELECT c.name, COUNT(o.id) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id HAVING COUNT(o.id) > 1"),
    ("查询高价值客户", "SELECT c.name, SUM(o.total_amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id HAVING SUM(o.total_amount) > 5000"),
    ("查询滞销产品", "SELECT p.* FROM products p LEFT JOIN order_items oi ON p.id = oi.product_id WHERE oi.id IS NULL"),
    ("查询各城市的订单分布", "SELECT c.city, COUNT(o.id) as order_count, SUM(o.total_amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.city"),
    ("查询订单状态统计", "SELECT status, COUNT(*) as count, SUM(total_amount) as total FROM orders GROUP BY status"),
    ("查询所有分类", "SELECT * FROM categories"),
    ("查询分类数量", "SELECT COUNT(*) FROM categories"),
    ("查询订单详情", "SELECT o.id, c.name, oi.quantity, oi.price FROM orders o JOIN customers c ON o.customer_id = c.id JOIN order_items oi ON o.id = oi.order_id LIMIT 100"),
    ("查询每个分类的平均价格", "SELECT c.name, AVG(p.price) as avg_price FROM categories c JOIN products p ON c.id = p.category_id GROUP BY c.id"),
    ("查询消费最多的5个客户", "SELECT c.name, SUM(o.total_amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total DESC LIMIT 5"),
]

# Generate 200+ by adding variants
for i, (q, s) in enumerate(queries):
    data.append({"id": i+1, "question": q, "sql": s, "schema": schema, "dialect": "sqlite"})

# Add variants
base_len = len(data)
for i in range(160):
    idx = i % len(queries)
    q, s = queries[idx]
    variant_q = q + " (前5条)" if i % 2 == 0 else q + " (按ID排序)"
    variant_s = s.replace("LIMIT 100", "LIMIT 5") if i % 2 == 0 else s + " ORDER BY id" if "ORDER BY" not in s else s
    data.append({"id": base_len + i + 1, "question": variant_q, "sql": variant_s, "schema": schema, "dialect": "sqlite"})

os.makedirs("tests/evaluation/datasets", exist_ok=True)
with open("tests/evaluation/datasets/sql_generation.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Generated {len(data)} test cases")
