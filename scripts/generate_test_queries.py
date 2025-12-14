# -*- coding: utf-8 -*-
"""
生成 200+ 条评测集
"""

import json
import random
from pathlib import Path

# 模板定义
TEMPLATES = {
    # ========== 简单查询 (simple) ==========
    "simple": [
        {"query": "查询所有产品", "sql": "SELECT * FROM products"},
        {"query": "查询所有客户", "sql": "SELECT * FROM customers"},
        {"query": "查询所有订单", "sql": "SELECT * FROM orders"},
        {"query": "查询所有销售记录", "sql": "SELECT * FROM sales_records"},
        {"query": "显示产品列表", "sql": "SELECT * FROM products"},
        {"query": "列出全部客户信息", "sql": "SELECT * FROM customers"},
        {"query": "获取订单数据", "sql": "SELECT * FROM orders"},
        {"query": "展示销售数据", "sql": "SELECT * FROM sales_records"},
        {"query": "查看产品表", "sql": "SELECT * FROM products"},
        {"query": "查看客户表", "sql": "SELECT * FROM customers"},
    ],

    # ========== 过滤查询 (filter) ==========
    "filter": [
        {"query": "查询价格大于5000的产品", "sql": "SELECT * FROM products WHERE price > 5000"},
        {"query": "查询价格小于1000的产品", "sql": "SELECT * FROM products WHERE price < 1000"},
        {"query": "查询价格等于3999的产品", "sql": "SELECT * FROM products WHERE price = 3999"},
        {"query": "查询手机类别的产品", "sql": "SELECT * FROM products WHERE category = '手机'"},
        {"query": "查询电脑类别的产品", "sql": "SELECT * FROM products WHERE category = '电脑'"},
        {"query": "查询配件类别的产品", "sql": "SELECT * FROM products WHERE category = '配件'"},
        {"query": "查询在售的产品", "sql": "SELECT * FROM products WHERE status = 'on_sale'"},
        {"query": "查询下架的产品", "sql": "SELECT * FROM products WHERE status = 'off_sale'"},
        {"query": "查询VIP客户", "sql": "SELECT * FROM customers WHERE level = 'VIP'"},
        {"query": "查询普通客户", "sql": "SELECT * FROM customers WHERE level = '普通'"},
        {"query": "查询金牌客户", "sql": "SELECT * FROM customers WHERE level = '金牌'"},
        {"query": "查询北京的客户", "sql": "SELECT * FROM customers WHERE region = '北京'"},
        {"query": "查询上海的客户", "sql": "SELECT * FROM customers WHERE region = '上海'"},
        {"query": "查询广州的客户", "sql": "SELECT * FROM customers WHERE region = '广州'"},
        {"query": "查询深圳的客户", "sql": "SELECT * FROM customers WHERE region = '深圳'"},
        {"query": "查询已完成的订单", "sql": "SELECT * FROM orders WHERE status = 'completed'"},
        {"query": "查询待支付的订单", "sql": "SELECT * FROM orders WHERE status = 'pending'"},
        {"query": "查询已取消的订单", "sql": "SELECT * FROM orders WHERE status = 'cancelled'"},
        {"query": "查询已发货的订单", "sql": "SELECT * FROM orders WHERE status = 'shipped'"},
        {"query": "查询华北地区的销售", "sql": "SELECT * FROM sales_records WHERE region = '华北'"},
        {"query": "查询华东地区的销售", "sql": "SELECT * FROM sales_records WHERE region = '华东'"},
        {"query": "查询华南地区的销售", "sql": "SELECT * FROM sales_records WHERE region = '华南'"},
        {"query": "查询西部地区的销售", "sql": "SELECT * FROM sales_records WHERE region = '西部'"},
    ],

    # ========== 范围查询 (range) ==========
    "range": [
        {"query": "查询价格在3000到6000之间的产品", "sql": "SELECT * FROM products WHERE price BETWEEN 3000 AND 6000"},
        {"query": "查询价格在1000到3000之间的产品", "sql": "SELECT * FROM products WHERE price BETWEEN 1000 AND 3000"},
        {"query": "查询价格在5000到10000之间的产品", "sql": "SELECT * FROM products WHERE price BETWEEN 5000 AND 10000"},
        {"query": "查询库存在100到500之间的产品", "sql": "SELECT * FROM products WHERE stock BETWEEN 100 AND 500"},
        {"query": "查询库存少于100的产品", "sql": "SELECT * FROM products WHERE stock < 100"},
        {"query": "查询库存大于500的产品", "sql": "SELECT * FROM products WHERE stock > 500"},
        {"query": "查询订单金额大于1000的订单", "sql": "SELECT * FROM orders WHERE amount > 1000"},
        {"query": "查询订单金额小于500的订单", "sql": "SELECT * FROM orders WHERE amount < 500"},
        {"query": "查询订单金额在500到2000之间", "sql": "SELECT * FROM orders WHERE amount BETWEEN 500 AND 2000"},
        {"query": "查询数量大于5的订单", "sql": "SELECT * FROM orders WHERE quantity > 5"},
    ],

    # ========== 排序查询 (sort) ==========
    "sort": [
        {"query": "按价格从高到低排序产品", "sql": "SELECT * FROM products ORDER BY price DESC"},
        {"query": "按价格从低到高排序产品", "sql": "SELECT * FROM products ORDER BY price ASC"},
        {"query": "按库存从多到少排序产品", "sql": "SELECT * FROM products ORDER BY stock DESC"},
        {"query": "按库存从少到多排序产品", "sql": "SELECT * FROM products ORDER BY stock ASC"},
        {"query": "按名称排序产品", "sql": "SELECT * FROM products ORDER BY name"},
        {"query": "按订单金额降序排列", "sql": "SELECT * FROM orders ORDER BY amount DESC"},
        {"query": "按订单金额升序排列", "sql": "SELECT * FROM orders ORDER BY amount ASC"},
        {"query": "按创建时间排序订单", "sql": "SELECT * FROM orders ORDER BY created_at DESC"},
        {"query": "按销售额排序销售记录", "sql": "SELECT * FROM sales_records ORDER BY amount DESC"},
        {"query": "按日期排序销售记录", "sql": "SELECT * FROM sales_records ORDER BY date DESC"},
    ],

    # ========== TopK 查询 (topk) ==========
    "topk": [
        {"query": "查询库存最多的5个产品", "sql": "SELECT * FROM products ORDER BY stock DESC LIMIT 5"},
        {"query": "查询价格最高的10个产品", "sql": "SELECT * FROM products ORDER BY price DESC LIMIT 10"},
        {"query": "查询价格最低的5个产品", "sql": "SELECT * FROM products ORDER BY price ASC LIMIT 5"},
        {"query": "查询金额最高的10个订单", "sql": "SELECT * FROM orders ORDER BY amount DESC LIMIT 10"},
        {"query": "查询最近的20个订单", "sql": "SELECT * FROM orders ORDER BY created_at DESC LIMIT 20"},
        {"query": "查询销售额最高的5条记录", "sql": "SELECT * FROM sales_records ORDER BY amount DESC LIMIT 5"},
        {"query": "查询前3名客户", "sql": "SELECT * FROM customers LIMIT 3"},
        {"query": "查询库存最少的10个产品", "sql": "SELECT * FROM products ORDER BY stock ASC LIMIT 10"},
        {"query": "查询最新注册的10个客户", "sql": "SELECT * FROM customers ORDER BY created_at DESC LIMIT 10"},
        {"query": "查询数量最多的5个订单", "sql": "SELECT * FROM orders ORDER BY quantity DESC LIMIT 5"},
    ],

    # ========== 聚合查询 (aggregation) ==========
    "aggregation": [
        {"query": "统计产品总数", "sql": "SELECT COUNT(*) as count FROM products"},
        {"query": "统计客户总数", "sql": "SELECT COUNT(*) as count FROM customers"},
        {"query": "统计订单总数", "sql": "SELECT COUNT(*) as count FROM orders"},
        {"query": "计算产品平均价格", "sql": "SELECT AVG(price) as avg_price FROM products"},
        {"query": "计算产品最高价格", "sql": "SELECT MAX(price) as max_price FROM products"},
        {"query": "计算产品最低价格", "sql": "SELECT MIN(price) as min_price FROM products"},
        {"query": "计算产品价格总和", "sql": "SELECT SUM(price) as total_price FROM products"},
        {"query": "计算总销售额", "sql": "SELECT SUM(amount) as total FROM orders"},
        {"query": "计算平均订单金额", "sql": "SELECT AVG(amount) as avg_amount FROM orders"},
        {"query": "计算最大订单金额", "sql": "SELECT MAX(amount) as max_amount FROM orders"},
        {"query": "统计各类别产品数量", "sql": "SELECT category, COUNT(*) as count FROM products GROUP BY category"},
        {"query": "统计各地区客户数量", "sql": "SELECT region, COUNT(*) as count FROM customers GROUP BY region"},
        {"query": "统计各状态订单数量", "sql": "SELECT status, COUNT(*) as count FROM orders GROUP BY status"},
        {"query": "统计各会员等级客户数", "sql": "SELECT level, COUNT(*) as count FROM customers GROUP BY level"},
        {"query": "统计各地区销售总额", "sql": "SELECT region, SUM(amount) as total FROM sales_records GROUP BY region"},
        {"query": "统计各产品类别销售额", "sql": "SELECT product_category, SUM(amount) as total FROM sales_records GROUP BY product_category"},
        {"query": "统计各销售员业绩", "sql": "SELECT salesperson, SUM(amount) as total FROM sales_records GROUP BY salesperson"},
        {"query": "统计在售产品数量", "sql": "SELECT COUNT(*) as count FROM products WHERE status = 'on_sale'"},
        {"query": "统计VIP客户数量", "sql": "SELECT COUNT(*) as count FROM customers WHERE level = 'VIP'"},
        {"query": "统计已完成订单数量", "sql": "SELECT COUNT(*) as count FROM orders WHERE status = 'completed'"},
    ],

    # ========== 分组排序 (group_sort) ==========
    "group_sort": [
        {"query": "查询销售额最高的地区", "sql": "SELECT region, SUM(amount) as total FROM sales_records GROUP BY region ORDER BY total DESC LIMIT 1"},
        {"query": "查询销售额最高的销售员", "sql": "SELECT salesperson, SUM(amount) as total FROM sales_records GROUP BY salesperson ORDER BY total DESC LIMIT 1"},
        {"query": "查询订单最多的客户", "sql": "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 1"},
        {"query": "查询销量最高的产品类别", "sql": "SELECT product_category, SUM(amount) as total FROM sales_records GROUP BY product_category ORDER BY total DESC LIMIT 1"},
        {"query": "按销售额排序各地区", "sql": "SELECT region, SUM(amount) as total FROM sales_records GROUP BY region ORDER BY total DESC"},
        {"query": "按客户数排序各地区", "sql": "SELECT region, COUNT(*) as count FROM customers GROUP BY region ORDER BY count DESC"},
        {"query": "按产品数排序各类别", "sql": "SELECT category, COUNT(*) as count FROM products GROUP BY category ORDER BY count DESC"},
        {"query": "查询平均订单金额最高的客户", "sql": "SELECT customer_id, AVG(amount) as avg_amount FROM orders GROUP BY customer_id ORDER BY avg_amount DESC LIMIT 1"},
    ],

    # ========== 时间查询 (time_filter) ==========
    "time_filter": [
        {"query": "查询2024年的订单", "sql": "SELECT * FROM orders WHERE created_at LIKE '2024%'"},
        {"query": "查询2024年注册的客户", "sql": "SELECT * FROM customers WHERE created_at LIKE '2024%'"},
        {"query": "查询12月份的销售记录", "sql": "SELECT * FROM sales_records WHERE date LIKE '%-12-%'"},
        {"query": "查询1月份的订单", "sql": "SELECT * FROM orders WHERE created_at LIKE '%-01-%'"},
        {"query": "查询今年的销售数据", "sql": "SELECT * FROM sales_records WHERE date LIKE '2024%'"},
        {"query": "查询上个月的订单", "sql": "SELECT * FROM orders WHERE created_at LIKE '2024-11%'"},
        {"query": "查询本月的销售记录", "sql": "SELECT * FROM sales_records WHERE date LIKE '2024-12%'"},
        {"query": "查询Q4的销售数据", "sql": "SELECT * FROM sales_records WHERE date >= '2024-10-01' AND date <= '2024-12-31'"},
    ],

    # ========== 时间序列 (time_series) ==========
    "time_series": [
        {"query": "统计每日订单数量", "sql": "SELECT DATE(created_at) as date, COUNT(*) as count FROM orders GROUP BY DATE(created_at)"},
        {"query": "统计每日销售额", "sql": "SELECT date, SUM(amount) as total FROM sales_records GROUP BY date ORDER BY date"},
        {"query": "查询12月份的销售趋势", "sql": "SELECT date, SUM(amount) as daily_total FROM sales_records WHERE date LIKE '2024-12%' GROUP BY date ORDER BY date"},
        {"query": "统计每月订单数量", "sql": "SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count FROM orders GROUP BY month"},
        {"query": "统计每月销售额趋势", "sql": "SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM sales_records GROUP BY month ORDER BY month"},
        {"query": "查询每周销售数据", "sql": "SELECT strftime('%W', date) as week, SUM(amount) as total FROM sales_records GROUP BY week"},
        {"query": "统计各月客户注册数", "sql": "SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count FROM customers GROUP BY month"},
        {"query": "查询销售额月度环比", "sql": "SELECT strftime('%Y-%m', date) as month, SUM(amount) as total FROM sales_records GROUP BY month ORDER BY month"},
    ],

    # ========== 关联查询 (join) ==========
    "join": [
        {"query": "查询张三的所有订单", "sql": "SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.name = '张三'"},
        {"query": "查询李四的订单", "sql": "SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.name = '李四'"},
        {"query": "查询购买iPhone的客户", "sql": "SELECT DISTINCT c.* FROM customers c JOIN orders o ON c.id = o.customer_id JOIN products p ON o.product_id = p.id WHERE p.name LIKE '%iPhone%'"},
        {"query": "查询购买手机的客户", "sql": "SELECT DISTINCT c.* FROM customers c JOIN orders o ON c.id = o.customer_id JOIN products p ON o.product_id = p.id WHERE p.category = '手机'"},
        {"query": "查询VIP客户的订单", "sql": "SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.level = 'VIP'"},
        {"query": "查询北京客户的订单", "sql": "SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.region = '北京'"},
        {"query": "查询订单及对应产品信息", "sql": "SELECT o.*, p.name as product_name, p.price FROM orders o JOIN products p ON o.product_id = p.id"},
        {"query": "查询订单及客户信息", "sql": "SELECT o.*, c.name as customer_name, c.region FROM orders o JOIN customers c ON o.customer_id = c.id"},
        {"query": "查询手机类产品的订单", "sql": "SELECT o.* FROM orders o JOIN products p ON o.product_id = p.id WHERE p.category = '手机'"},
        {"query": "查询高价产品的订单", "sql": "SELECT o.* FROM orders o JOIN products p ON o.product_id = p.id WHERE p.price > 5000"},
    ],

    # ========== 关联聚合 (join_aggregation) ==========
    "join_aggregation": [
        {"query": "统计各客户的消费总额", "sql": "SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name"},
        {"query": "查询消费最多的客户", "sql": "SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total DESC LIMIT 1"},
        {"query": "统计各产品的销售数量", "sql": "SELECT p.name, SUM(o.quantity) as total_qty FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id, p.name"},
        {"query": "查询销量最高的产品", "sql": "SELECT p.name, SUM(o.quantity) as total_qty FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id, p.name ORDER BY total_qty DESC LIMIT 1"},
        {"query": "统计各地区客户消费总额", "sql": "SELECT c.region, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.region"},
        {"query": "统计各会员等级消费总额", "sql": "SELECT c.level, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.level"},
        {"query": "统计各类别产品销售额", "sql": "SELECT p.category, SUM(o.amount) as total FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.category"},
        {"query": "查询各客户订单数量", "sql": "SELECT c.name, COUNT(o.id) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name"},
        {"query": "查询各产品订单数量", "sql": "SELECT p.name, COUNT(o.id) as order_count FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.id, p.name"},
        {"query": "统计VIP客户总消费", "sql": "SELECT SUM(o.amount) as total FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.level = 'VIP'"},
    ],

    # ========== 复杂查询 (complex) ==========
    "complex": [
        {"query": "查询消费超过平均值的客户", "sql": "SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name HAVING total > (SELECT AVG(amount) FROM orders)"},
        {"query": "查询没有订单的客户", "sql": "SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.id IS NULL"},
        {"query": "查询没有销售的产品", "sql": "SELECT p.* FROM products p LEFT JOIN orders o ON p.id = o.product_id WHERE o.id IS NULL"},
        {"query": "查询订单数超过3的客户", "sql": "SELECT c.name, COUNT(o.id) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name HAVING order_count > 3"},
        {"query": "查询销售额占比最高的地区", "sql": "SELECT region, SUM(amount) as total, SUM(amount) * 100.0 / (SELECT SUM(amount) FROM sales_records) as percentage FROM sales_records GROUP BY region ORDER BY total DESC LIMIT 1"},
        {"query": "查询价格高于平均价格的产品", "sql": "SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products)"},
        {"query": "查询库存低于平均库存的产品", "sql": "SELECT * FROM products WHERE stock < (SELECT AVG(stock) FROM products)"},
        {"query": "查询每个地区销售额最高的销售员", "sql": "SELECT region, salesperson, SUM(amount) as total FROM sales_records GROUP BY region, salesperson ORDER BY region, total DESC"},
    ],

    # ========== 模糊查询 (fuzzy) ==========
    "fuzzy": [
        {"query": "查询名称包含iPhone的产品", "sql": "SELECT * FROM products WHERE name LIKE '%iPhone%'"},
        {"query": "查询名称包含Pro的产品", "sql": "SELECT * FROM products WHERE name LIKE '%Pro%'"},
        {"query": "查询姓张的客户", "sql": "SELECT * FROM customers WHERE name LIKE '张%'"},
        {"query": "查询姓李的客户", "sql": "SELECT * FROM customers WHERE name LIKE '李%'"},
        {"query": "查询邮箱包含qq的客户", "sql": "SELECT * FROM customers WHERE email LIKE '%qq%'"},
        {"query": "查询邮箱包含163的客户", "sql": "SELECT * FROM customers WHERE email LIKE '%163%'"},
        {"query": "查询名称以Air开头的产品", "sql": "SELECT * FROM products WHERE name LIKE 'Air%'"},
        {"query": "查询名称以Max结尾的产品", "sql": "SELECT * FROM products WHERE name LIKE '%Max'"},
    ],

    # ========== 去重查询 (distinct) ==========
    "distinct": [
        {"query": "查询所有产品类别", "sql": "SELECT DISTINCT category FROM products"},
        {"query": "查询所有客户地区", "sql": "SELECT DISTINCT region FROM customers"},
        {"query": "查询所有会员等级", "sql": "SELECT DISTINCT level FROM customers"},
        {"query": "查询所有订单状态", "sql": "SELECT DISTINCT status FROM orders"},
        {"query": "查询所有销售地区", "sql": "SELECT DISTINCT region FROM sales_records"},
        {"query": "查询所有销售员", "sql": "SELECT DISTINCT salesperson FROM sales_records"},
        {"query": "查询所有产品状态", "sql": "SELECT DISTINCT status FROM products"},
        {"query": "查询有订单的客户ID", "sql": "SELECT DISTINCT customer_id FROM orders"},
    ],

    # ========== 条件组合 (multi_condition) ==========
    "multi_condition": [
        {"query": "查询价格大于3000且在售的产品", "sql": "SELECT * FROM products WHERE price > 3000 AND status = 'on_sale'"},
        {"query": "查询北京或上海的VIP客户", "sql": "SELECT * FROM customers WHERE (region = '北京' OR region = '上海') AND level = 'VIP'"},
        {"query": "查询已完成且金额大于1000的订单", "sql": "SELECT * FROM orders WHERE status = 'completed' AND amount > 1000"},
        {"query": "查询华北或华东地区的销售记录", "sql": "SELECT * FROM sales_records WHERE region IN ('华北', '华东')"},
        {"query": "查询手机或电脑类产品", "sql": "SELECT * FROM products WHERE category IN ('手机', '电脑')"},
        {"query": "查询库存少于100或已下架的产品", "sql": "SELECT * FROM products WHERE stock < 100 OR status = 'off_sale'"},
        {"query": "查询VIP或金牌客户", "sql": "SELECT * FROM customers WHERE level IN ('VIP', '金牌')"},
        {"query": "查询待支付或已取消的订单", "sql": "SELECT * FROM orders WHERE status IN ('pending', 'cancelled')"},
    ],

    # ========== 计数统计 (count) ==========
    "count": [
        {"query": "有多少个产品", "sql": "SELECT COUNT(*) as count FROM products"},
        {"query": "有多少个客户", "sql": "SELECT COUNT(*) as count FROM customers"},
        {"query": "有多少个订单", "sql": "SELECT COUNT(*) as count FROM orders"},
        {"query": "手机类产品有多少个", "sql": "SELECT COUNT(*) as count FROM products WHERE category = '手机'"},
        {"query": "VIP客户有多少人", "sql": "SELECT COUNT(*) as count FROM customers WHERE level = 'VIP'"},
        {"query": "已完成订单有多少", "sql": "SELECT COUNT(*) as count FROM orders WHERE status = 'completed'"},
        {"query": "北京有多少客户", "sql": "SELECT COUNT(*) as count FROM customers WHERE region = '北京'"},
        {"query": "价格超过5000的产品有几个", "sql": "SELECT COUNT(*) as count FROM products WHERE price > 5000"},
    ],

    # ========== 求和统计 (sum) ==========
    "sum": [
        {"query": "所有产品的总库存", "sql": "SELECT SUM(stock) as total_stock FROM products"},
        {"query": "所有订单的总金额", "sql": "SELECT SUM(amount) as total_amount FROM orders"},
        {"query": "所有销售记录的总额", "sql": "SELECT SUM(amount) as total FROM sales_records"},
        {"query": "手机类产品总库存", "sql": "SELECT SUM(stock) as total_stock FROM products WHERE category = '手机'"},
        {"query": "VIP客户总消费", "sql": "SELECT SUM(o.amount) as total FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.level = 'VIP'"},
        {"query": "华北地区总销售额", "sql": "SELECT SUM(amount) as total FROM sales_records WHERE region = '华北'"},
        {"query": "12月份总销售额", "sql": "SELECT SUM(amount) as total FROM sales_records WHERE date LIKE '%-12-%'"},
        {"query": "已完成订单总金额", "sql": "SELECT SUM(amount) as total FROM orders WHERE status = 'completed'"},
    ],

    # ========== 平均值 (avg) ==========
    "avg": [
        {"query": "产品平均价格是多少", "sql": "SELECT AVG(price) as avg_price FROM products"},
        {"query": "订单平均金额", "sql": "SELECT AVG(amount) as avg_amount FROM orders"},
        {"query": "平均库存量", "sql": "SELECT AVG(stock) as avg_stock FROM products"},
        {"query": "手机类产品平均价格", "sql": "SELECT AVG(price) as avg_price FROM products WHERE category = '手机'"},
        {"query": "VIP客户平均消费", "sql": "SELECT AVG(o.amount) as avg FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.level = 'VIP'"},
        {"query": "各地区平均销售额", "sql": "SELECT region, AVG(amount) as avg_amount FROM sales_records GROUP BY region"},
        {"query": "各类别产品平均价格", "sql": "SELECT category, AVG(price) as avg_price FROM products GROUP BY category"},
        {"query": "各销售员平均业绩", "sql": "SELECT salesperson, AVG(amount) as avg_amount FROM sales_records GROUP BY salesperson"},
    ],

    # ========== 最值查询 (minmax) ==========
    "minmax": [
        {"query": "最贵的产品是哪个", "sql": "SELECT * FROM products ORDER BY price DESC LIMIT 1"},
        {"query": "最便宜的产品", "sql": "SELECT * FROM products ORDER BY price ASC LIMIT 1"},
        {"query": "库存最多的产品", "sql": "SELECT * FROM products ORDER BY stock DESC LIMIT 1"},
        {"query": "库存最少的产品", "sql": "SELECT * FROM products ORDER BY stock ASC LIMIT 1"},
        {"query": "金额最大的订单", "sql": "SELECT * FROM orders ORDER BY amount DESC LIMIT 1"},
        {"query": "金额最小的订单", "sql": "SELECT * FROM orders ORDER BY amount ASC LIMIT 1"},
        {"query": "最高单笔销售记录", "sql": "SELECT * FROM sales_records ORDER BY amount DESC LIMIT 1"},
        {"query": "产品最高价格是多少", "sql": "SELECT MAX(price) as max_price FROM products"},
        {"query": "产品最低价格是多少", "sql": "SELECT MIN(price) as min_price FROM products"},
        {"query": "订单最大金额是多少", "sql": "SELECT MAX(amount) as max_amount FROM orders"},
    ],

    # ========== 空值查询 (null) ==========
    "null": [
        {"query": "查询没有邮箱的客户", "sql": "SELECT * FROM customers WHERE email IS NULL"},
        {"query": "查询有邮箱的客户", "sql": "SELECT * FROM customers WHERE email IS NOT NULL"},
        {"query": "查询没有电话的客户", "sql": "SELECT * FROM customers WHERE phone IS NULL"},
        {"query": "查询有电话的客户", "sql": "SELECT * FROM customers WHERE phone IS NOT NULL"},
    ],

    # ========== 字段选择 (select_fields) ==========
    "select_fields": [
        {"query": "查询产品名称和价格", "sql": "SELECT name, price FROM products"},
        {"query": "查询客户姓名和地区", "sql": "SELECT name, region FROM customers"},
        {"query": "查询订单ID和金额", "sql": "SELECT id, amount FROM orders"},
        {"query": "查询产品名称、类别和库存", "sql": "SELECT name, category, stock FROM products"},
        {"query": "查询客户姓名、等级和地区", "sql": "SELECT name, level, region FROM customers"},
        {"query": "查询销售日期和金额", "sql": "SELECT date, amount FROM sales_records"},
        {"query": "只看产品名称", "sql": "SELECT name FROM products"},
        {"query": "只看客户邮箱", "sql": "SELECT email FROM customers"},
    ],

    # ========== 别名查询 (alias) ==========
    "alias": [
        {"query": "查询产品数量并命名为总数", "sql": "SELECT COUNT(*) as 总数 FROM products"},
        {"query": "查询销售总额并命名为总销售额", "sql": "SELECT SUM(amount) as 总销售额 FROM sales_records"},
        {"query": "查询各地区客户数并命名", "sql": "SELECT region as 地区, COUNT(*) as 客户数 FROM customers GROUP BY region"},
        {"query": "查询产品均价并命名", "sql": "SELECT AVG(price) as 平均价格 FROM products"},
    ],
}


def generate_queries():
    """生成评测集"""
    queries = []
    query_id = 1

    for category, items in TEMPLATES.items():
        for item in items:
            queries.append({
                "id": query_id,
                "query": item["query"],
                "expected_sql": item["sql"],
                "category": category,
            })
            query_id += 1

    return queries


def main():
    output_path = Path(__file__).parent.parent / "mock_data" / "queries" / "test_queries.json"

    queries = generate_queries()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(queries)} test queries")
    print(f"Output: {output_path}")

    # 统计各类别数量
    from collections import Counter
    categories = Counter(q["category"] for q in queries)
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
