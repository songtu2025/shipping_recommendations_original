import re
import pandas as pd
import numpy as np
import calendar
import math
import datetime
from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ---------------------------构建未来最近n天的日均销量列表，例如构建未来30天的销量列表，该日均及天数由运营直接提供-------------------
# 动态扩展场景，写一个自动识别 日均N/天数N 的通用版本，不需要手工写死 日均1、日均2
def get_avg_days_pairs(columns):
    pairs = []

    for col in columns:
        match = re.fullmatch(r"日均(\d+)", str(col))
        if not match:
            continue

        n = match.group(1)
        avg_col = f"日均{n}"
        days_col = f"天数{n}"

        if days_col in columns:
            pairs.append((int(n), avg_col, days_col))

    pairs.sort(key=lambda x: x[0])

    return [(avg_col, days_col) for _, avg_col, days_col in pairs]


def parse_days(days_value):
    if pd.isna(days_value):
        return 0

    text = str(days_value).strip()
    text = text.replace("～", "~").replace("-", "~")

    if "~" in text:
        start, end = text.split("~", 1)
        start = int(float(start.strip()))
        end = int(float(end.strip()))
        return end - start

    return int(float(text))

def build_daily_list(row, avg_days_pairs):
    daily_list = []

    for avg_col, days_col in avg_days_pairs:
        avg_value = row[avg_col]
        days_value = row[days_col]

        if pd.isna(avg_value) or pd.isna(days_value):
            continue

        days = parse_days(days_value)

        if days <= 0:
            continue

        daily_list.extend([avg_value] * days)

    return daily_list


def build_daily_sales_list_prefix(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    avg_days_pairs = get_avg_days_pairs(df.columns)

    result = (
        df.groupby(["站点", "Listing"], as_index=False)
          .first()
    )

    drop_cols = ["月份", "Listing_月度预估销量"]
    result = result.drop(columns=[col for col in drop_cols if col in result.columns])

    result["日均列表"] = result.apply(
        lambda row: build_daily_list(row, avg_days_pairs),
        axis=1
    )

    result = result[["站点", "Listing", "日均列表"]]

    return result

def scale_msku_daily_list(row):
    """
    将 Listing 维度日均列表按 款式销占比 和 SKU销占比 拆分到 MSKU 维度。
    """
    daily_list = row["日均列表"]
    style_share = row["款式销占比"]
    sku_share = row["SKU销占比"]

    # 如果listing维度的日均列表为空，则返回空列表
    if not isinstance(daily_list, list):
        return []

    if pd.isna(style_share) or pd.isna(sku_share):
        return []

    factor = style_share * sku_share

    return [value * factor for value in daily_list]

def msku_avg_days_list(sales_df, listing_daily_ls):
    """
    合并 Listing 日均列表，并生成 MSKU 日均列表。
    """
    msku_avg_days_df = sales_df.merge(
        listing_daily_ls[["站点", "Listing", "日均列表"]],
        how="left",
        on=["站点", "Listing"]
    )

    msku_avg_days_df["MSKU日均列表"] = msku_avg_days_df.apply(scale_msku_daily_list, axis=1)

    # 删除Listing的日均列表
    msku_avg_days_df = msku_avg_days_df.drop(columns=["日均列表"])

    return msku_avg_days_df

# 计算库存情况的函数
def calculate_inventory(start_date, initial_inventory, daily_sales, restock_dates, restock_quantities):
    """
    计算库存情况的函数
    Parameters:
    - start_date: datetime.datetime，起始日期
    - initial_inventory: float，初始库存
    - daily_sales: list，每日销售量的列表
    - restock_dates: list，补货日期的列表
    - restock_quantities: list，每次补货数量的列表

    Returns:
    Tuple，包含缺货天数、缺货期间销售量损失、缺货时间段列表、在库可售天数
    """

    start_date = pd.Timestamp(start_date).normalize()
    current_date = start_date
    inventory = initial_inventory  # 初始库存
    days_without_stock = 0  # 缺货天数
    lost_sales = 0  # 缺货期间销售量损失
    stockout_periods = []  # 缺货时间段列表
    available_sales = len(daily_sales)  # 初始化总目标可售天数
    is_first = True  # 记录是否是第一个缺货时间段

    # 同一天的多笔到货合并，当天先入库再销售。
    restocks = {}
    for date, quantity in zip(restock_dates, restock_quantities):
        date = pd.Timestamp(date).normalize()
        restocks[date] = restocks.get(date, 0) + quantity

    # 遍历每日销售量的列表，模拟每一天的销售情况。
    for sales_quantity in daily_sales:

        # 检查当前日期是否有补货
        inventory += restocks.get(current_date, 0)

        # 根据每日销售更新库存
        inventory -= sales_quantity
        # 小数日均累减可能留下极小负数，数学上的零库存不记为缺货。
        if math.isclose(inventory, 0, abs_tol=1e-10):
            inventory = 0

        # 检查库存是否为负数（缺货）
        if inventory < 0:
            # 如果 stockout_periods 为空或者上一个缺货时间段的结束日期不是前一天，表示这是一个新的缺货时间段
            if not stockout_periods or stockout_periods[-1]['end_date'] != current_date - timedelta(days=1):
                # 断货开始时间距离当前的天数，计算从 start_date 到当前日期的天数 lost_sales_day
                lost_sales_day = (current_date - start_date).days
                stockout_periods.append(
                    {'start_date': current_date, 'end_date': current_date, 'lost_sales': abs(inventory),
                     'lost_sales_day': lost_sales_day, 'duration': 1,
                     'daily_lost_sales': [abs(inventory)]})
            # 否则，更新最后一个缺货时间段的结束日期、销售量损失和持续时间。
            else:
                stockout_periods[-1]['end_date'] = current_date
                stockout_periods[-1]['lost_sales'] += abs(inventory)
                stockout_periods[-1]['duration'] += 1
                stockout_periods[-1]['daily_lost_sales'].append(abs(inventory))

            # 在库可售天数
            if is_first: # 如果是第一个缺货时间段，记录从 start_date 到当前日期的天数作为在库可售天数。
                # print(f'current_date: {current_date},type：{type(current_date)}')
                available_sales = (current_date - start_date).days
                is_first = False
            days_without_stock += 1 # 增加缺货天数
            lost_sales += abs(inventory) # 累加缺货期间的销售量损失
            inventory = 0  # 将库存设置为 0，表示缺货

        # 移动到下一天；将 current_date 增加一天，模拟下一天的销售情况。
        current_date += timedelta(days=1)
        # print(f"current_date: {current_date}, sales_quantity: {sales_quantity}, inventory: {inventory}, lost_sales: {lost_sales}, days_without_stock: {days_without_stock}")

    return days_without_stock, lost_sales, stockout_periods, available_sales


def calculate_available_sales(initial_inventory, daily_sales):
    """
    计算在库可售天数

    Parameters:
        initial_inventory (float): 初始库存
        daily_sales (list): 每日销售量列表

    Returns:
        available_sales(int): 在库可售天数
    """
    inventory = initial_inventory  # 初始库存
    available_sales = 0  # 初始化在库可售天数
    for sales_quantity in daily_sales:
        inventory -= sales_quantity
        if math.isclose(inventory, 0, abs_tol=1e-10):
            inventory = 0
        if inventory >= 0:
            available_sales += 1
        else:
            break
    return available_sales

def process_datas(predict_df, sales_df, inventory_df, shipment_df, current_date=None):
    """
      处理数据，生成合并后的数据框、发货数据透视表和SKU库存状况表

    Parameters:
    predict_df (DataFrame): 预测数据
    sales_df (DataFrame): 销售数据
    inventory_df (DataFrame): 库存数据
    shipment_df (DataFrame): 发货数据

    Returns:
    merge_df (DataFrame): 合并后的数据框
    shipment_ret (DataFrame): 发货数据透视表
    sku_inv (DataFrame): SKU库存状况表
    """
    # 从仓库中提取 店铺-站点 信息
    shipment_df['店铺-站点'] = shipment_df['仓库'].str.split('_FBA', expand=True)[0]

    # SKU维度的库存状况。这里的状态有：已出运、未出库、提货中
    sku_inv = shipment_df.pivot_table(index=['店铺-站点', 'SKU', 'MSKU'], columns=['状态'], values='发货量',
                                      aggfunc=sum, fill_value=0).reset_index()

    # 透视并汇总发货数据
    shipment_ret = shipment_df.pivot_table(index=['店铺-站点', 'SKU', 'MSKU', '预计到仓日期'], values='发货量',
                                           aggfunc=sum, fill_value=0).reset_index()
    shipment_ret['预计到仓日期'] = pd.to_datetime(shipment_ret['预计到仓日期'])
    shipment_ret['ID'] = shipment_ret[['MSKU', 'SKU', '店铺-站点']].sum(axis=1)

    # 与库存模拟共用起始日期，起始日为第0天，并计入当月预测。
    predict_df = predict_df.copy()
    predict_df['月份'] = pd.to_datetime(predict_df['月份']).dt.normalize()
    predict_df['当月总天数'] = predict_df['月份'].apply(lambda x: calendar.monthrange(x.year, x.month)[1])  # 获取当月的总天数
    reference_date = pd.Timestamp(current_date if current_date is not None else datetime.datetime.now()).normalize()
    predict_df['当前可用天数'] = (predict_df['月份'] + pd.offsets.MonthEnd(0) - reference_date).dt.days + 1

    predict_df = predict_df.query('当前可用天数 > 0').sort_values('月份', kind='stable')
    # 对 '当前可用天数' 列进行 clip 操作
    predict_df['当前可用天数'] = np.clip(predict_df['当前可用天数'], a_min=0, a_max=predict_df['当月总天数'])

    # 合并数据框
    merge_df = pd.merge(left=sales_df,
                        right=predict_df[['站点', 'Listing', '当月总天数', '当前可用天数', 'Listing_月度预估销量']],
                        on=['站点', 'Listing'], how='left')

    # 计算MSKU每日均值
    merge_df['MSKU日均'] = (merge_df['Listing_月度预估销量'] / merge_df['当月总天数']) * merge_df['款式销占比'] * \
                           merge_df['SKU销占比']

    # 与库存数据合并
    merge_df = pd.merge(left=merge_df, right=inventory_df, left_on=['店铺-站点', '积加SKU', 'MSKU'],
                        right_on=['仓库', 'SKU', 'MSKU'], how='left')

    # 删除不必要的列
    merge_df.drop(columns=['仓库', 'SKU', '当月总天数'], inplace=True)

    # 计算ID列
    merge_df['ID'] = merge_df[['MSKU', '积加SKU', '店铺-站点']].sum(axis=1)

    return merge_df, shipment_ret, sku_inv

def replace_prefix(a, b):
    """用自定义日均覆盖预测前段，保持原预测长度且不修改输入列表。"""
    result = b.copy()
    count = min(len(a), len(result))
    result[:count] = a[:count]
    return result

def get_daily_sales_parm(target_days, temp_df):
    """按真实月份展开预测，自定义MSKU日均从第0天覆盖，最多取目标天数。"""
    remaining = int(target_days)
    daily_sales = []
    for average, days in zip(temp_df['MSKU日均'], temp_df['当前可用天数']):
        if remaining <= 0:
            break
        count = min(int(days), remaining)
        daily_sales.extend([average] * count)
        remaining -= count
    custom_daily_sales = temp_df['MSKU日均列表'].iloc[0]
    return replace_prefix(custom_daily_sales, daily_sales)

def get_total_sales(target_days, temp_df):
    """
    获取指定目标天数的总销量的函数。

    Parameters:
    - target_days: int，目标天数
    - temp_df: DataFrame，包含销售数据的DataFrame

    Returns:
    int，指定目标天数的总销量
    """

    sales_avg = temp_df['MSKU日均'].reset_index(drop=True)
    curr_days = temp_df['当前可用天数'].reset_index(drop=True)
    # 找到累计和超过目标天数的索引
    cumulative_sum = np.cumsum(curr_days)
    index = np.argmax(cumulative_sum > target_days)
    # print(f'-------------{curr_days}----------------')
    # 更新 curr_days 和 sales_avg
    if index < len(curr_days):
        curr_days.iloc[index] = target_days - (cumulative_sum[index] - curr_days.iloc[index])
    total_sales = sum(sales_avg.iloc[:index + 1] * curr_days.iloc[:index + 1])

    return int(total_sales)


def display_stockout_details(stockout_periods):
    """
    构建断货详情：断货时间段、销售量损失和缺货天数的函数。

    Parameters:
    - stockout_periods: list，包含断货时间段信息的列表
    """
    details = ''

    # 存在在途货件且断货
    if stockout_periods:  #如果 stockout_periods 列表不为空，表示存在断货时间段。
        length = len(stockout_periods)
        for i, period in enumerate(stockout_periods):
            # 断货开始日期、断货结束日期、断货天数、损失销量
            info = f"{period['start_date'].strftime('%Y-%m-%d')}   {period['end_date'].strftime('%Y-%m-%d')}|{period['duration']}|{round(period['lost_sales'], 6)}|{int(period['lost_sales_day'])}"
            if (i + 1) != length:
                details = details + info + '|'
            else:
                details = details + info
    else:
        details = '未发生断货'

    return details


def process_stockout_dataframe(stockout_dict):
    """
    处理断货信息字典并生成包含断货详情的DataFrame的函数。

    Parameters:
    - stockout_dict: dict，包含断货信息的字典

    Returns:
    - pd.DataFrame，包含断货详情的DataFrame
    """
    # 创建DataFrame
    stockout_df = pd.DataFrame(data=stockout_dict)

    # 将'断货详情'列的字符串按照 '|' 进行拆分，扩展为多列
    stockout_detail = stockout_df['断货详情'].str.split('|', expand=True)
    # 全部未缺货时也保留首组详情列，供主程序按固定列名导出。
    group_count = max(1, (stockout_detail.shape[1] + 3) // 4)
    stockout_detail = stockout_detail.reindex(columns=range(group_count * 4))

    # 获取拆分后的 DataFrame 的列名列表
    columns = stockout_detail.columns.to_list()

    # 循环处理每组 4 列的数据
    for i in range(0, len(columns) // 4):
        # 提取当前一组列名
        src_names = columns[i * 4: i * 4 + 4]

        # 重命名列名，添加更具描述性的名字
        new_column_names = [f'断货时间{i + 1}', f'断货总天数{i + 1}', f'损失销量{i + 1}', f'断货开始天数{i + 1}']
        stockout_detail.rename(columns=dict(zip(src_names, new_column_names)), inplace=True)

        # 将新列中的 NaN 值填充为 0，然后将数据类型转换为整数
        stockout_detail[new_column_names[1:]] = stockout_detail[new_column_names[1:]]  # .fillna(0).astype(dtype=int)
    # 合并DataFrame
    stockout_df = pd.concat(objs=[stockout_df, stockout_detail], axis=1)
    # stockout_df.to_excel("构建断货详情_processStockoutDataframe函数输出.xlsx", index = False)

    return stockout_df


# 生成发货建议
def generate_shipment_suggestion(daily_shortages, dhl_start_days, dhl_deadline, airport_deadline):
    """按第0天的实际缺口分渠道；到货当天可售，区间左闭右开。"""
    windows = (
        daily_shortages[dhl_start_days:dhl_deadline],
        daily_shortages[dhl_deadline:airport_deadline],
        daily_shortages[airport_deadline:],
    )
    # 每个渠道先合计再向上取整；消除浮点累加产生的极小误差。
    return tuple(math.ceil(round(sum(window), 10)) for window in windows)


# 定义一个函数来生成仓库优先级顺序
def generate_warehouse_priority(target_warehouse, warehouses):
    priority = [target_warehouse]  # 优先从目标仓库发货
    for warehouse in warehouses:
        if warehouse != target_warehouse:
            priority.append(warehouse)  # 将其他仓库按顺序添加到优先级列表中
    return priority


# 分配货物的函数
def allocate_goods(warehouses, shipping_channels, shipping_priority, target_warehouse):
    # 生成仓库优先级顺序
    warehouse_priority = generate_warehouse_priority(target_warehouse, warehouses)

    # 初始化分配结果
    allocation = {channel: {} for channel in shipping_channels}
    shortfall = {channel: 0 for channel in shipping_channels}  # 初始化缺口记录

    # 按发货渠道优先级分配货物
    for channel in shipping_priority:
        remaining = shipping_channels[channel]  # 当前渠道需要发货的剩余数量
        for warehouse in warehouse_priority:
            if remaining <= 0:
                break  # 如果当前渠道的货物已经分配完毕，跳出循环
            available = warehouses[warehouse]  # 当前仓库可用库存
            if available > 0:
                if available >= remaining:
                    allocation[channel][warehouse] = remaining  # 从当前仓库分配所需的全部货物
                    warehouses[warehouse] -= remaining  # 更新当前仓库的剩余库存
                    remaining = 0  # 当前渠道的货物分配完毕
                else:
                    allocation[channel][warehouse] = available  # 从当前仓库分配所有可用货物
                    warehouses[warehouse] = 0  # 当前仓库的货物分配完毕
                    remaining -= available  # 更新当前渠道需要分配的剩余数量

        if remaining > 0:
            shortfall[channel] = remaining  # 如果货物分配不足，记录缺口

    return allocation, warehouses, shortfall  # 返回分配结果、更新后的仓库库存和缺口记录


# 定义提取数字的函数
def extract_numbers(text, pattern):
    if text == '':
        return 0
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return 0


def process_value(value):
    """
    根据传入值判断返回结果：
    - 如果传入的值在 0 和 1 之间，返回 1。
    - 否则，将传入的值转换为整数并返回。
    """
    if 0 < value < 1:
        return 1
    else:
        return int(value)