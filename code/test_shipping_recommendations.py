import datetime

import pandas as pd
import pytest

from tools import (
    calculate_available_sales,
    calculate_inventory,
    generate_shipment_suggestion,
    get_daily_sales_parm,
    process_datas,
    process_stockout_dataframe,
)


START = datetime.datetime(2026, 11, 9)


def daily_shortages(periods, days):
    shortages = [0.0] * days
    for period in periods:
        start = period["lost_sales_day"]
        shortages[start:start + period["duration"]] = period["daily_lost_sales"]
    return shortages


def test_natural_month_sample():
    nov, dec, jan = 28.5667829310983, 24.0681914355545, 22.3341125640058
    forecast = pd.DataFrame({
        "MSKU日均": [nov, dec, jan],
        "当前可用天数": [22, 31, 31],
    })
    demand = get_daily_sales_parm(65, forecast)
    assert demand == [nov] * 22 + [dec] * 31 + [jan] * 12

    days, lost, periods, first_stockout = calculate_inventory(
        START, 695, demand, [datetime.datetime(2026, 11, 25)], [378]
    )
    assert lost == pytest.approx(569.5925097544216)
    assert days == 25
    assert first_stockout == 40
    assert periods[0]["start_date"] == datetime.datetime(2026, 12, 19)
    shortages = daily_shortages(periods, 65)
    assert sum(shortages) == pytest.approx(lost)
    assert generate_shipment_suggestion(shortages, 6, 25, 40) == (0, 0, 570)


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (5, (0, 0, 0)),
        (6, (2, 0, 0)),
        (7, (2, 0, 0)),
        (24, (2, 0, 0)),
        (25, (0, 2, 0)),
        (26, (0, 2, 0)),
        (39, (0, 2, 0)),
        (40, (0, 0, 2)),
        (41, (0, 0, 2)),
        (64, (0, 0, 2)),
    ],
)
def test_shipping_boundaries_and_last_day(day, expected):
    shortages = [0.0] * 65
    shortages[day] = 1.25
    assert generate_shipment_suggestion(shortages, 6, 25, 40) == expected


def test_gap_starting_before_dhl_still_replenishes_after_arrival():
    assert generate_shipment_suggestion([10.0] * 15, 6, 25, 40) == (90, 0, 0)


def test_first_shortage_day_only_counts_unmet_demand():
    days, lost, periods, first_stockout = calculate_inventory(
        START, 65, [10.0] * 7, [], []
    )
    assert (days, lost, first_stockout) == (1, 5, 6)
    shortages = daily_shortages(periods, 7)
    assert shortages == [0.0] * 6 + [5.0]
    assert generate_shipment_suggestion(shortages, 6, 25, 40) == (5, 0, 0)


def test_small_arrival_during_stockout_reduces_that_days_shortage():
    days, lost, periods, first_stockout = calculate_inventory(
        START, 0, [10.0] * 9, [START + datetime.timedelta(days=7)], [3]
    )
    assert (days, lost, first_stockout) == (9, 87, 0)
    shortages = daily_shortages(periods, 9)
    assert shortages == [10.0] * 7 + [7.0, 10.0]
    assert generate_shipment_suggestion(shortages, 6, 25, 40) == (27, 0, 0)


def test_selling_exactly_available_inventory_is_not_a_stockout():
    days, lost, periods, first_stockout = calculate_inventory(
        START, 10, [10.0, 10.0], [], []
    )
    assert (days, lost, first_stockout) == (1, 10, 1)
    assert periods[0]["start_date"] == START + datetime.timedelta(days=1)


def test_fractional_daily_sales_do_not_create_a_rounding_stockout():
    demand = [10 / 3] * 3
    assert calculate_inventory(START, 10, demand, [], []) == (0, 0, [], 3)
    assert calculate_available_sales(10, demand) == 3


def test_simulation_uses_the_starting_calendar_day_as_day_zero():
    start = START.replace(hour=18, minute=45)
    _, _, periods, first_stockout = calculate_inventory(start, 0, [10], [], [])
    assert first_stockout == 0
    assert periods[0]['lost_sales_day'] == 0


def test_multiple_arrivals_same_day_are_all_available_before_sales():
    days, lost, periods, first_stockout = calculate_inventory(
        START, 0, [10.0], [START, START], [4, 6]
    )
    assert (days, lost, periods, first_stockout) == (0, 0, [], 1)


@pytest.mark.parametrize("target", [10, 20, 21, 50, 65])
def test_daily_sales_target_within_first_month_and_across_months(target):
    forecast = pd.DataFrame({
        "MSKU日均": [1.0, 2.0, 3.0],
        "当前可用天数": [20, 30, 31],
    })
    expected = ([1.0] * 20 + [2.0] * 30 + [3.0] * 31)[:target]
    assert get_daily_sales_parm(target, forecast) == expected


def test_explicit_historical_date_is_normalized_and_months_are_sorted():
    predict = pd.DataFrame({
        "站点": ["US"] * 3,
        "Listing": ["listing"] * 3,
        "月份": pd.to_datetime(["2021-01-01", "2020-11-01", "2020-12-01"]),
        "Listing_月度预估销量": [930, 300, 620],
    })
    sales = pd.DataFrame({
        "站点": ["US"], "Listing": ["listing"], "店铺-站点": ["shop:US"],
        "MSKU": ["msku"], "积加SKU": ["sku"],
        "款式销占比": [1.0], "SKU销占比": [1.0],
    })
    inventory = pd.DataFrame({
        "仓库": ["shop:US"], "SKU": ["sku"], "MSKU": ["msku"],
        "FBA可用库存": [10],
    })
    shipments = pd.DataFrame({
        "仓库": ["shop:US_FBA"], "SKU": ["sku"], "MSKU": ["msku"],
        "状态": ["已出运"], "预计到仓日期": pd.to_datetime(["2020-11-25"]),
        "发货量": [5],
    })
    merged, _, _ = process_datas(
        predict, sales, inventory, shipments,
        current_date=pd.Timestamp("2020-11-09 18:45:00"),
    )
    assert merged["当前可用天数"].tolist() == [22, 31, 31]
    assert merged["MSKU日均"].tolist() == [10.0, 20.0, 30.0]


def test_no_stockouts_still_provides_named_detail_columns():
    result = process_stockout_dataframe({"ID": ["x"], "断货详情": ["未发生断货"]})
    expected = {"断货时间1", "断货总天数1", "损失销量1", "断货开始天数1"}
    assert expected.issubset(result.columns)
    assert all(isinstance(column, str) for column in result.columns)
