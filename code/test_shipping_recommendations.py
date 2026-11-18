import datetime

import pandas as pd
import pytest

import tools
import tools_three


START = datetime.datetime(2026, 11, 9)


@pytest.fixture(params=[tools, tools_three], ids=["tools", "tools_three"])
def toolset(request):
    return request.param


def daily_shortages(periods, days):
    shortages = [0.0] * days
    for period in periods:
        start = period["lost_sales_day"]
        shortages[start:start + period["duration"]] = period["daily_lost_sales"]
    return shortages


def test_natural_month_sample(toolset):
    nov, dec, jan = 28.5667829310983, 24.0681914355545, 22.3341125640058
    forecast = pd.DataFrame({
        "MSKU日均": [nov, dec, jan],
        "当前可用天数": [22, 31, 31],
        "MSKU日均列表": [[], [], []],
    })
    demand = toolset.get_daily_sales_parm(65, forecast)
    assert demand == [nov] * 22 + [dec] * 31 + [jan] * 12

    days, lost, periods, first_stockout = toolset.calculate_inventory(
        START, 695, demand, [datetime.datetime(2026, 11, 25)], [378]
    )
    assert lost == pytest.approx(569.5925097544216)
    assert days == 25
    assert first_stockout == 40
    assert periods[0]["start_date"] == datetime.datetime(2026, 12, 19)
    shortages = daily_shortages(periods, 65)
    assert sum(shortages) == pytest.approx(lost)
    assert toolset.generate_shipment_suggestion(shortages, 6, 25, 40) == (0, 0, 570)


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
def test_shipping_boundaries_and_last_day(toolset, day, expected):
    shortages = [0.0] * 65
    shortages[day] = 1.25
    assert toolset.generate_shipment_suggestion(shortages, 6, 25, 40) == expected


def test_gap_starting_before_dhl_still_replenishes_after_arrival(toolset):
    assert toolset.generate_shipment_suggestion([10.0] * 15, 6, 25, 40) == (90, 0, 0)


def test_first_shortage_day_only_counts_unmet_demand(toolset):
    days, lost, periods, first_stockout = toolset.calculate_inventory(
        START, 65, [10.0] * 7, [], []
    )
    assert (days, lost, first_stockout) == (1, 5, 6)
    shortages = daily_shortages(periods, 7)
    assert shortages == [0.0] * 6 + [5.0]
    assert toolset.generate_shipment_suggestion(shortages, 6, 25, 40) == (5, 0, 0)


def test_small_arrival_during_stockout_reduces_that_days_shortage(toolset):
    days, lost, periods, first_stockout = toolset.calculate_inventory(
        START, 0, [10.0] * 9, [START + datetime.timedelta(days=7)], [3]
    )
    assert (days, lost, first_stockout) == (9, 87, 0)
    shortages = daily_shortages(periods, 9)
    assert shortages == [10.0] * 7 + [7.0, 10.0]
    assert toolset.generate_shipment_suggestion(shortages, 6, 25, 40) == (27, 0, 0)


def test_selling_exactly_available_inventory_is_not_a_stockout(toolset):
    days, lost, periods, first_stockout = toolset.calculate_inventory(
        START, 10, [10.0, 10.0], [], []
    )
    assert (days, lost, first_stockout) == (1, 10, 1)
    assert periods[0]["start_date"] == START + datetime.timedelta(days=1)


def test_fractional_daily_sales_do_not_create_a_rounding_stockout(toolset):
    demand = [10 / 3] * 3
    assert toolset.calculate_inventory(START, 10, demand, [], []) == (0, 0, [], 3)
    assert toolset.calculate_available_sales(10, demand) == 3


def test_simulation_uses_the_starting_calendar_day_as_day_zero(toolset):
    start = START.replace(hour=18, minute=45)
    _, _, periods, first_stockout = toolset.calculate_inventory(start, 0, [10], [], [])
    assert first_stockout == 0
    assert periods[0]['lost_sales_day'] == 0


def test_multiple_arrivals_same_day_are_all_available_before_sales(toolset):
    days, lost, periods, first_stockout = toolset.calculate_inventory(
        START, 0, [10.0], [START, START], [4, 6]
    )
    assert (days, lost, periods, first_stockout) == (0, 0, [], 1)


@pytest.mark.parametrize("target", [10, 20, 21, 50, 65])
def test_daily_sales_target_within_first_month_and_across_months(toolset, target):
    forecast = pd.DataFrame({
        "MSKU日均": [1.0, 2.0, 3.0],
        "当前可用天数": [20, 30, 31],
        "MSKU日均列表": [[], [], []],
    })
    expected = ([1.0] * 20 + [2.0] * 30 + [3.0] * 31)[:target]
    assert toolset.get_daily_sales_parm(target, forecast) == expected


@pytest.fixture
def merge_inputs():
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
        "MSKU日均列表": [[]],
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
    return predict, sales, inventory, shipments


def test_explicit_historical_date_is_normalized_and_months_are_sorted(toolset, merge_inputs):
    merged, _, _ = toolset.process_datas(
        *merge_inputs,
        current_date=pd.Timestamp("2020-11-09 18:45:00"),
    )
    assert merged["当前可用天数"].tolist() == [22, 31, 31]
    assert merged["MSKU日均"].tolist() == [10.0, 20.0, 30.0]


def test_no_stockouts_still_provides_named_detail_columns(toolset):
    result = toolset.process_stockout_dataframe({"ID": ["x"], "断货详情": ["未发生断货"]})
    expected = {"断货时间1", "断货总天数1", "损失销量1", "断货开始天数1"}
    assert expected.issubset(result.columns)
    assert all(isinstance(column, str) for column in result.columns)


def prefix_forecast(prefix):
    return pd.DataFrame({
        "MSKU日均": [10.0, 20.0],
        "当前可用天数": [2, 31],
        "MSKU日均列表": [prefix, []],
    })


def test_custom_prefix_returns_to_the_actual_calendar_month_forecast():
    forecast = prefix_forecast([99.0, 98.0, 97.0])
    assert tools_three.get_daily_sales_parm(5, forecast) == [99.0, 98.0, 97.0, 20.0, 20.0]


def test_custom_prefix_cannot_extend_the_target_horizon():
    forecast = prefix_forecast([99.0, 98.0, 97.0, 96.0, 95.0, 94.0])
    assert tools_three.get_daily_sales_parm(3, forecast) == [99.0, 98.0, 97.0]


def test_custom_prefix_and_month_lengths_are_not_modified():
    prefix = [99.0, 98.0, 97.0]
    forecast = prefix_forecast(prefix)
    original = forecast.copy(deep=True)
    demand = tools_three.get_daily_sales_parm(5, forecast)
    demand[0] = -1
    assert prefix == [99.0, 98.0, 97.0]
    pd.testing.assert_frame_equal(forecast, original)


def test_operating_daily_sales_are_scaled_and_keep_the_calendar_fallback(merge_inputs):
    predict, sales, inventory, shipments = merge_inputs
    predict["日均1"] = [None, 80.0, None]
    predict["天数1"] = [None, 2, None]
    sales["款式销占比"] = 0.5
    sales["SKU销占比"] = 0.25
    listing_prefixes = tools_three.build_daily_sales_list_prefix(predict)
    msku_sales = tools_three.msku_avg_days_list(sales, listing_prefixes)
    assert msku_sales["MSKU日均列表"].iloc[0] == [10.0, 10.0]

    merged, _, _ = tools_three.process_datas(
        predict, msku_sales, inventory, shipments,
        current_date=pd.Timestamp("2020-11-09 18:45:00"),
    )
    demand = tools_three.get_daily_sales_parm(25, merged)
    assert demand == [10.0] * 2 + [1.25] * 20 + [2.5] * 3


@pytest.mark.parametrize(
    ("window_shortages", "expected"),
    [
        ((0.2, 0.2, 0.3), (1, 0, 0)),
        ((0.2, 1.1, 0.5), (1, 1, 0)),
        ((0.2, 0.0, 0.3), (1, 0, 0)),
        ((0.0, 0.2, 0.3), (0, 1, 0)),
        ((0.3, 0.6, 0.1), (1, 0, 0)),
        ((0.33333333336, 0.33333333336, 0.33333333328), (1, 0, 0)),
    ],
    ids=[
        "one_unit_covers_three_windows",
        "air_rounds_only_the_remaining_shortage",
        "surplus_survives_an_empty_air_window",
        "no_early_dhl_for_an_empty_window",
        "decimal_sum_boundary",
        "carry_keeps_precision_between_windows",
    ],
)
def test_rounding_surplus_carries_forward_and_covers_demand(window_shortages, expected):
    arrivals = [START + datetime.timedelta(days=day) for day in (6, 25, 40)]
    shortages = [0.0] * 65
    for day, shortage in zip((6, 25, 40), window_shortages):
        shortages[day] = shortage

    suggestion = tools_three.generate_shipment_suggestion(shortages, 6, 25, 40)
    assert suggestion == expected

    _, lost, periods, _ = tools_three.calculate_inventory(
        START, 0, shortages, arrivals, suggestion,
    )
    assert lost == 0
    assert daily_shortages(periods, 65) == [0.0] * 65


def test_rounding_surplus_with_existing_stock_and_arrivals_preserves_unavoidable_loss():
    demand = [0.0] * 65
    for day, quantity in ((0, 0.2), (1, 0.6), (2, 0.1), (5, 0.2),
                          (6, 0.5), (25, 0.5), (40, 0.3)):
        demand[day] = quantity
    original_arrivals = [START + datetime.timedelta(days=day) for day in (6, 25)]
    original_quantities = [0.3, 0.3]
    _, baseline_loss, periods, _ = tools_three.calculate_inventory(
        START, 0.45, demand, original_arrivals, original_quantities,
    )
    shortages = daily_shortages(periods, 65)
    assert baseline_loss == pytest.approx(1.35)
    assert sum(shortages[:6]) == pytest.approx(0.65)
    assert [shortages[day] for day in (6, 25, 40)] == pytest.approx([0.2, 0.2, 0.3])

    suggestion = tools_three.generate_shipment_suggestion(shortages, 6, 25, 40)
    assert suggestion == (1, 0, 0)
    new_arrivals = [START + datetime.timedelta(days=day) for day in (6, 25, 40)]
    _, actual_loss, periods, _ = tools_three.calculate_inventory(
        START, 0.45, demand,
        original_arrivals + new_arrivals,
        original_quantities + list(suggestion),
    )
    remaining_shortages = daily_shortages(periods, 65)
    assert remaining_shortages[:6] == pytest.approx(shortages[:6])
    assert remaining_shortages[6:] == [0.0] * 59
    assert actual_loss == pytest.approx(0.65)
