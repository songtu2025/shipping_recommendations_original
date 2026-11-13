import pandas as pd
from month_sales_utils import aggregate_last_n_months_sales, aggregate_current_month_sales

def build_sample(now: pd.Timestamp) -> pd.DataFrame:
    data = [
        {"purchaseDate": now - pd.DateOffset(months=2), "销量": 5, "店铺-站点": "A:US", "MSKU": "X1"},
        {"purchaseDate": now - pd.DateOffset(months=2), "销量": 7, "店铺-站点": "A:US", "MSKU": "X2"},
        {"purchaseDate": now - pd.DateOffset(months=1), "销量": 3, "店铺-站点": "A:US", "MSKU": "X1"},
        {"purchaseDate": now - pd.DateOffset(months=1), "销量": 4, "店铺-站点": "A:US", "MSKU": "X2"},
        {"purchaseDate": now, "销量": 2, "店铺-站点": "A:US", "MSKU": "X1"},
        {"purchaseDate": now, "销量": 9, "店铺-站点": "A:US", "MSKU": "X2"},
    ]
    return pd.DataFrame(data)

def main():
    now = pd.Timestamp.now().normalize()
    df = build_sample(now)
    last_two = aggregate_last_n_months_sales(df, n=2, now=now)
    curr = aggregate_current_month_sales(df, now=now)
    msku_x1_last = int(last_two.loc[last_two["MSKU"] == "X1", "上月销量"].iloc[0])
    msku_x2_last = int(last_two.loc[last_two["MSKU"] == "X2", "上月销量"].iloc[0])
    msku_x1_curr = int(curr.loc[curr["MSKU"] == "X1", "本月销量"].iloc[0])
    msku_x2_curr = int(curr.loc[curr["MSKU"] == "X2", "本月销量"].iloc[0])
    assert msku_x1_last == 8
    assert msku_x2_last == 11
    assert msku_x1_curr == 2
    assert msku_x2_curr == 9
    print("ok")

if __name__ == "__main__":
    main()

