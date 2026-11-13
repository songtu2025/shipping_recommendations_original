import pandas as pd

def aggregate_last_n_months_sales(sales_df: pd.DataFrame, n: int = 2, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if now is None:
        now = pd.Timestamp.now()
    ref_period = pd.Timestamp(year=now.year, month=now.month, day=1).to_period("M")
    periods = [ref_period - i for i in range(1, n + 1)]
    df = sales_df.copy()
    df["purchaseDate"] = pd.to_datetime(df["purchaseDate"])
    df["purchase_month"] = df["purchaseDate"].dt.to_period("M")
    df = df.loc[df["purchase_month"].isin(periods)]
    ret = (
        df.groupby(["店铺-站点", "MSKU"], as_index=False)["销量"]
        .sum()
        .rename(columns={"销量": "上月销量"})
    )
    return ret

def aggregate_current_month_sales(sales_df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if now is None:
        now = pd.Timestamp.now()
    month_start = pd.Timestamp(year=now.year, month=now.month, day=1)
    df = sales_df.copy()
    df["purchaseDate"] = pd.to_datetime(df["purchaseDate"])
    df = df.loc[df["purchaseDate"] >= month_start]
    ret = (
        df.groupby(["店铺-站点", "MSKU"], as_index=False)["销量"]
        .sum()
        .rename(columns={"销量": "本月销量"})
    )
    return ret

