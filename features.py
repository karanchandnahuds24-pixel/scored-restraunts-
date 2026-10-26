import numpy as np, pandas as pd

RAW = "SkyCity_Auckland_Restaurants___Bars.csv"

def load_and_engineer(path=RAW):
    df = pd.read_csv(path)
    n = df.MonthlyOrders
    # --- Consistent channel shares (all as share of TOTAL orders) ---
    df["share_instore"] = df.InStoreOrders / n
    df["share_ue"] = df.UberEatsOrders / n
    df["share_dd"] = df.DoorDashOrders / n
    df["share_sd"] = df.SelfDeliveryOrders / n
    df["agg_dependence"] = df.share_ue + df.share_dd
    df["channel_balance"] = 1 - (df[["share_instore","share_ue","share_dd","share_sd"]]**2).sum(axis=1)

    # --- Revenue / profit totals and margins ---
    df["total_revenue"] = df[["InStoreRevenue","UberEatsRevenue","DoorDashRevenue","SelfDeliveryRevenue"]].sum(axis=1)
    df["total_net_profit"] = df[["InStoreNetProfit","UberEatsNetProfit","DoorDashNetProfit","SelfDeliveryNetProfit"]].sum(axis=1)
    df["blended_margin"] = df.total_net_profit / df.total_revenue
    for ch, r, p in [("instore","InStoreRevenue","InStoreNetProfit"),("ue","UberEatsRevenue","UberEatsNetProfit"),
                     ("dd","DoorDashRevenue","DoorDashNetProfit"),("sd","SelfDeliveryRevenue","SelfDeliveryNetProfit")]:
        df[f"margin_{ch}"] = df[p] / df[r]
    df["margin_spread"] = df[["margin_instore","margin_ue","margin_dd","margin_sd"]].std(axis=1)
    df["agg_revenue_share"] = (df.UberEatsRevenue + df.DoorDashRevenue) / df.total_revenue
    df["commission_drag"] = (df.UberEatsRevenue + df.DoorDashRevenue) * df.CommissionRate / df.total_revenue
    df["sd_cost_ratio"] = df.SD_DeliveryTotalCost / df.SelfDeliveryRevenue
    df["cost_load"] = df.COGSRate + df.OPEXRate

    # --- KPIs from the brief ---
    df["kpi_scale"] = n * df.GrowthFactor
    df["kpi_cost_discipline"] = df.cost_load  # lower is better; also see cost_vs_segment
    df["cost_vs_segment"] = df.cost_load - df.groupby("Segment").cost_load.transform("median")
    df["kpi_agg_dependence"] = df.agg_dependence
    df["demand_density"] = n / (np.pi * df.DeliveryRadiusKM**2)          # orders per km2 of reach
    df["kpi_headroom"] = 1 / df.demand_density                           # km2 of reach per order
    df["kpi_revenue_quality"] = df.AOV * df.blended_margin
    df["log_scale"] = np.log(df.kpi_scale)
    df["log_headroom"] = np.log(df.kpi_headroom)
    df["growth_pct"] = (df.GrowthFactor - 1) * 100
    return df

# Features used for PCA / clustering (ratios & rates only -> no raw-size collinearity)
CLUSTER_FEATURES = [
    "GrowthFactor","AOV","log_scale",
    "COGSRate","OPEXRate","CommissionRate","blended_margin","margin_spread",
    "share_instore","agg_dependence","share_sd","channel_balance",
    "DeliveryRadiusKM","log_headroom","sd_cost_ratio",
]
