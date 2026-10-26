import json, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="SkyCity Growth Readiness", layout="wide", page_icon="📈")

@st.cache_data
def load():
    df = pd.read_csv("scored_restaurants.csv")
    meta = json.load(open("pipeline_meta.json"))
    pr = lambda s: s.rank(pct=True)
    df["r_Growth rate"] = pr(df.GrowthFactor); df["r_Scale"] = pr(df.kpi_scale); df["r_AOV"] = pr(df.AOV)
    df["r_Cost discipline"] = pr(-df.cost_load); df["r_Net margin"] = pr(df.blended_margin)
    df["r_Channel balance"] = pr(df.channel_balance); df["r_Low aggregator dep."] = pr(-df.agg_dependence)
    df["r_Delivery reach"] = pr(df.DeliveryRadiusKM); df["r_Headroom"] = pr(df.log_headroom); df["r_Delivery efficiency"] = pr(-df.sd_cost_ratio)
    return df, meta
df, meta = load()
SPOKES = [c for c in df.columns if c.startswith("r_")]
PILLARS = ["pillar_Growth", "pillar_Cost", "pillar_Channel", "pillar_Logistics"]
REC_COL = {"Optimize & Expand": "#2a9d8f", "Rebalance Channels": "#e9c46a", "Hold / Stabilize": "#e76f51"}

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
sel = lambda label, col: st.sidebar.multiselect(label, sorted(df[col].unique()), default=sorted(df[col].unique()))
subs, cuis, segs = sel("Subregion", "Subregion"), sel("Cuisine", "CuisineType"), sel("Segment", "Segment")
f = df[df.Subregion.isin(subs) & df.CuisineType.isin(cuis) & df.Segment.isin(segs)]
st.sidebar.metric("Restaurants in view", f"{len(f):,} / {len(df):,}")
if f.empty: st.warning("No restaurants match these filters."); st.stop()

st.title("SkyCity Auckland: Restaurant Growth Readiness")
c = st.columns(4)
c[0].metric("Avg GPI", f"{f.GPI.mean():.1f}"); c[1].metric("Avg net margin", f"{f.blended_margin.mean():.1%}")
c[2].metric("Avg aggregator dependence", f"{f.agg_dependence.mean():.0%}")
c[3].metric("Ready to Optimize & Expand", f"{(f.recommendation=='Optimize & Expand').mean():.0%}")

tabs = st.tabs(["Cluster map", "GPI scorecards", "Driver radar", "Strategy panel", "Cluster comparison"])

# ---------------- Tab 1 ----------------
with tabs[0]:
    proj = st.radio("Projection", ["UMAP", "PCA (PC1 vs PC2)"], horizontal=True)
    x, y = ("umap_x", "umap_y") if proj == "UMAP" else ("PC1", "PC2")
    colour = st.radio("Colour by", ["cluster", "recommendation", "Segment", "Subregion"], horizontal=True)
    fig = px.scatter(f, x=x, y=y, color=colour, hover_data=["RestaurantID", "RestaurantName", "CuisineType", "GPI", "blended_margin"],
                     opacity=.75, height=560, color_discrete_map=REC_COL if colour == "recommendation" else None)
    st.plotly_chart(fig, width="stretch")
    st.caption("Projections are for visualisation. Clusters were fitted on the first 6 principal components. "
               "Silhouette is about 0.20, so archetype boundaries are soft: restaurants sit on a continuum.")
    with st.expander("What do the axes mean? (PCA loadings)"):
        L = pd.DataFrame(meta["loadings"]).iloc[:, :5]
        st.dataframe(L.style.background_gradient(cmap="RdBu", axis=None, vmin=-.7, vmax=.7).format("{:.2f}"), width="stretch")
        st.markdown("**PC1** cost pressure vs in-store orientation · **PC2** delivery reach / logistics · **PC3** channel leverage (self-delivery) · **PC4** growth momentum (growth, AOV) · **PC5** commission and margin dispersion")

# ---------------- Tab 2 ----------------
with tabs[1]:
    top = f.sort_values("GPI", ascending=False)
    cols = ["RestaurantID", "RestaurantName", "CuisineType", "Segment", "Subregion", "cluster", "GPI", "recommendation", "blended_margin", "agg_dependence", "GrowthFactor"]
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Ranked scorecard")
        st.dataframe(top[cols].rename(columns={"blended_margin": "net margin", "agg_dependence": "aggregator dep."}).style
                     .format({"GPI": "{:.1f}", "net margin": "{:.1%}", "aggregator dep.": "{:.0%}", "GrowthFactor": "{:.2f}"})
                     .background_gradient(subset=["GPI"], cmap="Greens"), height=520, width="stretch", hide_index=True)
        st.download_button("Download filtered scorecard (CSV)", top[cols].to_csv(index=False), "gpi_scorecard.csv")
    with right:
        st.subheader("GPI distribution")
        st.plotly_chart(px.histogram(f, x="GPI", color="recommendation", nbins=30, color_discrete_map=REC_COL, height=250), width="stretch")
        st.subheader("Average pillar scores")
        pv = f[PILLARS].mean().rename(lambda s: s.replace("pillar_", ""))
        st.plotly_chart(px.bar(pv, labels={"value": "score (0-1)", "index": ""}, height=250).update_layout(showlegend=False), width="stretch")

# ---------------- Tab 3 ----------------
with tabs[2]:
    st.subheader("Feature contribution radar")
    opts = f.sort_values("GPI", ascending=False).apply(lambda r: f"{r.RestaurantID} · {r.RestaurantName} ({r.Subregion})", axis=1)
    pick = st.selectbox("Restaurant", opts.tolist())
    rid = int(pick.split(" · ")[0]); r = df[df.RestaurantID == rid].iloc[0]
    cl_avg = df[df.cluster == r.cluster][SPOKES].mean()
    names = [s[2:] for s in SPOKES]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r[SPOKES].astype(float).tolist() + [r[SPOKES[0]]], theta=names + [names[0]], fill="toself", name="This restaurant"))
    fig.add_trace(go.Scatterpolar(r=cl_avg.tolist() + [cl_avg.iloc[0]], theta=names + [names[0]], name=f"Cluster avg: {r.cluster}"))
    fig.add_trace(go.Scatterpolar(r=[.5] * (len(names) + 1), theta=names + [names[0]], name="Network median", line=dict(dash="dot", color="grey")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=520)
    a, b = st.columns([3, 2]); a.plotly_chart(fig, width="stretch")
    with b:
        st.metric("GPI", f"{r.GPI:.1f}"); st.write(f"**Cluster:** {r.cluster}"); st.write(f"**Recommendation:** {r.recommendation}")
        st.caption("Each spoke is the restaurant's percentile rank across the whole network (1 = best). Cost and dependence spokes are inverted so outward is always better.")

# ---------------- Tab 4 ----------------
with tabs[3]:
    st.subheader("Strategy recommendations")
    cnt = f.recommendation.value_counts()
    cols3 = st.columns(3)
    PLAY = {"Optimize & Expand": "Top-40% GPI with healthy margins. Prioritise for investment: extend delivery reach, grow high-margin channels, replicate the model.",
            "Rebalance Channels": "Healthy margins but above-median aggregator dependence and mid/low GPI. Shift orders to in-store and self-delivery, renegotiate commission.",
            "Hold / Stabilize": "Net margin below 5%, or low GPI without a clear channel lever. Fix unit economics (COGS, OPEX, menu pricing) before any expansion spend."}
    for col, k in zip(cols3, REC_COL):
        col.markdown(f"#### {k}"); col.metric("Restaurants", int(cnt.get(k, 0))); col.caption(PLAY[k])
    st.divider()
    st.markdown("**Restaurant drill-down: why this recommendation?**")
    sel_r = st.selectbox("Restaurant ", opts.tolist(), key="strat")
    r2 = df[df.RestaurantID == int(sel_r.split(" · ")[0])].iloc[0]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("GPI", f"{r2.GPI:.1f}"); k2.metric("Net margin", f"{r2.blended_margin:.1%}"); k3.metric("Aggregator dep.", f"{r2.agg_dependence:.0%}")
    k4.metric("Commission drag", f"{r2.commission_drag:.1%}"); k5.metric("Self-delivery cost / rev", f"{r2.sd_cost_ratio:.1%}")
    st.success(f"**{r2.recommendation}** · cluster: {r2.cluster} · {r2.drivers}")
    ch = pd.DataFrame({"channel": ["In-store", "Uber Eats", "DoorDash", "Self-delivery"],
                       "margin": [r2.margin_instore, r2.margin_ue, r2.margin_dd, r2.margin_sd],
                       "order share": [r2.share_instore, r2.share_ue, r2.share_dd, r2.share_sd]})
    g1, g2 = st.columns(2)
    g1.plotly_chart(px.bar(ch, x="channel", y="margin", title="Net margin by channel", height=300).update_yaxes(tickformat=".0%"), width="stretch")
    g2.plotly_chart(px.pie(ch, names="channel", values="order share", title="Order mix", height=300), width="stretch")

# ---------------- Tab 5 ----------------
with tabs[4]:
    st.subheader("Cluster comparison")
    agg = f.groupby("cluster").agg(Restaurants=("GPI", "size"), GPI=("GPI", "mean"), Net_margin=("blended_margin", "mean"), Agg_dependence=("agg_dependence", "mean"),
                                   Self_delivery_share=("share_sd", "mean"), In_store_share=("share_instore", "mean"), Radius_km=("DeliveryRadiusKM", "mean"),
                                   Growth_factor=("GrowthFactor", "mean"), AOV=("AOV", "mean"), Cost_load=("cost_load", "mean"), Commission_drag=("commission_drag", "mean"),
                                   Monthly_orders=("MonthlyOrders", "mean"))
    st.dataframe(agg.style.format({"GPI": "{:.1f}", "Net_margin": "{:.1%}", "Agg_dependence": "{:.0%}", "Self_delivery_share": "{:.0%}", "In_store_share": "{:.0%}",
                                   "Radius_km": "{:.1f}", "Growth_factor": "{:.3f}", "AOV": "${:.2f}", "Cost_load": "{:.0%}", "Commission_drag": "{:.1%}", "Monthly_orders": "{:,.0f}"})
                 .background_gradient(cmap="RdYlGn", subset=["GPI", "Net_margin"]), width="stretch")
    pv = f.groupby("cluster")[PILLARS].mean().reset_index().melt("cluster", var_name="pillar", value_name="score")
    pv["pillar"] = pv.pillar.str.replace("pillar_", "")
    st.plotly_chart(px.bar(pv, x="pillar", y="score", color="cluster", barmode="group", height=380), width="stretch")
    mix = pd.crosstab(f.cluster, f.Segment, normalize="index")
    st.plotly_chart(px.bar(mix, barmode="stack", title="Segment composition of each cluster", height=320).update_yaxes(tickformat=".0%"), width="stretch")
    with st.expander("Methodology & robustness"):
        gs = meta["gpi_sensitivity"]
        st.markdown(f"- Features: ratios and rates only (15), power-transformed, robust-scaled, PCA to {meta['n_pcs']} components (85% variance)\n"
                    f"- K-Means k=4 chosen by lowest Davies-Bouldin; agreement with Ward hierarchical ARI = {meta['ARI_kmeans_vs_ward_k4']}\n"
                    f"- Bootstrap stability (80% subsamples): mean ARI {meta['bootstrap_ARI_mean']}, min {meta['bootstrap_ARI_min']}\n"
                    f"- DBSCAN flags only {meta['dbscan_outliers']} outliers, so the data has no meaningful outlier tail\n"
                    f"- GPI = equal-weight mean of four percentile-ranked pillars. Under 1,000 random weightings, rank correlation with baseline averages {gs['spearman_mean']} "
                    f"(worst {gs['spearman_min']}); top-10% overlap averages {gs['top10_overlap_mean']:.0%}")
        st.dataframe(pd.DataFrame(meta["k_selection"]), hide_index=True)
