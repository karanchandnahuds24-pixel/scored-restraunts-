# SkyCity Auckland: Restaurant Growth Readiness

## Run it
```
pip install -r requirements.txt
python pipeline.py          # builds scored_restaurants.csv + pipeline_meta.json (about 1 min)
streamlit run app.py        # launches the dashboard
```

## Files
- `features.py`  – KPI/feature engineering (all channel shares recomputed on a total-orders basis)
- `pipeline.py`  – preprocessing, PCA, K-Means/Ward/DBSCAN, bootstrap stability, GPI, weight sensitivity, recommendations, UMAP
- `app.py`       – Streamlit dashboard (5 tabs, sidebar filters)
- `scored_restaurants.csv` – one row per restaurant: cluster, GPI, 4 pillar scores, recommendation, drivers
- `pipeline_meta.json` – PCA loadings, k-selection table, stability and sensitivity results

## Method in one paragraph
15 ratio/rate features -> Yeo-Johnson power transform -> robust scaling -> PCA (6 PCs, 85% variance) -> K-Means k=4
(cross-checked with Ward hierarchical, bootstrap stability, DBSCAN outliers). GPI = 100 x equal-weight mean of four
percentile-ranked pillars (Growth, Cost resilience, Channel balance, Logistics). Recommendation = ordered rules on
blended margin, GPI and aggregator dependence. Categorical fields (cuisine, segment, subregion) are used to profile and filter, not to cluster.
