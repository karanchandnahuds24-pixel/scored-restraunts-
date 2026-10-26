"""SkyCity restaurant growth-readiness pipeline: features -> PCA -> clustering -> GPI -> recommendations."""
import json, numpy as np, pandas as pd, umap
from scipy.stats import spearmanr
from features import load_and_engineer, CLUSTER_FEATURES, RAW
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

SEED, K = 42, 4
df = load_and_engineer(RAW)
meta = {}

# ---------- Preprocess ----------
Z = RobustScaler().fit_transform(PowerTransformer().fit_transform(df[CLUSTER_FEATURES]))
pca_full = PCA(random_state=SEED).fit(Z)
pca = PCA(n_components=0.85, random_state=SEED).fit(Z)
P = pca.transform(Z)
meta["n_pcs"] = int(P.shape[1])
meta["explained_variance"] = pca.explained_variance_ratio_.round(4).tolist()
meta["loadings"] = pd.DataFrame(pca.components_.T, index=CLUSTER_FEATURES,
                                columns=[f"PC{i+1}" for i in range(P.shape[1])]).round(3).to_dict()
for i in range(P.shape[1]): df[f"PC{i+1}"] = P[:, i]

# ---------- Model selection ----------
sel = []
for k in range(2, 9):
    lab = KMeans(k, n_init=30, random_state=SEED).fit_predict(P)
    hc = AgglomerativeClustering(k, linkage="ward").fit_predict(P)
    sel.append(dict(k=k, silhouette=round(silhouette_score(P, lab), 3),
                    davies_bouldin=round(davies_bouldin_score(P, lab), 3),
                    ARI_kmeans_vs_ward=round(adjusted_rand_score(lab, hc), 3)))
meta["k_selection"] = sel

km = KMeans(K, n_init=50, random_state=SEED).fit(P)
df["cluster_id"] = km.labels_
hc4 = AgglomerativeClustering(K, linkage="ward").fit_predict(P)
meta["ARI_kmeans_vs_ward_k4"] = round(adjusted_rand_score(km.labels_, hc4), 3)

# Bootstrap stability (80% subsamples, ARI on the points in the subsample)
rng = np.random.default_rng(SEED); aris = []
for _ in range(50):
    idx = rng.choice(len(P), int(.8 * len(P)), replace=False)
    sub = KMeans(K, n_init=10, random_state=int(rng.integers(1e6))).fit(P[idx])
    aris.append(adjusted_rand_score(km.labels_[idx], sub.predict(P[idx])))
meta["bootstrap_ARI_mean"], meta["bootstrap_ARI_min"] = round(float(np.mean(aris)), 3), round(float(np.min(aris)), 3)

# DBSCAN as outlier detector (eps from k-distance 96th percentile)
d, _ = NearestNeighbors(n_neighbors=10).fit(P).kneighbors(P)
eps = float(np.percentile(d[:, -1], 96))
db = DBSCAN(eps=eps, min_samples=10).fit(P)
df["is_outlier"] = db.labels_ == -1
meta["dbscan_eps"], meta["dbscan_outliers"] = round(eps, 3), int(df.is_outlier.sum())

# ---------- Archetype labelling (rule-based on centroids, reproducible) ----------
prof = df.groupby("cluster_id")[["blended_margin", "agg_dependence", "DeliveryRadiusKM"]].mean()
names = {}
names[prof.blended_margin.idxmin()] = "Overextended, Low Return"
rest = prof.drop(prof.blended_margin.idxmin())
names[rest.agg_dependence.idxmin()] = "Scalable Self-Delivery Leaders"
rest = rest.drop(rest.agg_dependence.idxmin())
names[rest.DeliveryRadiusKM.idxmax()] = "Aggregator-Dependent, Over-Stretched Reach"
names[rest.DeliveryRadiusKM.idxmin()] = "Stable Local Performers"
df["cluster"] = df.cluster_id.map(names)

# ---------- Growth Potential Index ----------
pr = lambda s: s.rank(pct=True)
pillars = pd.DataFrame({
    "Growth":    (pr(df.GrowthFactor) + pr(df.kpi_scale) + pr(df.AOV)) / 3,
    "Cost":      (pr(-df.cost_load) + pr(df.blended_margin) + pr(df.kpi_revenue_quality)) / 3,
    "Channel":   (pr(df.channel_balance) + pr(-df.agg_dependence) + pr(-df.commission_drag)) / 3,
    "Logistics": (pr(df.DeliveryRadiusKM) + pr(df.log_headroom) + pr(-df.sd_cost_ratio)) / 3,
})
for c in pillars: df[f"pillar_{c}"] = pillars[c]
W = np.array([.25] * 4)
df["GPI"] = 100 * pillars.values @ W

# Weight sensitivity: 1000 random Dirichlet weight vectors
base_rank = df.GPI.rank()
base_top = set(df.GPI.nlargest(int(.1 * len(df))).index)
rhos, overlaps = [], []
for w in np.random.default_rng(SEED).dirichlet(np.ones(4), 1000):
    g = pillars.values @ w
    rhos.append(spearmanr(base_rank, g).statistic)
    overlaps.append(len(base_top & set(pd.Series(g, index=df.index).nlargest(len(base_top)).index)) / len(base_top))
meta["gpi_sensitivity"] = dict(spearman_mean=round(float(np.mean(rhos)), 3), spearman_min=round(float(np.min(rhos)), 3),
                               top10_overlap_mean=round(float(np.mean(overlaps)), 3), top10_overlap_min=round(float(np.min(overlaps)), 3))

# ---------- Recommendation rules ----------
# Ordered, transparent rules (first match wins):
#  1. Hold / Stabilize : blended margin < 5%  -> cannot self-fund growth; fix unit economics first
#  2. Optimize & Expand: GPI in top 40% of the network
#  3. Rebalance Channels: healthy margin, mid/low GPI, aggregator dependence at/above network median
#  4. Hold / Stabilize : healthy margin, low GPI, no dominant aggregator issue -> protect, monitor
MARGIN_FLOOR = 0.05
gpi_hi = df.GPI.quantile(.60)
agg_med = df.agg_dependence.median()
def recommend(r):
    if r.blended_margin < MARGIN_FLOOR: return "Hold / Stabilize"
    if r.GPI >= gpi_hi: return "Optimize & Expand"
    if r.agg_dependence >= agg_med: return "Rebalance Channels"
    return "Hold / Stabilize"
df["recommendation"] = df.apply(recommend, axis=1)

def driver(r):
    pill = r[["pillar_Growth", "pillar_Cost", "pillar_Channel", "pillar_Logistics"]]
    pill.index = [i.replace("pillar_", "") for i in pill.index]
    return f"Strongest: {pill.idxmax()}; weakest: {pill.idxmin()}"
df["drivers"] = df.apply(driver, axis=1)

# ---------- UMAP for the visual map only ----------
um = umap.UMAP(n_neighbors=30, min_dist=.3, random_state=SEED).fit_transform(P)
df["umap_x"], df["umap_y"] = um[:, 0], um[:, 1]

df.to_csv("scored_restaurants.csv", index=False)
json.dump(meta, open("pipeline_meta.json", "w"), indent=1)

print(json.dumps({k: v for k, v in meta.items() if k != "loadings"}, indent=1))
print(df.groupby("cluster").agg(n=("GPI", "size"), GPI=("GPI", "mean"), margin=("blended_margin", "mean")).round(3))
print(pd.crosstab(df.cluster, df.recommendation, margins=True))
print(df.recommendation.value_counts(normalize=True).round(3))
