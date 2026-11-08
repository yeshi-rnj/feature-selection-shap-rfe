"""
Feature selection pipeline: SHAP vs built-in importance vs RFE.

Dataset : Breast Cancer Wisconsin (30 numeric features, many highly correlated),
          bundled with scikit-learn, so nothing needs to be downloaded.
Model   : Random Forest.

Run:
    pip install -r requirements.txt
    python feature_selection.py

What it produces
    *.png , *.csv                   plots and tables (same folder)
    README.md                       the block between the RESULTS markers is
                                    regenerated with real numbers and the plots
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# ------------------------------------------------------------------ config
SEED = 42
OUT = Path(".")
README = Path("README.md")
K_FIXED = 10            # size used for the fixed RFE / top-K SHAP comparisons
TOLERANCE = 0.005       # "competitive" = within 0.005 CV ROC-AUC of the baseline
TOP_N_PLOT = 15
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def make_model(n_estimators=300):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=SEED, n_jobs=-1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- data
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target                     # target: 0 = malignant, 1 = benign
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
features = list(X.columns)
print(f"{len(features)} features | train {len(X_train)} | test {len(X_test)}")


def evaluate(cols, label):
    """Fit on train, score on test, plus 5-fold CV AUC on train only."""
    model = make_model().fit(X_train[cols], y_train)
    proba = model.predict_proba(X_test[cols])[:, 1]
    pred = model.predict(X_test[cols])
    cv_auc = cross_val_score(make_model(150), X_train[cols], y_train, cv=cv, scoring="roc_auc").mean()
    return model, {
        "Model": label,
        "Features": len(cols),
        "Accuracy": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "CV ROC-AUC (train)": cv_auc,
    }


# ------------------------------------------------------------- 1. baseline
baseline, base_res = evaluate(features, "Baseline (all features)")
print("Baseline:", {k: round(v, 4) if isinstance(v, float) else v for k, v in base_res.items()})

# ------------------------------------------------------------------ 2. SHAP
explainer = shap.TreeExplainer(baseline)


def shap_positive_class(values):
    """Normalise shap's output across versions to an (n_samples, n_features) array."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


sv_test = shap_positive_class(explainer.shap_values(X_test))     # for the plots
sv_train = shap_positive_class(explainer.shap_values(X_train))   # for ranking / selection

plt.figure()
shap.summary_plot(sv_test, X_test, max_display=TOP_N_PLOT, show=False)
plt.title("SHAP summary (test set, class 1 = benign)")
save(plt.gcf(), "shap_summary_beeswarm.png")

plt.figure()
shap.summary_plot(sv_test, X_test, plot_type="bar", max_display=TOP_N_PLOT, show=False)
save(plt.gcf(), "shap_summary_bar.png")

shap_imp = pd.Series(np.abs(sv_train).mean(axis=0), index=features)

# Direction of effect: does a high feature value push toward benign (+) or malignant (-)?
direction = pd.Series(
    {f: np.corrcoef(X_test[f], sv_test[:, i])[0, 1] for i, f in enumerate(features)}
)

# ------------------------------------------------- 3. built-in importance
builtin_imp = pd.Series(baseline.feature_importances_, index=features)

# ------------------------------------------------------------------ 4. RFE
rfecv = RFECV(
    make_model(150), step=1, cv=cv, scoring="roc_auc",
    min_features_to_select=3, n_jobs=-1,
).fit(X_train, y_train)
rfecv_cols = [f for f, keep in zip(features, rfecv.support_) if keep]
print(f"RFECV picked {len(rfecv_cols)} features")

# n_features_to_select=1 gives a complete elimination order (rank 1 = survived longest)
rfe_order = RFE(make_model(150), n_features_to_select=1, step=1).fit(X_train, y_train)
rfe_rank = pd.Series(rfe_order.ranking_, index=features)
rfe_fixed_cols = list(rfe_rank.sort_values().head(K_FIXED).index)

scores = rfecv.cv_results_["mean_test_score"]
sizes = np.arange(rfecv.min_features_to_select, rfecv.min_features_to_select + len(scores))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sizes, scores, marker="o", ms=4)
ax.axvline(rfecv.n_features_, color="crimson", ls="--", label=f"RFECV optimum = {rfecv.n_features_}")
ax.set(xlabel="Number of features", ylabel="CV ROC-AUC", title="RFECV: performance vs number of features")
ax.legend()
save(fig, "rfecv_curve.png")

# ------------------------------------------------------ 5. rank comparison
ranks = pd.DataFrame({
    "SHAP": shap_imp.rank(ascending=False, method="first").astype(int),
    "Built-in": builtin_imp.rank(ascending=False, method="first").astype(int),
    "RFE": rfe_rank,
})
ranks["Consensus score"] = ranks[["SHAP", "Built-in", "RFE"]].mean(axis=1)
ranks = ranks.sort_values("Consensus score")
ranks["Consensus rank"] = np.arange(1, len(ranks) + 1)
ranks["RFECV keeps"] = [f in rfecv_cols for f in ranks.index]
ranks.to_csv(OUT / "feature_ranking_comparison.csv")

rho, _ = spearmanr(ranks["SHAP"], ranks["Built-in"])
top = {
    "SHAP": set(shap_imp.nlargest(K_FIXED).index),
    "Built-in": set(builtin_imp.nlargest(K_FIXED).index),
    "RFE": set(rfe_fixed_cols),
}
agree_all = sorted(top["SHAP"] & top["Built-in"] & top["RFE"])
overlap = {
    "SHAP vs Built-in": len(top["SHAP"] & top["Built-in"]),
    "SHAP vs RFE": len(top["SHAP"] & top["RFE"]),
    "Built-in vs RFE": len(top["Built-in"] & top["RFE"]),
}

# --------------------------------------- 6. choose the final production set
# Features are added in consensus order; the smallest k whose CV AUC (train only,
# so no test leakage) is within TOLERANCE of the baseline wins.
consensus_order = list(ranks.index)
curve = []
for k in range(3, len(features) + 1):
    auc = cross_val_score(make_model(150), X_train[consensus_order[:k]], y_train,
                          cv=cv, scoring="roc_auc").mean()
    curve.append((k, auc))
curve = pd.DataFrame(curve, columns=["k", "cv_auc"])
threshold = base_res["CV ROC-AUC (train)"] - TOLERANCE
best_k = int(curve.loc[curve["cv_auc"] >= threshold, "k"].min())
final_cols = consensus_order[:best_k]
curve.to_csv(OUT / "consensus_subset_curve.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve["k"], curve["cv_auc"], marker="o", ms=4, label="Top-k by consensus rank")
ax.axhline(base_res["CV ROC-AUC (train)"], color="grey", ls=":", label="All 30 features")
ax.axhline(threshold, color="grey", ls="--", lw=0.8, label=f"Baseline - {TOLERANCE}")
ax.axvline(best_k, color="crimson", ls="--", label=f"Chosen k = {best_k}")
ax.set(xlabel="Number of features (added in consensus order)", ylabel="CV ROC-AUC (train)",
       title="How few features are enough?")
ax.legend(fontsize=8)
save(fig, "consensus_subset_curve.png")

# ------------------------------------------------------ 7. retrain + compare
rows = [base_res]
for cols, label in [
    (rfecv_cols, f"RFECV subset ({len(rfecv_cols)})"),
    (rfe_fixed_cols, f"RFE top-{K_FIXED}"),
    (list(shap_imp.nlargest(K_FIXED).index), f"SHAP top-{K_FIXED}"),
    (final_cols, f"Consensus top-{best_k} (recommended)"),
]:
    rows.append(evaluate(cols, label)[1])
perf = pd.DataFrame(rows)
perf["Δ AUC vs baseline"] = perf["ROC-AUC"] - base_res["ROC-AUC"]
perf.to_csv(OUT / "performance_comparison.csv", index=False)
print(perf.round(4).to_string(index=False))

# ------------------------------------------------------------- 8. plots
# performance bars
fig, ax = plt.subplots(figsize=(9, 4.5))
metrics = ["Accuracy", "F1", "ROC-AUC"]
width = 0.8 / len(metrics)
for i, m in enumerate(metrics):
    ax.bar(np.arange(len(perf)) + i * width, perf[m], width, label=m)
ax.set_xticks(np.arange(len(perf)) + width)
ax.set_xticklabels([f"{m}\n({n} feat.)" for m, n in zip(perf["Model"], perf["Features"])], fontsize=7)
ax.set_ylim(max(0.0, perf[metrics].min().min() - 0.03), 1.005)
ax.set_title("Test performance: full feature set vs reduced subsets")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
save(fig, "performance_comparison.png")

# SHAP vs built-in (top 15, normalised)
top_feats = list(ranks.index[:TOP_N_PLOT])
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
(shap_imp[top_feats] / shap_imp.sum()).iloc[::-1].plot.barh(ax=axes[0], color="#1f77b4", title="SHAP (mean |value|, normalised)")
(builtin_imp[top_feats] / builtin_imp.sum()).iloc[::-1].plot.barh(ax=axes[1], color="#ff7f0e", title="Built-in importance (normalised)")
save(fig, "shap_vs_builtin.png")


def heatmap(df, name, title, cmap, fmt, vmin=None, vmax=None, figsize=(8, 6), annot_size=8):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(df.shape[0]), df.index, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, fmt.format(df.values[i, j]), ha="center", va="center", fontsize=annot_size)
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, name)


heatmap(ranks.loc[top_feats, ["SHAP", "Built-in", "RFE"]], "ranking_heatmap.png",
        "Feature rank by method (1 = most important)", "viridis_r", "{:.0f}",
        vmin=1, vmax=len(features), figsize=(5.5, 6.5), annot_size=9)
heatmap(X[top_feats].corr(), "correlation_heatmap.png",
        "Correlation between the top-15 features (why methods disagree)", "RdBu_r", "{:.2f}",
        vmin=-1, vmax=1, figsize=(9, 7.5), annot_size=6)

# ------------------------------------------------ 9. data-driven write-up
def md_table(df, floatfmt="{:.4f}"):
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = []
    for _, r in df.iterrows():
        body.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r) + " |")
    return "\n".join([head, sep] + body)


def bullets(items):
    return "\n".join(f"- {i}" for i in items) if items else "- none"


corr = X.corr().abs()
top5 = consensus_order[:5]
spread = ranks[["SHAP", "Built-in", "RFE"]].max(axis=1) - ranks[["SHAP", "Built-in", "RFE"]].min(axis=1)
disagree = []
for f in spread.sort_values(ascending=False).index[:5]:
    r = ranks.loc[f]
    partner = corr.loc[f, [p for p in top5 if p != f]].idxmax()
    note = f"; correlated r={corr.loc[f, partner]:.2f} with `{partner}`" if corr.loc[f, partner] > 0.85 else ""
    disagree.append(f"`{f}`: SHAP #{r['SHAP']}, built-in #{r['Built-in']}, RFE #{r['RFE']}{note}")

dir_rows = pd.DataFrame({
    "Feature": top_feats[:8],
    "SHAP rank": [int(ranks.loc[f, "SHAP"]) for f in top_feats[:8]],
    "Effect of a HIGH value": ["pushes toward benign" if direction[f] > 0 else "pushes toward malignant"
                               for f in top_feats[:8]],
})

rank_table = ranks.loc[top_feats, ["SHAP", "Built-in", "RFE", "Consensus rank", "RFECV keeps"]].reset_index(names="Feature")
rank_table["RFECV keeps"] = rank_table["RFECV keeps"].map({True: "yes", False: "no"})

perf_show = perf.copy()
perf_show["Δ AUC vs baseline"] = perf_show["Δ AUC vs baseline"].map(lambda v: f"{v:+.4f}")
rec = perf.iloc[-1]
dropped = [f for f in features if f not in final_cols]

results = f"""{START}
<!-- This block is regenerated by feature_selection.py. Edit text outside the markers. -->

## Results

**Headline:** the recommended subset uses **{best_k} of {len(features)} features** and reaches test ROC-AUC
**{rec['ROC-AUC']:.4f}** vs **{base_res['ROC-AUC']:.4f}** for the full model
(Δ {rec['Δ AUC vs baseline']:+.4f}), with CV ROC-AUC within {TOLERANCE} of the baseline.

### 1. Baseline vs reduced models

{md_table(perf_show)}

![Performance comparison](performance_comparison.png)

### 2. SHAP: what matters and in which direction

![SHAP beeswarm](shap_summary_beeswarm.png)

Each dot is one test sample; colour is the feature value, x-position is its push on the prediction
(class 1 = benign). Direction for the top features:

{md_table(dir_rows)}

![SHAP bar](shap_summary_bar.png)

### 3. SHAP vs built-in importance

Spearman correlation between the two full rankings: **{rho:.3f}**.

![SHAP vs built-in](shap_vs_builtin.png)

### 4. Recursive Feature Elimination

RFECV selected **{len(rfecv_cols)}** features by cross-validated ROC-AUC.

![RFECV curve](rfecv_curve.png)

### 5. Rankings side by side

{md_table(rank_table)}

![Ranking heatmap](ranking_heatmap.png)

Full table: [`feature_ranking_comparison.csv`](feature_ranking_comparison.csv)

### 6. Where the methods agree and disagree

**Top-{K_FIXED} overlap:** SHAP/Built-in {overlap['SHAP vs Built-in']}/{K_FIXED},
SHAP/RFE {overlap['SHAP vs RFE']}/{K_FIXED}, Built-in/RFE {overlap['Built-in vs RFE']}/{K_FIXED}.

**Agreement:** features in the top {K_FIXED} of all three methods:
{', '.join(f'`{f}`' for f in agree_all) if agree_all else 'none'}.

**Largest disagreements (rank spread across methods):**

{bullets(disagree)}

![Correlation heatmap](correlation_heatmap.png)

The heatmap shows why. Many top features are near-duplicates (radius, perimeter and area measure
the same thing). When features carry the same signal, each method splits the credit differently:
impurity-based importance spreads it unpredictably across the correlated group, SHAP spreads it
according to how the trained trees actually used each feature, and RFE's result depends on which
member of the group happens to be eliminated first. Disagreement here is mostly redundancy, not
a sign that one method is wrong.

### 7. Final feature set

Features are added in consensus order (mean of the three ranks); the smallest set whose
cross-validated AUC stays within {TOLERANCE} of the full model is kept. Selection uses training data only.

![Subset size curve](consensus_subset_curve.png)

**Keep ({best_k}):** {', '.join(f'`{f}`' for f in final_cols)}

**Drop ({len(dropped)}):** {', '.join(f'`{f}`' for f in dropped)}

Why this set: all three methods rank these features highly, the set performs about as well as all
{len(features)} features, and it is cheaper to collect, faster to serve and easier to explain.

### Limitations
- The test set has {len(y_test)} samples, so differences of a few thousandths in AUC are within noise.
- SHAP and built-in rankings come from a single Random Forest; other model families may rank differently.
- Correlated features make "importance" ambiguous. A different but equivalent subset would likely perform the same.
{END}"""

header = f"""# Feature Selection: SHAP vs Built-in Importance vs RFE

Identifies the most predictive features in the Breast Cancer Wisconsin dataset ({len(features)} features)
using three methods, then verifies that a reduced feature set performs as well as the full one.

## How to run
```bash
pip install -r requirements.txt
python feature_selection.py
```
Plots and tables are written to this same folder, and the Results section below is regenerated.

## Pipeline
1. Baseline Random Forest on all features
2. SHAP values (`TreeExplainer`) with summary plots
3. Built-in `feature_importances_` compared against SHAP
4. `RFECV` for the optimal subset size, plus a full RFE elimination order
5. Retrain on each subset and compare with the baseline
6. Consensus ranking across the three methods and a final recommended feature set

"""

if README.exists() and START in README.read_text() and END in README.read_text():
    text = README.read_text()
    new = text[: text.index(START)] + results + text[text.index(END) + len(END):]
else:
    new = header + results + "\n"
README.write_text(new)
print(f"\nDone. plots, tables and README.md updated in this folder.")
