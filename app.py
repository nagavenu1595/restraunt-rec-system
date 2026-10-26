"""
Zomato Bangalore Restaurant Recommender  (Business Analytics DA)
Run:  streamlit run app.py
Needs zomato.csv in the same folder (or give the path in the sidebar).
"""
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import kagglehub
import glob

st.set_page_config(page_title="Zomato Restaurant Recommender", page_icon="🍽️", layout="wide")

COST_COL = "approx_cost(for two people)"
USE_COLS = ["name", "online_order", "book_table", "rate", "votes", "location",
            "rest_type", "cuisines", COST_COL]


# ----------------------------------------------------------------------------
# 1. DATA LOADING + CLEANING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading dataset from Kaggle (first run only)...")
def get_dataset_path():
    folder = kagglehub.dataset_download("himanshupoddar/zomato-bangalore-restaurants")
    csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}")
    csvs.sort(key=lambda p: (os.path.basename(p).lower() != "zomato.csv", p))
    return csvs[0]


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(path):
    # Read only the header first and check every column we need actually exists
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing_cols = [c for c in USE_COLS if c not in header]
    if missing_cols:
        raise KeyError(f"Missing columns {missing_cols}. Columns found in file: {header}")
    # usecols skips the huge reviews_list / menu_item columns -> fast, low memory
    raw = pd.read_csv(path, usecols=USE_COLS)
    log = {"raw_rows": len(raw)}

    df = raw.drop_duplicates().copy()
    log["after_dedup"] = len(df)

    # rate: "4.1/5", "3.8 /5", "NEW", "-", "0"  ->  float or NaN
    df["rate"] = pd.to_numeric(
        df["rate"].astype(str).str.extract(r"(\d+\.?\d*)\s*/\s*5")[0], errors="coerce")
    df["cost"] = pd.to_numeric(
        df[COST_COL].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

    df = df.dropna(subset=["rate", "cost", "votes", "location", "rest_type", "cuisines"])
    log["after_missing"] = len(df)

    # The same restaurant appears once per listing category -> keep one row per
    # (name, location) so tests do not count a restaurant several times.
    df = df.drop_duplicates(subset=["name", "location"]).copy()
    log["final_rows"] = len(df)

    df["main_cuisine"] = df["cuisines"].str.split(",").str[0].str.strip()
    df["main_type"] = df["rest_type"].str.split(",").str[0].str.strip()
    df["log_votes"] = np.log1p(df["votes"])
    df["online_bin"] = (df["online_order"] == "Yes").astype(int)
    df["book_bin"] = (df["book_table"] == "Yes").astype(int)

    # Bayesian (IMDB-style) weighted rating to handle "4.5 from 20 votes"
    C = df["rate"].mean()
    m = df["votes"].quantile(0.60)
    df["adj_rate"] = (df["votes"] / (df["votes"] + m)) * df["rate"] + \
                     (m / (df["votes"] + m)) * C
    log["C"], log["m"] = C, m
    return df.reset_index(drop=True), log


def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("🍽️ Zomato Recommender")
try:
    path = get_dataset_path()
    df, log = load_data(path)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

st.sidebar.success(f"{log['final_rows']:,} unique restaurants loaded")
st.sidebar.caption(f"Source: {path}")

st.title("Bangalore Restaurant Recommendation System")
st.caption("Statistically adjusted ratings + hypothesis tests + segmentation, built on the Zomato Bangalore dataset")

tab_rec, tab_eda, tab_stats, tab_seg, tab_about = st.tabs(
    ["🎯 Recommender", "📊 EDA", "🧪 Statistical Tests", "🧩 Segments (K-means)", "📝 Method & Cleaning"])

# ----------------------------------------------------------------------------
# TAB 1: RECOMMENDER
# ----------------------------------------------------------------------------
with tab_rec:
    st.subheader("Tell us what you want")
    c1, c2, c3 = st.columns(3)
    with c1:
        budget = st.slider("Max budget for two (₹)", 100, int(df["cost"].quantile(0.99)), 800, step=50)
        top_n = st.slider("How many recommendations", 3, 15, 5)
    with c2:
        loc_opts = sorted(df["location"].unique())
        locations = st.multiselect("Area(s) (blank = anywhere)", loc_opts)
        top_cuisines = df["main_cuisine"].value_counts().head(40).index.tolist()
        cuisines = st.multiselect("Cuisine(s) (blank = any)", sorted(top_cuisines))
    with c3:
        type_opts = sorted(df["main_type"].value_counts().head(15).index)
        types = st.multiselect("Restaurant type (blank = any)", type_opts)
        online = st.checkbox("Must offer online ordering")
        book = st.checkbox("Must offer table booking")
        min_votes = st.number_input("Minimum votes (reliability filter)", 0, 5000, 20, step=10)

    st.markdown("**What matters to you?** (weights are normalised automatically)")
    w1, w2, w3 = st.columns(3)
    wr = w1.slider("Quality (adjusted rating)", 0, 10, 6)
    wp = w2.slider("Popularity (votes)", 0, 10, 2)
    wa = w3.slider("Affordability", 0, 10, 2)

    d = df[df["cost"] <= budget]
    d = d[d["votes"] >= min_votes]
    if locations:
        d = d[d["location"].isin(locations)]
    if cuisines:
        pat = "|".join(re.escape(c) for c in cuisines)
        d = d[d["cuisines"].str.contains(pat, case=False, na=False)]
    if types:
        d = d[d["main_type"].isin(types)]
    if online:
        d = d[d["online_order"] == "Yes"]
    if book:
        d = d[d["book_table"] == "Yes"]

    if d.empty:
        st.warning("No restaurants match these filters. Try a higher budget, fewer filters or lower minimum votes.")
    else:
        tot = max(wr + wp + wa, 1)
        d = d.copy()
        d["score"] = (wr * minmax(d["adj_rate"]) +
                      wp * minmax(d["log_votes"]) +
                      wa * minmax(-d["cost"])) / tot
        res = d.sort_values("score", ascending=False).head(top_n)

        st.success(f"{len(d):,} restaurants match. Top {len(res)} for you:")
        show = res[["name", "location", "main_type", "cuisines", "rate", "votes",
                    "adj_rate", "cost", "online_order", "book_table", "score"]].copy()
        show.columns = ["Restaurant", "Area", "Type", "Cuisines", "Raw rating", "Votes",
                        "Adjusted rating", "Cost for two (₹)", "Online", "Table booking", "Score"]
        show[["Raw rating", "Adjusted rating", "Score"]] = show[["Raw rating", "Adjusted rating", "Score"]].round(3)
        show.index = range(1, len(show) + 1)
        st.dataframe(show, use_container_width=True)

        fig = px.bar(res.sort_values("score"), x="score", y="name", orientation="h",
                     color="cost", color_continuous_scale="Viridis",
                     labels={"score": "Recommendation score", "name": "", "cost": "Cost (₹)"})
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Why adjusted rating? Raw-rating top picks vs our picks"):
            raw_top = d.sort_values("rate", ascending=False).head(top_n)[["name", "rate", "votes"]]
            raw_top.columns = ["Restaurant", "Raw rating", "Votes"]
            st.write("**Ranked by raw rating only** (note how few votes some have):")
            st.dataframe(raw_top.reset_index(drop=True), use_container_width=True)
            st.caption("Adjusted rating = (v/(v+m))·R + (m/(v+m))·C, where R = restaurant rating, v = its votes, "
                       f"m = {log['m']:.0f} (60th percentile of votes), C = {log['C']:.2f} (mean rating). "
                       "Restaurants with few votes are pulled towards the overall mean.")

# ----------------------------------------------------------------------------
# TAB 2: EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Descriptive statistics")
    desc = df[["rate", "votes", "cost", "adj_rate"]].describe().T
    desc["skewness"] = df[["rate", "votes", "cost", "adj_rate"]].skew()
    desc["median"] = df[["rate", "votes", "cost", "adj_rate"]].median()
    st.dataframe(desc.round(2), use_container_width=True)

    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="rate", nbins=30, title="Distribution of ratings"),
                   use_container_width=True)
    b.plotly_chart(px.histogram(df[df["cost"] <= df["cost"].quantile(0.99)], x="cost", nbins=40,
                                title="Distribution of cost for two (99th pct cut)"),
                   use_container_width=True)

    top_t = df["main_type"].value_counts().head(8).index
    c, e = st.columns(2)
    c.plotly_chart(px.box(df[df["main_type"].isin(top_t)], x="main_type", y="rate",
                          title="Rating by restaurant type"), use_container_width=True)
    e.plotly_chart(px.bar(df["location"].value_counts().head(15).reset_index(),
                          x="location", y="count", title="Top 15 areas by number of restaurants"),
                   use_container_width=True)

    st.subheader("Correlation (Spearman)")
    corr = df[["rate", "log_votes", "cost", "online_bin", "book_bin"]].corr(method="spearman")
    st.plotly_chart(px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: STATISTICAL TESTS
# ----------------------------------------------------------------------------
with tab_stats:
    alpha = 0.05
    st.caption(f"All tests at α = {alpha}")

    # --- Test 1: t-test
    st.markdown("### 1. Does online ordering relate to rating? (Independent t-test)")
    st.write("**H0:** mean rating is the same with and without online ordering. **H1:** it differs.")
    g1 = df.loc[df["online_order"] == "Yes", "rate"]
    g0 = df.loc[df["online_order"] == "No", "rate"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    u, pu = stats.mannwhitneyu(g1, g0)
    lev = stats.levene(g1, g0)
    sh = stats.shapiro(df["rate"].sample(min(len(df), 5000), random_state=1))
    pooled = np.sqrt((g1.var() + g0.var()) / 2)
    d_cohen = (g1.mean() - g0.mean()) / pooled
    st.dataframe(pd.DataFrame({
        "Statistic": ["Mean (online)", "Mean (no online)", "Welch t", "p (t-test)", "Mann-Whitney p",
                      "Levene p (equal var?)", "Shapiro p (normality, sample)", "Cohen's d"],
        "Value": [g1.mean(), g0.mean(), t, p, pu, lev.pvalue, sh.pvalue, d_cohen]}).round(4),
        hide_index=True)
    st.info(("Reject H0: " if p < alpha else "Fail to reject H0: ") +
            f"ratings {'differ significantly' if p < alpha else 'do not differ significantly'} by online ordering "
            f"(effect size d = {d_cohen:.2f}). Shapiro p < 0.05 means ratings are not perfectly normal, so Mann-Whitney is a robustness check.")

    # --- Test 2: ANOVA
    st.markdown("### 2. Does rating differ across restaurant types? (One-way ANOVA + Kruskal-Wallis + Tukey)")
    st.write("**H0:** all restaurant types have the same mean rating. **H1:** at least one differs.")
    vc = df["main_type"].value_counts()
    keep = vc[vc >= 30].index.tolist()[:8]
    sub = df[df["main_type"].isin(keep)]
    groups = [g["rate"].values for _, g in sub.groupby("main_type")]
    F, pF = stats.f_oneway(*groups)
    H, pH = stats.kruskal(*groups)
    levA = stats.levene(*groups)
    grand = sub["rate"].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((sub["rate"] - grand) ** 2).sum()
    eta2 = ssb / sst
    st.dataframe(pd.DataFrame({
        "Statistic": ["Types compared", "ANOVA F", "ANOVA p", "Kruskal-Wallis H", "Kruskal p",
                      "Levene p", "Eta-squared"],
        "Value": [", ".join(keep), F, pF, H, pH, levA.pvalue, eta2]}), hide_index=True)
    st.info(("Reject H0: " if pF < alpha else "Fail to reject H0: ") +
            f"restaurant type explains about {eta2*100:.1f}% of rating variance. "
            "Levene p < 0.05 suggests unequal variances, so rely on Kruskal-Wallis as confirmation.")
    top5 = vc.index[:5].tolist()
    tk = pairwise_tukeyhsd(df[df["main_type"].isin(top5)]["rate"],
                           df[df["main_type"].isin(top5)]["main_type"])
    tk_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
    st.write("Tukey HSD post-hoc (5 most common types):")
    st.dataframe(tk_df, hide_index=True, use_container_width=True)

    # --- Test 3: Chi-square
    st.markdown("### 3. Is table booking associated with high ratings? (Chi-square test of independence)")
    st.write("**H0:** table booking and rating category are independent. **H1:** they are associated. "
             "High rating = 4.0 or above.")
    cat = np.where(df["rate"] >= 4.0, "High (≥4.0)", "Lower (<4.0)")
    ct = pd.crosstab(df["book_table"], cat)
    chi2, pc, dof, _ = stats.chi2_contingency(ct)
    cramers = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    st.dataframe(ct, use_container_width=True)
    st.write(f"χ² = {chi2:.2f}, dof = {dof}, p = {pc:.4g}, Cramér's V = {cramers:.3f}")
    st.info(("Reject H0: " if pc < alpha else "Fail to reject H0: ") +
            f"table booking is {'associated' if pc < alpha else 'not associated'} with high rating "
            f"(effect size V = {cramers:.2f}).")

    # --- Regression
    st.markdown("### 4. What drives rating? (Multiple linear regression)")
    model = smf.ols("rate ~ log_votes + cost + online_bin + book_bin", data=df).fit()
    ci = model.conf_int()
    coef = pd.DataFrame({"Coefficient": model.params, "Std error": model.bse,
                         "t": model.tvalues, "p-value": model.pvalues,
                         "CI low": ci[0], "CI high": ci[1]})
    st.dataframe(coef.round(4), use_container_width=True)
    st.write(f"R² = {model.rsquared:.3f}, Adjusted R² = {model.rsquared_adj:.3f}, "
             f"F-test p = {model.f_pvalue:.3g}, n = {int(model.nobs):,}")
    st.info("Interpret each coefficient holding the others constant, e.g. the log_votes coefficient is the change in "
            "rating for a one-unit rise in log(1+votes). Low R² means rating depends on many factors not in this data (taste, service).")
    resid = px.scatter(x=model.fittedvalues, y=model.resid, opacity=0.3,
                       labels={"x": "Fitted rating", "y": "Residual"}, title="Residuals vs fitted")
    st.plotly_chart(resid, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4: K-MEANS SEGMENTS
# ----------------------------------------------------------------------------
with tab_seg:
    st.subheader("Restaurant segments (K-means on rating, log votes and cost)")
    feats = ["rate", "log_votes", "cost"]
    Xs = StandardScaler().fit_transform(df[feats])

    ks = range(2, 9)
    rng = np.random.RandomState(1)
    idx = rng.choice(len(df), size=min(5000, len(df)), replace=False)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=1).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs[idx], km.labels_[idx]))
    ca, cb = st.columns(2)
    ca.plotly_chart(px.line(x=list(ks), y=inertia, markers=True, title="Elbow method",
                            labels={"x": "k", "y": "Inertia"}), use_container_width=True)
    cb.plotly_chart(px.line(x=list(ks), y=sil, markers=True, title="Silhouette score",
                            labels={"x": "k", "y": "Silhouette"}), use_container_width=True)

    k_final = st.slider("Choose k", 2, 8, 4)
    km = KMeans(n_clusters=k_final, n_init=10, random_state=1).fit(Xs)
    seg = df.copy()
    seg["Cluster"] = km.labels_.astype(str)
    prof = seg.groupby("Cluster").agg(Restaurants=("name", "count"), Avg_rating=("rate", "mean"),
                                      Median_votes=("votes", "median"), Avg_cost=("cost", "mean"))
    st.dataframe(prof.round(2), use_container_width=True)
    plot_df = seg[seg["cost"] <= seg["cost"].quantile(0.99)].sample(min(6000, len(seg)), random_state=1)
    st.plotly_chart(px.scatter(plot_df, x="cost", y="rate", color="Cluster", size_max=6,
                               hover_name="name", opacity=0.6,
                               title="Segments: cost vs rating"), use_container_width=True)
    st.caption("Name the clusters from the profile table in your report, e.g. 'budget quick bites', "
               "'popular mid-range', 'premium dining'.")

# ----------------------------------------------------------------------------
# TAB 5: METHOD
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("Data cleaning log")
    st.dataframe(pd.DataFrame({
        "Step": ["Raw rows", "After removing exact duplicates", "After dropping missing/NEW/'-' rating, cost, etc.",
                 "After keeping one row per (restaurant, area)"],
        "Rows": [log["raw_rows"], log["after_dedup"], log["after_missing"], log["final_rows"]]}),
        hide_index=True)
    st.markdown("""
**Method summary**
1. Cleaning: parse `rate` ("4.1/5"), remove commas in cost, drop NEW / '-' / missing, de-duplicate.
2. EDA: descriptive statistics, skewness, histograms, boxplots, Spearman correlation.
3. Inference: Welch t-test, ANOVA + Tukey (with Levene, Shapiro, Kruskal-Wallis checks), chi-square, multiple regression.
4. Segmentation: K-means with elbow and silhouette.
5. Recommendation: Bayesian-adjusted rating, min-max normalised, weighted with popularity and affordability, after user filters.

**Limitations:** single snapshot, self-selected ratings, Bangalore only, no personal user history (so this is content-based, not collaborative filtering).
""")