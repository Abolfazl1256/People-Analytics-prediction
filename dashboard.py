# ======= imports =======
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load, dump
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, auc, silhouette_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ======= Streamlit base config =======
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide", page_icon="📊")
st.caption("Made by Abolfazl Asjadi | contact: https://t.me/Abolfazl_A1256")

# ======= UI defaults =======
pio.templates.default = "plotly_dark"
BASE_FONT = dict(family="Segoe UI, Vazirmatn, IRANSans, Arial", size=13)
ATTRITION_COLORS = {"Yes": "#E74C3C", "No": "#1f77b4"}   # Red=Leave, Blue=Stay

# ======= CSS for top nav =======
st.markdown(
    """
    <style>
    div[data-baseweb="radio"] > div { flex-direction: row; justify-content: center; gap: .75rem; flex-wrap: wrap; }
    div[data-baseweb="radio"] label {
        background: #14171f; padding: 8px 14px; border-radius: 10px; color: #fff;
        font-weight: 600; border: 1px solid #3a3f47; transition: .2s;
    }
    div[data-baseweb="radio"] label:hover { border-color:#4CAF50; box-shadow: 0 0 0 2px rgba(76,175,80,.25); }
    div[data-baseweb="radio"] input:checked + div { background: #4CAF50 !important; color: #fff !important; }
    [data-testid="stMetricValue"] { font-size: 42px; }
    [data-testid="stMetricDelta"] { font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ======= helper figs =======
def pretty_layout(fig, height=840):
    fig.update_layout(
        height=height,
        margin=dict(t=80, r=20, b=40, l=50),
        font=BASE_FONT,
        legend=dict(orientation="h", y=1.08, x=0, title=""),
        hoverlabel=dict(bgcolor="#121319", font_size=12, font_color="white"),
        bargap=0.25,
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
    )
    return fig

def show_fig(fig, height=460):
    fig.update_layout(
        height=height,
        margin=dict(t=60, r=20, b=40, l=50),
        font=BASE_FONT,
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        legend=dict(orientation="h", y=1.08, x=0, title="")
    )
    st.plotly_chart(fig, use_container_width=True)

# ======= constants =======
MODEL_PATH = r"C:\Users\asjadi\Desktop\project\best_model.joblib"
RANDOM_STATE = 42

# ======= data loading =======
@st.cache_data
def load_df_from_upload(file):
    if file is None:
        try:
            return pd.read_excel("IBM HR.xlsx", sheet_name="IBM HR")
        except Exception:
            return None
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# ======= ML helpers =======
def build_preprocess(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", ohe)])

    return ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)],
                             remainder="drop")

def fit_and_eval_all(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocess = build_preprocess(X)

    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=11),
    }
    param_grid = {
        "LogReg": {"clf__C": [0.1, 1, 3]},
        "DecisionTree": {"clf__max_depth": [4, 6, 10, None], "clf__min_samples_split": [2, 10, 20]},
        "RandomForest": {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 8, 14]},
        "KNN": {"clf__n_neighbors": [5, 9, 11, 15]},
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, best_estimators, pr_curves = [], {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocess), ("clf", clf)])
        gs = GridSearchCV(pipe, param_grid[name], scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)

        best_estimators[name] = gs.best_estimator_
        y_pred = gs.predict(X_te)
        y_prob = gs.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        pre = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1  = f1_score(y_te, y_pred, zero_division=0)

        precision, recall, _ = precision_recall_curve(y_te, y_prob)
        pr_auc = auc(recall, precision)
        pr_curves[name] = (precision, recall, pr_auc)

        results.append({"model": name, "best_params": gs.best_params_,
                        "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1, "PR_AUC": pr_auc})

    res_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    best_name = res_df.iloc[0]["model"]
    return res_df, best_estimators[best_name], (X_tr, X_te, y_tr, y_te), pr_curves

def predict_with_pipeline(mdl, sample_dict, threshold=0.3):
    prep = mdl.named_steps.get("prep", None)
    cols = list(prep.feature_names_in_) if (prep is not None and hasattr(prep, "feature_names_in_")) else None
    if cols is None:
        raise ValueError("ستون‌های آموزشی از مدل استخراج نشدند.")

    # detect numeric/categorical
    num_cols, cat_cols = [], []
    if hasattr(prep, "transformers_"):
        for name, _, cols_sel in prep.transformers_:
            if name == "num": num_cols = list(cols_sel)
            if name == "cat": cat_cols = list(cols_sel)

    # build complete row
    row = {c: np.nan for c in cols}
    for k, v in sample_dict.items():
        if k in row:
            row[k] = v
    X_input = pd.DataFrame([row], columns=cols)

    for c in num_cols:
        if c in X_input: X_input[c] = pd.to_numeric(X_input[c], errors="coerce")
    for c in cat_cols:
        if c in X_input: X_input[c] = X_input[c].astype("object")

    prob = mdl.predict_proba(X_input)[0, 1]
    label = "Leave" if prob >= threshold else "Stay"
    return label, float(prob)

# ======= DATA IN =======
st.sidebar.header("1) آپلود داده")
uploaded = st.sidebar.file_uploader("CSV یا Excel", type=["csv", "xlsx"])
df = load_df_from_upload(uploaded)

if df is None:
    st.info("برای شروع، فایل دیتا را آپلود کن.")
    st.stop()

df["_Attrition"] = (df["Attrition"].astype(str).str.strip().str.lower() == "yes").astype(int)
for c in ["Over18", "EmployeeCount", "StandardHours"]:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

df_no_label = df.drop(columns=["Attrition"]).copy() if "Attrition" in df.columns else df.copy()

st.sidebar.header("2) فیلترها")
def get_opts(col, fallback):
    return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else fallback

department = st.sidebar.multiselect("Department", get_opts("Department", []))
gender     = st.sidebar.multiselect("Gender", get_opts("Gender", []))
overtime   = st.sidebar.multiselect("OverTime", get_opts("OverTime", []))
age_range  = st.sidebar.slider("Age range", 18, int(df.get("Age", pd.Series([60])).max()), (18, 60))
inc_min = int(df.get("MonthlyIncome", pd.Series([500])).min()) if "MonthlyIncome" in df else 500
inc_max = int(df.get("MonthlyIncome", pd.Series([30000])).max()) if "MonthlyIncome" in df else 30000
inc_range  = st.sidebar.slider("MonthlyIncome range", inc_min, inc_max, (1009, 19999), step=100)

mask = pd.Series(True, index=df.index)
if department: mask &= df["Department"].astype(str).isin(department)
if gender:     mask &= df["Gender"].astype(str).isin(gender)
if overtime:   mask &= df["OverTime"].astype(str).isin(overtime)
if "Age" in df: mask &= df["Age"].between(*age_range, inclusive="both")
if "MonthlyIncome" in df: mask &= df["MonthlyIncome"].between(*inc_range, inclusive="both")

df_f = df[mask].copy()
st.sidebar.markdown(f"**Rows after filter:** {len(df_f):,}")

# ======= NAV =======
page = st.radio(
    " ",
    ("خلاصه", "EDA", "ارزیابی الگوریتم", "پیش‌بینی کارمند جدید", "افراد در معرض ریسک", "chat GPT", "Q&A", "What-If", "Cohorts"),
    horizontal=True, label_visibility="collapsed"
)

# ===================== Page 1: Overview =====================
if page == "خلاصه":
    # --- KPI Cards (Status bar) ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("تعداد کارکنان", f"{len(df_f):,}")
    with k2:
        if "MonthlyIncome" in df_f.columns:
            st.metric("میانگین حقوق ماهانه", f"{df_f['MonthlyIncome'].mean():,.0f}")
    with k3:
        if "_Attrition" in df_f.columns:
            st.metric("Attrition rate", f"{df_f['_Attrition'].mean():.1%}")
    with k4:
        if "OverTime" in df_f.columns:
            ot_yes = (df_f["OverTime"].astype(str) == "Yes").mean()
            st.metric("درصد افرادی که اضافه کار بالا دارند", f"{ot_yes:.1%}")

    st.markdown("---")

    # --- Cross-filter (اختیاری) + دیتای صفحه ---
    df_cf = df_f.copy()
    if "Department" in df_cf.columns:
        sel_dept = st.selectbox(
            "نمایش جزئیات برای Department",
            ["(همه)"] + get_opts("Department", [])
        )
        if sel_dept != "(همه)":
            df_cf = df_cf[df_cf["Department"].astype(str) == sel_dept]

    # --- Stacked Bar: Attrition by Department + دیتالیبل نرخ روی ستون قرمز ---
    if {"Department", "Attrition"}.issubset(df_cf.columns):
        g = (
            df_cf.assign(Attrition=df_cf["Attrition"].astype(str))
                 .groupby(["Department", "Attrition"])
                 .size().reset_index(name="count")
        )
        # مجموع هر دپارتمان برای محاسبه نرخ
        totals = g.groupby("Department")["count"].sum().reset_index(name="total")
        g = g.merge(totals, on="Department")
        g["rate"] = g["count"] / g["total"] * 100

        fig_dep = px.bar(
            g, x="Department", y="count", color="Attrition", barmode="stack",
            color_discrete_map={"Yes": "#E74C3C", "No": "#1f77b4"},
            category_orders={"Attrition": ["No", "Yes"]},  # قرمز رویِ ستون قرار بگیرد
            title="نرخ خروج هر دپارتمان",
            hover_data={"count":":,", "Department":True, "Attrition":True, "rate":":.1f"}
        )

        # دیتالیبلِ نرخ خروج روی بالای ستون قرمز (Yes)
        # چون استک است، y را روی total می‌گذاریم تا بالای کل ستون بیاید
        for dep, grp in g[g["Attrition"] == "Yes"].groupby("Department"):
            total_val = float(grp["total"].iloc[0])
            rate_val  = float(grp["rate"].iloc[0])
            fig_dep.add_annotation(
                x=dep, y=total_val + 3,  # کمی بالاتر از سقف ستون
                text=f"{rate_val:.1f}%",
                showarrow=False,
                font=dict(color="white", size=12, family="Segoe UI"),
                align="center"
            )

        st.plotly_chart(fig_dep, use_container_width=True)

    # --- Pie: Overall Attrition Share ---
    if "Attrition" in df_cf.columns:
        pie_df = df_cf["Attrition"].astype(str).value_counts().reset_index()
        pie_df.columns = ["Attrition", "count"]
        fig_pie = px.pie(
            pie_df, names="Attrition", values="count",
            color="Attrition",
            color_discrete_map={"Yes": "#E74C3C", "No": "#1f77b4"},
            title="سهم از ترک کار سازمان", hole=0.35
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ===================== Page 2: EDA =====================
elif page == "EDA":
    st.subheader("🔎 EDA")

    # ----------------- 1) Interactive Correlation Heatmap (Plotly) -----------------
    num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        corr = df_f[num_cols].corr(numeric_only=True)
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="RdBu", zmin=-1, zmax=1,
                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.2f}<extra></extra>"
            )
        )
        fig_hm.update_layout(
            title="Correlation Heatmap (interactive)",
            height=520, margin=dict(t=50, l=60, r=20, b=40)
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # ----------------- 2) Age Histogram with adjustable bins -----------------
    if {"Age", "Attrition"}.issubset(df_f.columns):
        bins = st.slider("تعداد bins هیستوگرام سن", 5, 60, 20, 1, key="eda_bins_age")
        fig_hist = px.histogram(
            df_f, x="Age", color="Attrition", nbins=bins,
            barmode="overlay", opacity=0.6,
            color_discrete_map=ATTRITION_COLORS,
            title="Age Distribution by Attrition"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ----------------- 3) 3D Scatter + Selectable Correlation -----------------
    # همه‌چیز روی df_f است (نه dfp) تا اروری از نبودن متغیر رخ ندهد.
    num_cols_all = df_f.select_dtypes(include=[np.number]).columns.tolist()
    has_needed = {"Age", "MonthlyIncome", "DistanceFromHome"}.issubset(df_f.columns)

    if has_needed and len(num_cols_all) >= 3:
        st.markdown("### 🔎 اکتشاف سه‌بعدی + همبستگی")

        c_left, c_mid, c_right = st.columns([1.2, 1.2, 1])

        with c_left:
            x3d = st.selectbox(
                "X (عددی)", options=num_cols_all,
                index=(num_cols_all.index("Age") if "Age" in num_cols_all else 0),
                key="eda_x3d"
            )
            y3d = st.selectbox(
                "Y (عددی)", options=num_cols_all,
                index=(num_cols_all.index("MonthlyIncome") if "MonthlyIncome" in num_cols_all else min(1, len(num_cols_all)-1)),
                key="eda_y3d"
            )
            z3d = st.selectbox(
                "Z (عددی)", options=num_cols_all,
                index=(num_cols_all.index("DistanceFromHome") if "DistanceFromHome" in num_cols_all else min(2, len(num_cols_all)-1)),
                key="eda_z3d"
            )

        with c_mid:
            marker_size = st.slider("اندازه‌ی نقاط (3D)", 2, 10, 4, 1, key="eda_marker3d")
            opacity3d   = st.slider("شفافیت نقاط", 0.2, 1.0, 0.7, 0.05, key="eda_opacity3d")
            lo = 200 if len(df_f) >= 200 else max(50, len(df_f)//5 or 1)
            hi = min(2000, max(1, len(df_f)))
            default_n = min(800, hi)
            sample_n = st.slider("تعداد نمونه (برای سرعت)", lo, hi, default_n, 50, key="eda_sample3d")

        with c_right:
            corr_x = st.selectbox(
                "همبستگی: متغیر اول", options=num_cols_all,
                index=(num_cols_all.index("Age") if "Age" in num_cols_all else 0),
                key="eda_corrx"
            )
            corr_y = st.selectbox(
                "همبستگی: متغیر دوم", options=num_cols_all,
                index=(num_cols_all.index("MonthlyIncome") if "MonthlyIncome" in num_cols_all else min(1, len(num_cols_all)-1)),
                key="eda_corry"
            )
            corr_method = st.radio("روش همبستگی", ["Pearson", "Spearman"], horizontal=True, key="eda_corrm")

        # نمونه‌گیری (برای رندر نرم 3D)
        df3 = df_f.sample(sample_n, random_state=RANDOM_STATE) if len(df_f) > sample_n else df_f.copy()

        # 3D Scatter (حباب‌ها کوچکتر و خواناتر)
        fig3d = px.scatter_3d(
            df3, x=x3d, y=y3d, z=z3d,
            color=("Attrition" if "Attrition" in df3.columns else None),
            color_discrete_map=ATTRITION_COLORS,
            opacity=opacity3d,
            hover_data={x3d:":,.0f", y3d:":,.0f", z3d:":,.0f"},
            title=f"3D Scatter: {x3d} • {y3d} • {z3d}"
        )
        fig3d.update_traces(marker=dict(size=marker_size))
        fig3d.update_layout(scene=dict(
            xaxis_title=x3d, yaxis_title=y3d, zaxis_title=z3d,
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))
        ))
        st.plotly_chart(fig3d, use_container_width=True)

        # محاسبه‌ی همبستگیِ انتخابی + نمایش بصری کمکی
        s1 = pd.to_numeric(df_f[corr_x], errors="coerce")
        s2 = pd.to_numeric(df_f[corr_y], errors="coerce")
        clean = pd.DataFrame({corr_x: s1, corr_y: s2}).dropna()

        if len(clean) >= 3:
            corr_val = clean[corr_x].corr(clean[corr_y], method=("pearson" if corr_method == "Pearson" else "spearman"))
            k1, k2 = st.columns(2)
            with k1:
                st.metric(f"ضریب همبستگی ({corr_method})", f"{corr_val:.3f}")
            with k2:
                st.caption(f"نمونه‌های معتبر: {len(clean):,}")

            view_mode = st.radio(
                "نمایش دوبعدیِ کمک‌بصری", ["Scatter", "Density (Heatmap)"],
                horizontal=True, key="eda_corr_view"
            )
            if view_mode == "Scatter":
                fig2d = px.scatter(
                    clean, x=corr_x, y=corr_y,
                    color=(df_f.loc[clean.index, "Attrition"].astype(str) if "Attrition" in df_f.columns else None),
                    color_discrete_map=ATTRITION_COLORS,
                    opacity=0.7,
                    hover_data={corr_x:":,.0f", corr_y:":,.0f"},
                    title=f"Scatter: {corr_x} vs {corr_y}"
                )
                fig2d.update_traces(marker=dict(size=6, line=dict(width=0)))
            else:
                fig2d = px.density_heatmap(
                    clean, x=corr_x, y=corr_y,
                    nbinsx=30, nbinsy=30, color_continuous_scale="Viridis",
                    title=f"Density Heatmap: {corr_x} vs {corr_y}"
                )
            st.plotly_chart(fig2d, use_container_width=True)
        else:
            st.info("برای محاسبه‌ی همبستگی، داده‌ی کافی پیدا نشد.")

    st.markdown("---")

    # ----------------- 4) K-Means + Attrition Filter + Cluster Labels -----------------
    st.subheader("🧩 خوشه‌بندی (K-Means) + فیلتر Attrition")
    attr_filter = st.radio(
        "وضعیت Attrition برای خوشه‌بندی",
        ("هر دو", "فقط ترک کرده‌ها (Yes)", "فقط مانده‌ها (No)"),
        horizontal=True, key="eda_attr_filter"
    )
    base_df = df_f.copy()
    if "Attrition" in base_df.columns:
        if attr_filter == "فقط ترک کرده‌ها (Yes)":
            base_df = base_df[base_df["Attrition"].astype(str) == "Yes"]
        elif attr_filter == "فقط مانده‌ها (No)":
            base_df = base_df[base_df["Attrition"].astype(str) == "No"]

    num_cols_eda = base_df.select_dtypes(include=[np.number]).columns.tolist()
    with st.expander("راهنما", expanded=False):
        st.write("ستون‌های عددی را انتخاب کن، استانداردسازی می‌شود، KMeans اجرا و با PCA دو‌بعدی (رنگ=خوشه، نماد=Attrition) نمایش داده می‌شود.")

    choose_cols = st.multiselect(
        "ستون‌های عددی برای خوشه‌بندی",
        options=num_cols_eda,
        default=[c for c in ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany"] if c in num_cols_eda],
        key="eda_km_cols"
    )
    k = st.slider("تعداد خوشه‌ها (k)", 2, 8, 3, 1, key="eda_km_k")

    # برچسب‌گذاری ساده‌ی خوشه‌ها
    def _label_cluster(row):
        parts = []
        if "MonthlyIncome" in row and not pd.isna(row.get("MonthlyIncome_q3")) and row["MonthlyIncome"] >= row["MonthlyIncome_q3"]:
            parts.append("High Income")
        elif "MonthlyIncome" in row and not pd.isna(row.get("MonthlyIncome_q1")) and row["MonthlyIncome"] <= row["MonthlyIncome_q1"]:
            parts.append("Low Income")
        if "YearsAtCompany" in row and row["YearsAtCompany"] <= 2:
            parts.append("Newcomers")
        if "Age" in row and row["Age"] <= 30:
            parts.append("Young")
        if "DistanceFromHome" in row and row["DistanceFromHome"] >= 15:
            parts.append("Far Home")
        return ", ".join(parts) if parts else "General"

    if st.button("🚀 اجرای خوشه‌بندی", key="eda_run_kmeans"):
        if len(choose_cols) < 2:
            st.warning("حداقل ۲ ستون انتخاب کن.")
        else:
            # فقط ردیف‌های کاملِ ستون‌های انتخابی
            Xc = base_df[choose_cols].dropna().copy()
            if Xc.empty:
                st.warning("پس از فیلتر Attrition یا حذف مقادیر خالی، داده‌ای باقی نمانده است.")
            else:
                # استانداردسازی و KMeans
                from sklearn.preprocessing import StandardScaler
                Xs = StandardScaler().fit_transform(Xc)
                km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                labels = km.fit_predict(Xs)

                # کیفیت
                sil = silhouette_score(Xs, labels)
                st.success(f"خوشه‌بندی انجام شد. Silhouette Score = **{sil:.3f}**")
                st.caption(f"تعداد رکوردهای استفاده‌شده: {len(Xc):,}")

                # پروفایل
                prof = Xc.assign(cluster=labels).groupby("cluster").agg(["mean", "median", "count"])
                st.write("**پروفایل خوشه‌ها:**")
                st.dataframe(prof, use_container_width=True)

                # برچسب پیشنهادی با صدک‌ها
                q = base_df[choose_cols].quantile([.25, .75])
                q1, q3 = q.iloc[0], q.iloc[1]
                centers = Xc.assign(cluster=labels).groupby("cluster").mean()
                for c in centers.columns:
                    centers[c + "_q1"] = q1.get(c, np.nan)
                    centers[c + "_q3"] = q3.get(c, np.nan)
                centers["label"] = centers.apply(_label_cluster, axis=1)
                st.write("**برچسب‌های پیشنهادی خوشه‌ها:**")
                st.dataframe(centers[["label"]], use_container_width=True)

                # آماده‌سازی برای نمودار: رنگ=خوشه، نماد=Attrition
                if "Attrition" in base_df.columns:
                    attr_sym = base_df.loc[Xc.index, "Attrition"].astype(str)
                else:
                    attr_sym = pd.Series(["Unknown"] * len(Xc), index=Xc.index)

                pts = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
                plot_df = pd.DataFrame({
                    "pc1": pts[:, 0], "pc2": pts[:, 1],
                    "cluster": labels.astype(str),
                    "Attrition": attr_sym.values
                })
                fig = px.scatter(
                    plot_df, x="pc1", y="pc2", color="cluster", symbol="Attrition",
                    symbol_map={"Yes": "x", "No": "circle"},
                    title="KMeans clusters (PCA 2D) — نماد بر اساس Attrition"
                )
                fig.update_traces(marker=dict(size=6, opacity=0.85))
                show_fig(fig, height=420)

                # جدول شمارش اعضای هر خوشه به تفکیک Attrition
                if "Attrition" in base_df.columns:
                    cnt = plot_df.groupby(["cluster", "Attrition"]).size().reset_index(name="count")
                    st.write("**تعداد افراد هر خوشه به تفکیک Attrition:**")
                    st.dataframe(cnt, use_container_width=True)

# ===================== Page 3: Model Evaluation =====================
elif page == "ارزیابی الگوریتم":
    st.subheader("📐 آماده‌سازی داده برای مدل")
    work_df = df_no_label.copy()
    target = "_Attrition"
    y = work_df[target].values
    X = work_df.drop(columns=[target])

    with st.spinner("درحال آموزش و اجرای مدل ها"):
        res_df, best_model, splits, pr_curves = fit_and_eval_all(X, y)
        X_tr, X_te, y_tr, y_te = splits

        st.success("آموزش و ارزیابی انجام شد.")
    st.write("**نتایج مقایسه‌ای (مرتب بر اساس F1):**")

    # 📊 ماتریس شرطی به‌جای نمودار PR
    st.subheader("📊 مقایسه‌ی مدل‌ها (ماتریس با فرمت‌بندی شرطی)")

    # فقط ستون‌های امتیازی را برداریم
    score_cols = ["Accuracy", "Precision", "Recall", "F1", "PR_AUC"]
    mat = (
        res_df
        .set_index("model")[score_cols]
        .astype(float)
        .round(3)
    )

    # هایلایت بهترین مقدار هر ستون
    def highlight_best(col):
        is_best = col == col.max()
        return ["font-weight:700; border:1px solid #555;" if v else "" for v in is_best]

    # استایل‌دهی: گرادیان رنگ + هایلایت
    sty = (
        mat.style
        .background_gradient(cmap="RdYlGn", axis=0)   # سبز=خوب، قرمز=ضعیف
        .apply(highlight_best, axis=0)                # بهترین هر ستون
        .format("{:.3f}")
        .set_caption("Higher is better (Green → Red)")
    )

    st.dataframe(sty, use_container_width=True)


    # --- Radar chart: compare models on metrics ---
    melt = res_df[["model","Accuracy","Precision","Recall","F1","PR_AUC"]].melt(
        id_vars="model", var_name="metric", value_name="val"
    )
    fig_radar = px.line_polar(melt, r="val", theta="metric", color="model", line_close=True,
                              title="Radar: مقایسه مدل‌ها")
    fig_radar.update_traces(fill='toself', opacity=0.6)
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- Confusion Matrix as heatmap (green=correct, red=error) ---
    y_pred = best_model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred Stay(0)","Pred Leave(1)"], y=["True Stay(0)","True Leave(1)"],
        colorscale=[ [0,"#0b6623"], [0.5,"#e8e8e8"], [1,"#b71c1c"] ],
        text=cm, texttemplate="%{text}",
        hovertemplate="x=%{x}<br>y=%{y}<br>count=%{z}<extra></extra>"
    ))
    fig_cm.update_layout(title="Confusion Matrix (Best)", height=420)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature Importance (Permutation)")
    from sklearn.inspection import permutation_importance

    # آماده‌سازی X_all مطابق ستون‌های آموزش
    X_all = df_no_label.copy()
    y_all = X_all["_Attrition"].values
    X_all = X_all.drop(columns=["_Attrition"])
    prep = best_model.named_steps["prep"]
    cols = list(prep.feature_names_in_)
    for c in cols:
        if c not in X_all: X_all[c] = np.nan
    X_all = X_all[cols]

    r = permutation_importance(best_model, X_all, y_all,
                               n_repeats=8, random_state=RANDOM_STATE, scoring="f1")
    imp = pd.DataFrame({"feature": cols, "importance": r.importances_mean}) \
            .sort_values("importance", ascending=False).head(20)
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="Top Features (Permutation)")
    st.plotly_chart(fig_imp, use_container_width=True)

# ===================== Page 4: New Prediction =====================
elif page == "پیش‌بینی کارمند جدید":
    st.subheader("🤖 بارگذاری مدل")
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load(MODEL_PATH)
            st.success(f"مدل خارجی بارگذاری شد: {MODEL_PATH}")
        except Exception as e:
            st.warning(f"لود مدل خارجی ناموفق بود:\n{e}")
    if model is None and os.path.exists("best_model.joblib"):
        try:
            model = load("best_model.joblib")
            st.success("best_model.joblib (داخل پوشه) بارگذاری شد.")
        except Exception as e:
            st.warning(f"لود best_model.joblib ناموفق: {e}")
    if model is None:
        st.error("مدلی برای پیش‌بینی پیدا نشد. ابتدا در صفحه ارزیابی، مدل را آموزش و ذخیره کن.")
        st.stop()

    st.markdown("**فرم ورودی**")
    c1, c2, c3 = st.columns(3)
    Age = c1.number_input("Age", 18, 70, 30)
    MonthlyIncome = c2.number_input("MonthlyIncome", 500, 30000, 5000, step=100)
    DistanceFromHome = c3.number_input("DistanceFromHome", 0, 60, 10)
    YearsAtCompany = c1.number_input("YearsAtCompany", 0, 40, 3)
    Education = c2.selectbox("Education (1-5)", [1,2,3,4,5], index=2)

    def opts(col, fallback):
        return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else fallback
    Gender = c3.selectbox("Gender", opts("Gender", ["Male","Female"]))
    Department = c1.selectbox("Department", opts("Department", ["Sales","Research & Development","Human Resources"]))
    JobRole = c2.selectbox("JobRole", opts("JobRole", ["Sales Executive"]))
    OverTime = c3.selectbox("OverTime", opts("OverTime", ["Yes","No"]))
    MaritalStatus = c1.selectbox("MaritalStatus", opts("MaritalStatus", ["Single","Married","Divorced"]))
    Threshold = c2.slider("Decision Threshold (Leave)", 0.05, 0.8, 0.3, 0.01)

    if st.button("🔮 Predict"):
     sample = {
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "DistanceFromHome": DistanceFromHome,
        "YearsAtCompany": YearsAtCompany,
        "Education": Education,
        "Gender": Gender,
        "Department": Department,
        "JobRole": JobRole,
        "OverTime": OverTime,
        "MaritalStatus": MaritalStatus,
    }

    try:
        label, prob = predict_with_pipeline(model, sample, threshold=Threshold)

        # متن نتیجه
        st.markdown(
            f"**نتیجه:** `{label}` | **Prob(Leave):** `{prob:.1%}` | **Threshold:** `{Threshold:.2f}`"
        )

        # --- Gauge برای نمایش احتمال ---
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={'suffix': " %"},
            delta={'reference': Threshold * 100},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E74C3C" if prob >= Threshold else "#1f77b4"},
                'steps': [
                    {'range': [0, 30], 'color': "#2b4c7e"},
                    {'range': [30, 60], 'color': "#ffa94d"},
                    {'range': [60, 100], 'color': "#ff6b6b"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': Threshold * 100,
                }
            }
        ))
        fig_g.update_layout(title="Gauge: احتمال Leave")
        st.plotly_chart(fig_g, use_container_width=True)

    except Exception as e:
        st.error(f"❌ خطای پیش‌بینی: {e}")


# ===================== Page 5: Q&A =====================
elif page == "Q&A":
    st.subheader("💬 پاسخ هوشمند به پرسش‌های مدیریتی")

    # 1) کدام دپارتمان بیشترین نرخ ترک کار را دارد و چرا؟
    st.markdown("**1) کدام دپارتمان بیشترین نرخ ترک کار را دارد و چرا؟**")
    if "Department" in df_f.columns:
        rate_by_dept = df_f.groupby("Department")["_Attrition"].mean().sort_values(ascending=False)
        st.write(rate_by_dept.to_frame("AttritionRate"))
        top_dept = rate_by_dept.index[0]
        st.info(f"The highest rate of abandonment is related to: **{top_dept}** at a rate of**{rate_by_dept.iloc[0]:.1%}**")
        hints = []
        if {"MonthlyIncome","Department"}.issubset(df_f.columns):
            mean_income = df_f.groupby("Department")["MonthlyIncome"].mean().sort_values()
            if top_dept in mean_income.index[:1]:
                hints.append("میانگین حقوق در این دپارتمان پایین‌تر از بقیه است.")
        if {"OverTime","Department"}.issubset(df_f.columns):
            ot_rate = df_f[df_f["OverTime"]=="Yes"].groupby("Department")["_Attrition"].mean().sort_values(ascending=False)
            if top_dept in ot_rate.index[:1]:
                hints.append("The Attrition rate is higher among overtime here.")
        if hints:
            st.write("Possible reasons:", "، ".join(hints))
    else:
        st.warning("The Department column is not available.")

    st.markdown("---")

    # نمودار تکمیلی OverTime vs Attrition
    if "OverTime" in df_f.columns:
        piv = df_f.groupby(["OverTime","Attrition"]).size().reset_index(name="count")
        fig = px.bar(piv, x="OverTime", y="count", color="Attrition",
                     color_discrete_map=ATTRITION_COLORS, barmode="group",
                     template="plotly_dark", title="Count by OverTime & Attrition")
        fig.update_traces(hovertemplate="OverTime=%{x}<br>Attrition=%{legendgroup}<br>Count=%{y}<extra></extra>")
        show_fig(fig, height=380)

    # 2) آیا OverTime رابطه مستقیمی با ترک کار دارد؟
    st.markdown("**2) آیا اضافه‌کاری رابطه مستقیمی با ترک کار دارد؟**")
    if "OverTime" in df_f.columns:
        base = df_f["_Attrition"].mean()
        ot_yes = df_f.loc[df_f["OverTime"]=="Yes","_Attrition"].mean()
        ot_no  = df_f.loc[df_f["OverTime"]=="No","_Attrition"].mean() if "No" in df_f["OverTime"].unique() else np.nan
        st.write(pd.DataFrame({"Overall":[base], "OT=Yes":[ot_yes], "OT=No":[ot_no]}).T.rename(columns={0:"AttritionRate"}))
        if ot_yes > base:
            st.info("Yes-the Attrition rate for overtime=Yes is higher than the average.")
        else:
            st.info("Strong evidence for overtime's direct relationship with Attrition was not seen (in current filtered data).")
    else:
        st.warning("Overtime column is not available.")

    st.markdown("---")

    # 3) کارکنان با چه ویژگی‌هایی بیشتر در معرض خروج هستند؟
    st.markdown("**3) کارکنانی با چه ویژگی‌هایی (سن، درآمد، سابقه) بیشتر در معرض خروج هستند؟**")
    hints = []
    if "Age" in df_f:
        young = df_f[df_f["Age"]<=df_f["Age"].quantile(0.25)]["_Attrition"].mean()
        old   = df_f[df_f["Age"]>=df_f["Age"].quantile(0.75)]["_Attrition"].mean()
        hints.append(f"Lower age group (Quartile1) rate of Attrition ≈ {young:.1%}، Elderly group (Quartile4) ≈ {old:.1%}.")
    if "MonthlyIncome" in df_f:
        low_inc = df_f[df_f["MonthlyIncome"]<=df_f["MonthlyIncome"].quantile(0.25)]["_Attrition"].mean()
        high_inc= df_f[df_f["MonthlyIncome"]>=df_f["MonthlyIncome"].quantile(0.75)]["_Attrition"].mean()
        hints.append(f"Wages lower than Q1 Attrition rate≈ {low_inc:.1%}، salary above Q3 ≈ {high_inc:.1%}.")
    if "YearsAtCompany" in df_f:
        low_ten = df_f[df_f["YearsAtCompany"]<=2]["_Attrition"].mean()
        hints.append(f" The tenure rate ≤2 years is often riskier ≈ {low_ten:.1%}).")
    if hints:
        st.write("- " + "\n- ".join(hints))

    st.markdown("---")

    st.markdown("**4) ریکال مدل شما چقدر است؟**")
    if "best_model_recall" in st.session_state:
        st.success(f"ریکال بهترین مدل (از صفحه ارزیابی): **{st.session_state['best_model_recall']:.3f}**")
    else:
        st.info("First, go to the model evaluation page and run the models to record the Recall.")

    st.markdown("---")

    st.markdown("**5) پیشنهادهای نگهداشت کارکنان**")
    st.write("""- مدیریت اضافه‌کاری و توازن زندگی-کار
- بازنگری حقوق/مزایا برای زیر میانگین‌ها
- برنامه‌های آنبوردینگ و منتورینگ برای تازه‌واردها
- بهبود رضایت شغلی/محیطی و مسیر رشد
- تسهیلات رفت‌وآمد یا دورکاری برای فاصله‌های زیاد
""")

# ===================== Page 6: High-Risk Stayers =====================
elif page == "افراد در معرض ریسک":
    st.subheader("🚨 فهرست کارکنانِ در حال کار با بیشترین احتمال خروج")

    # load model
    model = None
    if os.path.exists(MODEL_PATH):
        try: model = load(MODEL_PATH)
        except Exception as e: st.warning(f"لود مدل خارجی ناموفق بود:\n{e}")
    if model is None and os.path.exists("best_model.joblib"):
        try: model = load("best_model.joblib")
        except Exception as e: st.warning(f"لود best_model.joblib ناموفق: {e}")
    if model is None:
        st.error("مدلی برای امتیازدهی پیدا نشد. ابتدا در صفحه «ارزیابی الگوریتم» مدل را ذخیره کن.")
        st.stop()

    if "_Attrition" not in df_f.columns:
        st.error("ستون _Attrition در داده موجود نیست."); st.stop()

    still_here = df_f[df_f["_Attrition"] == 0].copy()
    if still_here.empty:
        st.info("با فیلترهای فعلی، موردی برای Stay یافت نشد."); st.stop()

    prep = model.named_steps.get("prep", None)
    if prep is None or not hasattr(prep, "feature_names_in_"):
        st.error("مدل باید Pipeline با گام 'prep' باشد."); st.stop()
    feature_cols = list(prep.feature_names_in_)

    X_batch_for_model = still_here.drop(columns=[c for c in ["EmployeeNumber","Attrition"] if c in still_here.columns])
    for c in feature_cols:
        if c not in X_batch_for_model.columns:
            X_batch_for_model[c] = np.nan
    X_batch_for_model = X_batch_for_model[feature_cols]

    try:
        risk_prob = model.predict_proba(X_batch_for_model)[:, 1]
    except Exception as e:
        st.error(f"خطا در پیش‌بینی گروهی: {e}"); st.stop()

    still_here["Risk(Leave)"] = risk_prob

    # risk factors (for heat-like styling)
    def inv_minmax(s): s=s.astype(float); return 1 - (s - s.min())/(s.max()-s.min()+1e-9)
    def minmax(s):    s=s.astype(float); return     (s - s.min())/(s.max()-s.min()+1e-9)

    if "OverTime" in still_here: still_here["rf_OverTime"] = (still_here["OverTime"].astype(str)=="Yes").astype(float)
    if "MonthlyIncome" in still_here: still_here["rf_LowIncome"] = inv_minmax(still_here["MonthlyIncome"])
    if "YearsAtCompany" in still_here:
        y = still_here["YearsAtCompany"].astype(float)
        still_here["rf_LowTenure"] = np.clip((5 - y) / 5.0, 0, 1)
    if "JobSatisfaction" in still_here:
        m = {1:1.0, 2:0.66, 3:0.33, 4:0.0}
        still_here["rf_JobSatLow"] = still_here["JobSatisfaction"].map(m).fillna(0.5)
    if "EnvironmentSatisfaction" in still_here:
        m2 = {1:1.0, 2:0.66, 3:0.33, 4:0.0}
        still_here["rf_EnvSatLow"] = still_here["EnvironmentSatisfaction"].map(m2).fillna(0.5)
    if "DistanceFromHome" in still_here:
        still_here["rf_FarHome"] = minmax(still_here["DistanceFromHome"])

    # Key Role heuristic
    key_mask = pd.Series(False, index=still_here.index)
    if "JobLevel" in still_here: key_mask |= (still_here["JobLevel"].fillna(1) >= 4)
    if "MonthlyIncome" in still_here:
        q3 = still_here["MonthlyIncome"].quantile(0.75)
        key_mask |= (still_here["MonthlyIncome"] >= q3)
    if "PerformanceRating" in still_here: key_mask |= (still_here["PerformanceRating"].fillna(3) >= 4)
    still_here.insert(1, "Key Role", np.where(key_mask, "Yes", "No"))

    show_cols = [c for c in ["EmployeeNumber","Key Role","Department","JobRole","Age","Gender",
                             "MonthlyIncome","YearsAtCompany","OverTime","JobSatisfaction",
                             "EnvironmentSatisfaction","DistanceFromHome","Risk(Leave)"] if c in still_here.columns]
    risk_cols = [c for c in still_here.columns if c.startswith("rf_")]

    top_k = st.slider("چند نفرِ اول نمایش داده شود؟", 5, min(200, len(still_here)), 30, 5)
    out = still_here.sort_values("Risk(Leave)", ascending=False).head(top_k)[show_cols + risk_cols]

    import matplotlib
    reds = matplotlib.cm.get_cmap("Reds")
    sty = (out.style
           .format({"Risk(Leave)": "{:.1%}", "MonthlyIncome": "{:,.0f}"})
           .set_caption("High-Risk Stayers")
           .background_gradient(cmap=reds, subset=risk_cols))
    st.write("**توضیح رنگ‌ها:** ستون‌های rf_* شدت عواملی را نشان می‌دهند که ریسک را بالا می‌برند (قرمز پررنگ=بدتر).")
    st.dataframe(sty, use_container_width=True)

    csv = out.drop(columns=risk_cols).to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ دانلود CSV", data=csv, file_name="high_risk_stayers.csv", mime="text/csv")

# ===================== Page 7: GPT Chatbot =====================
elif page == "chat GPT":
    from openai import OpenAI

    st.subheader("🤖 درمورد داده هات هر سوالی ازم داری بپرس")

    with st.expander("تنظیمات اتصال", expanded=False):
        st.write("کلید OpenAI را اینجا وارد کن (فقط در همین جلسه نگه‌داری می‌شود).")
        api_key = st.text_input("OPENAI_API_KEY", type="password", value=st.session_state.get("api_key",""))
        if st.checkbox("ذخیره موقت در Session", value=True):
            if api_key: st.session_state["api_key"] = api_key

    api_key = st.session_state.get("api_key") or st.secrets["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"]
    if not api_key:
        st.warning("کلید OpenAI را بده یا در محیط سیستم تنظیم کن.")
        st.stop()
    client = OpenAI(api_key=api_key)

    if "chat_history_fc" not in st.session_state:
        st.session_state.chat_history_fc = []

    def pandas_op(operation: str, column: str|None=None, group_by: str|None=None,
                  filter_expr: str|None=None, top_n: int|None=10, rate_of: str|None=None):
        data = df_f.copy()
        if filter_expr:
            try:
                import re
                if not re.fullmatch(r"[A-Za-z0-9_ .><=!&|'\-\(\)]+", filter_expr):
                    return {"error":"عبارت فیلتر مجاز نیست."}
                data = data.query(filter_expr)
            except Exception as e:
                return {"error": f"خطا در فیلتر: {e}"}
        try:
            if operation in ["mean","median","sum","count"] and column:
                if operation=="count":
                    val = int(data[column].shape[0]) if column not in data else int(data[column].count())
                else:
                    val = float(getattr(data[column], operation)())
                return {"result": val}
            if operation == "describe" and column:
                return {"table": data[column].describe().to_dict()}
            if operation == "group_mean" and column and group_by:
                tbl = data.groupby(group_by)[column].mean().reset_index().sort_values(column, ascending=False)
                fig = px.bar(tbl.head(top_n), x=group_by, y=column, title=f"میانگین {column} به تفکیک {group_by}")
                return {"table": tbl.head(top_n), "fig_json": fig.to_json()}

            if operation == "value_counts" and column:
                vc = data[column].astype(str).value_counts().head(top_n).to_frame("count")
                return {"table": vc.reset_index().rename(columns={"index":column})}
            if operation == "group_mean" and column and group_by:
                tbl = data.groupby(group_by)[column].mean().reset_index().sort_values(column, ascending=False)
                return {"table": tbl.head(top_n)}
            if operation == "attrition_rate_by" and group_by:
                if "_Attrition" not in data.columns: return {"error":"ستون _Attrition در داده نیست."}
                tbl = data.groupby(group_by)["_Attrition"].mean().reset_index().sort_values("_Attrition", ascending=False)
                tbl["_Attrition"] = (tbl["_Attrition"]*100).round(2)
                return {"table": tbl.head(top_n), "note":"مقادیر به درصد است."}
            if operation in ["preview","filter_preview"]:
                return {"table": data.head(min(top_n or 10, 50))}
            return {"error":"عملیات پشتیبانی نمی‌شود."}
        except Exception as e:
            return {"error": f"خطای اجرای عملیات: {e}"}

    tools = [{
        "type": "function",
        "function": {
            "name": "pandas_op",
            "description": "محاسبات روی df_f (میانگین/گروه‌بندی/نرخ Attrition و ...)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string",
                        "enum": ["mean","median","sum","count","describe","value_counts",
                                 "group_mean","attrition_rate_by","preview","filter_preview"]},
                    "column": {"type":"string"}, "group_by": {"type":"string"},
                    "filter_expr": {"type":"string"}, "top_n": {"type":"integer","default":10},
                    "rate_of": {"type":"string"}
                }, "required": ["operation"]
            }
        }
    }]

    # replay history
    for role, content in st.session_state.chat_history_fc:
        with st.chat_message("assistant" if role=="assistant" else "user"):
            st.markdown(content)

    user_msg = st.chat_input("سؤال مدیریتی/تحلیلی‌ات را بنویس…")
    if user_msg:
        with st.chat_message("user"): st.markdown(user_msg)

        def data_context_brief(df_ctx: pd.DataFrame) -> str:
            ctx = {"columns": list(df_ctx.columns),
                   "shape": list(df_ctx.shape),
                   "examples": df_ctx.head(3).to_dict(orient="records"),
                   "notes": "ستون هدف باینری: _Attrition (1=Leave, 0=Stay)"}
            return json.dumps(ctx, ensure_ascii=False)

        messages = [
            {"role":"system","content":
             "تو یک دستیار تحلیلی منابع انسانی هستی. اگر سوال شامل محاسبه است، از ابزار pandas_op استفاده کن. "
             "عدد دقیق بده و در پایان یک جمع‌بندی مدیریتی کوتاه ارائه کن."},
            {"role":"user","content": "Data context (JSON): " + data_context_brief(df_f)}
        ]
        for role, content in st.session_state.chat_history_fc[-6:]:
            messages.append({"role": role, "content": content})
        messages.append({"role":"user","content": user_msg})

        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools,
            tool_choice="auto", temperature=0.2, max_tokens=600
        )
        tool_calls = getattr(resp.choices[0].message, "tool_calls", None)
        final_answer = None

        if tool_calls:
            for tc in tool_calls:
                if tc.function.name == "pandas_op":
                    try: args = json.loads(tc.function.arguments or "{}")
                    except Exception: args = {}
                    result = pandas_op(**args)
                    if isinstance(result, dict) and "table" in result:
                        tbl = result["table"]
                        if isinstance(tbl, dict): tbl = pd.DataFrame([tbl])
                        st.dataframe(tbl, use_container_width=True)
                        if isinstance(result, dict) and "fig_json" in result:
                            fig = pio.from_json(result["fig_json"])
                            st.plotly_chart(fig, use_container_width=True)

                    follow_messages = messages + [resp.choices[0].message, {
                        "role": "tool", "tool_call_id": tc.id, "name": "pandas_op",
                        "content": json.dumps(result, ensure_ascii=False, default=str)
                    }]
                    resp2 = client.chat.completions.create(
                        model="gpt-4o-mini", messages=follow_messages,
                        temperature=0.2, max_tokens=600
                    )
                    final_answer = resp2.choices[0].message.content
                    break
        else:
            final_answer = resp.choices[0].message.content

        if not final_answer: final_answer = "نتوانستم پاسخی تولید کنم."
        with st.chat_message("assistant"): st.markdown(final_answer)
        st.session_state.chat_history_fc += [("user", user_msg), ("assistant", final_answer)]

    c1, c2 = st.columns(2)
    if c1.button("🧹 پاک کردن تاریخچه"):
        st.session_state.chat_history_fc = []; st.rerun()
    if c2.button("🔎 مثال‌های مفید"):
        st.info("نمونه‌ها: میانگین سن؟ | نرخ Attrition به تفکیک Department؟ | میانگین حقوق در Sales؟ | 10 شغلِ بالاترین حقوق؟")


# ===================== Page 8: What-If =====================

elif page == "What-If":
    st.subheader("تحلیل سناریوها (What-If)")
    # بارگذاری مدل
    model = None
    for p in [MODEL_PATH, "best_model.joblib"]:
        if os.path.exists(p):
            try:
                model = load(p); break
            except Exception:
                pass
    if model is None:
        st.warning("مدل در دسترس نیست. ابتدا صفحه ارزیابی را اجرا و ذخیره کن.")
        st.stop()

    colA, colB = st.columns(2)
    sal_up = colA.slider("افزایش حقوق (%)", 0, 50, 10, 5)
    ot_down = colB.slider("کاهش نرخ OverTime (%)", 0, 100, 20, 5)

    base = df_f.copy()
    scen = base.copy()
    if "MonthlyIncome" in scen:
        scen["MonthlyIncome"] = scen["MonthlyIncome"] * (1 + sal_up/100.0)
    if "OverTime" in scen:
        yes_idx = scen["OverTime"].astype(str) == "Yes"
        to_flip = int(yes_idx.sum() * (ot_down/100.0))
        if to_flip > 0:
            flip_idx = scen[yes_idx].sample(to_flip, random_state=RANDOM_STATE).index
            scen.loc[flip_idx, "OverTime"] = "No"

    # پیش‌بینی نرخ Attrition قبل/بعد
    def _proba_mean(m, data):
        prep = m.named_steps.get("prep"); cols = list(prep.feature_names_in_)
        X = data.copy()
        if "Attrition" in X: X = X.drop(columns=["Attrition"])
        for c in cols:
            if c not in X: X[c] = np.nan
        X = X[cols]
        return m.predict_proba(X)[:, 1].mean()

    base_rate = _proba_mean(model, base)
    scen_rate = _proba_mean(model, scen)

    cmp_df = pd.DataFrame({
        "Scenario": ["Baseline", "New"],
        "Expected Attrition (%)": [base_rate*100, scen_rate*100]
    })
    fig_s = px.bar(cmp_df, x="Scenario", y="Expected Attrition (%)", text="Expected Attrition (%)",
                   title="تأثیر سناریو بر Attrition (مدل)")
    fig_s.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_s, use_container_width=True)

# ===================== Page 9: Cohorts =====================
elif page == "Cohorts":
    st.subheader("تحلیل گروهی (Cohort)")
    if {"Age","YearsAtCompany","_Attrition"}.issubset(df_f.columns):
        age_bin = st.slider("بازه سنی (سال)", 5, 15, 10, 1)
        ten_bin = st.slider("بازه سابقه (سال)", 1, 10, 3, 1)
        dfc = df_f.copy()
        dfc["AgeBand"] = pd.cut(dfc["Age"],
                                bins=range(int(dfc["Age"].min()), int(dfc["Age"].max())+age_bin, age_bin),
                                right=False)
        dfc["TenureBand"] = pd.cut(dfc["YearsAtCompany"],
                                   bins=range(0, int(dfc["YearsAtCompany"].max())+ten_bin, ten_bin),
                                   right=False)
        coh = dfc.groupby(["AgeBand","TenureBand"])["_Attrition"].mean().reset_index()
        coh["_Attrition"] = (coh["_Attrition"]*100).round(1)
        coh_piv = coh.pivot(index="AgeBand", columns="TenureBand", values="_Attrition")
        fig_coh = go.Figure(go.Heatmap(
            z=coh_piv.values,
            x=[str(c) for c in coh_piv.columns],
            y=[str(i) for i in coh_piv.index],
            colorscale="YlOrRd",
            hovertemplate="Age=%{y}<br>Tenure=%{x}<br>Rate=%{z:.1f}%<extra></extra>"
        ))
        fig_coh.update_layout(title="Attrition Cohort Heatmap (%)", height=560)
        st.plotly_chart(fig_coh, use_container_width=True)
    else:
        st.info("ستون‌های Age/YearsAtCompany/_Attrition لازم است.")


