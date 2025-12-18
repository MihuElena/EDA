import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr

# =========================
# Config + UI
# =========================
st.set_page_config(page_title="EDA Explorer", page_icon="📊", layout="wide")
st.title("📊 Analiză exploratorie a datelor")

st.markdown("""
<style>
.block-container { padding-top: 4.5rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file: io.BytesIO, file_type: str, csv_sep: str, excel_sheet: str | None) -> pd.DataFrame:
    if file_type == "csv":
        df = pd.read_csv(uploaded_file, sep=csv_sep)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=excel_sheet if excel_sheet else 0)
    return df


def try_parse_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that look numeric (e.g., '1 234', '12%', '10,5') to numeric."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns.tolist()

    for col in obj_cols:
        s = out[col]
        if s.dropna().empty:
            continue

        # Heuristic: if extremely high cardinality, likely text -> skip
        non_na = s.dropna()
        if non_na.nunique() > max(50, int(0.4 * len(non_na))):
            continue

        cleaned = (
            s.astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")

        ok_ratio = numeric.notna().sum() / non_na.shape[0] if non_na.shape[0] else 0
        if ok_ratio >= 0.85:
            out[col] = numeric

    return out


def detect_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Optional: low-cardinality numerics treated as categorical (for tab 4)
    for col in numeric_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique > 0 and nunique <= min(20, max(5, int(0.01 * len(df)))):
            if col not in cat_cols:
                cat_cols.append(col)

    cat_cols = list(dict.fromkeys(cat_cols))
    return numeric_cols, cat_cols


def safe_min_max(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None, None
    return float(s.min()), float(s.max())


def apply_filters(df: pd.DataFrame, numeric_filters: dict, categorical_filters: dict) -> pd.DataFrame:
    filtered = df.copy()

    # numeric_filters: {col: (min_val, max_val)}
    for col, (lo, hi) in numeric_filters.items():
        if col in filtered.columns:
            filtered = filtered[filtered[col].between(lo, hi, inclusive="both") | filtered[col].isna()]

    # categorical_filters: {col: {"values": set([...]), "include_na": bool}}
    for col, payload in categorical_filters.items():
        if col not in filtered.columns:
            continue
        values = payload.get("values", None)
        include_na = payload.get("include_na", False)

        if values is None:
            continue

        mask = filtered[col].astype("string").isin(list(values))
        if include_na:
            mask = mask | filtered[col].isna()
        filtered = filtered[mask]

    return filtered


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss_cnt = df.isna().sum()
    miss_pct = (miss_cnt / len(df) * 100).replace([np.inf, -np.inf], np.nan)
    nunique = df.nunique(dropna=True)
    rep = pd.DataFrame(
        {
            "Coloană": df.columns,
            "Tip": [str(t) for t in df.dtypes],
            "Valori lipsă": miss_cnt.values,
            "Procent lipsă (%)": miss_pct.values,
            "Valori unice": nunique.values,
        }
    ).sort_values("Valori lipsă", ascending=False)
    return rep


def iqr_outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 4:
            rows.append({"Coloană": col, "Outlieri (IQR)": 0, "Procent outlieri (%)": 0.0, "Lower": np.nan, "Upper": np.nan})
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = (outliers / n * 100) if n else 0.0

        rows.append({"Coloană": col, "Outlieri (IQR)": int(outliers), "Procent outlieri (%)": float(pct), "Lower": float(lower), "Upper": float(upper)})

    return pd.DataFrame(rows).sort_values("Procent outlieri (%)", ascending=False)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =========================
# Sidebar upload
# =========================
st.sidebar.header("⚙️ Setări & Încărcare")

uploaded = st.sidebar.file_uploader("Încarcă fișier CSV sau Excel", type=["csv", "xlsx", "xls"])
csv_sep = st.sidebar.selectbox("Separator CSV", options=[",", ";", "\t", "|"], index=0)
parse_numeric_like = st.sidebar.checkbox("Conversie automată pentru coloane numeric-like", value=True)

excel_sheet = None
file_type = None

if uploaded is None:
    st.info("Încarcă un fișier CSV sau Excel din sidebar ca să începi.")
    st.stop()

name_lower = uploaded.name.lower()
if name_lower.endswith(".csv"):
    file_type = "csv"
elif name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
    file_type = "excel"
else:
    st.error("Tip de fișier necunoscut.")
    st.stop()

if file_type == "excel":
    excel_sheet = st.sidebar.text_input("Sheet (opțional)", value="", placeholder="Ex: Sheet1").strip() or None

# =========================
# Load + validate
# =========================
try:
    df_original = load_dataset(uploaded, file_type, csv_sep, excel_sheet)
    if parse_numeric_like:
        df_original = try_parse_numeric_like_columns(df_original)

    st.success(f"✅ Fișier citit corect: **{uploaded.name}**")
except Exception as e:
    st.error(f"❌ Eroare la citirea fișierului: {e}")
    st.stop()

if df_original is None or df_original.shape[0] == 0:
    st.warning("Dataset-ul este gol sau nu a putut fi încărcat corect.")
    st.stop()

numeric_cols, cat_cols = detect_column_types(df_original)

# =========================
# Tabs (requirements)
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1) Încărcare & Filtrare",
        "2) Prezentare & Valori lipsă",
        "3) Numeric",
        "4) Categoric",
        "5) Corelații & Outlieri",
    ]
)

# =========================
# TAB 1 - Requirement 1
# Fix for slider min==max included
# =========================
with tab1:
    st.subheader("📥 Încărcare & Filtrare")
    st.write("Primele 10 rânduri din dataset:")
    st.dataframe(df_original.head(10), use_container_width=True)

    st.divider()
    st.markdown("### Filtrare")

    before_rows = len(df_original)

    left, right = st.columns([1, 1], gap="large")
    numeric_filters = {}
    categorical_filters = {}

    with left:
        st.markdown("#### Filtre numerice (slidere)")
        if len(numeric_cols) == 0:
            st.info("Nu există coloane numerice detectate.")
        else:
            selected_num = st.multiselect(
                "Selectează coloane numerice pe care vrei să le filtrezi:",
                options=numeric_cols,
                default=[],
            )
            for col in selected_num:
                lo, hi = safe_min_max(df_original[col])
                if lo is None or hi is None:
                    st.caption(f"• {col}: fără valori numerice valide.")
                    continue

                # 🔧 Fix for Streamlit slider error (min must be < max)
                if lo >= hi:
                    st.info(f"Coloana „{col}” are o singură valoare (min=max={lo}). Nu se poate filtra cu slider.")
                    numeric_filters[col] = (lo, hi)
                    continue

                step = (hi - lo) / 200.0
                if step <= 0 or not np.isfinite(step):
                    step = 1.0

                val = st.slider(
                    f"{col}",
                    min_value=float(lo),
                    max_value=float(hi),
                    value=(float(lo), float(hi)),
                    step=float(step),
                )
                numeric_filters[col] = val

    with right:
        st.markdown("#### Filtre categorice (multiselect)")
        cat_candidates = df_original.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if len(cat_candidates) == 0:
            st.info("Nu există coloane categorice detectate.")
        else:
            cat_col = st.selectbox("Selectează o coloană categorică pentru filtrare:", options=["(fără)"] + cat_candidates, index=0)
            if cat_col != "(fără)":
                vc = df_original[cat_col].astype("string").fillna("NaN").value_counts()
                top_n = st.slider("Top N valori afișate în listă", 10, 200, 50, step=10)
                options = vc.head(top_n).index.tolist()

                chosen = st.multiselect(
                    f"Valori permise pentru **{cat_col}**:",
                    options=options,
                    default=options,
                )
                include_na = st.checkbox("Include și valorile lipsă (NaN)", value=False)

                categorical_filters[cat_col] = {"values": set(chosen), "include_na": include_na}

    df_filtered = apply_filters(df_original, numeric_filters, categorical_filters)
    after_rows = len(df_filtered)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rânduri înainte", f"{before_rows:,}")
    c2.metric("Rânduri după", f"{after_rows:,}")
    c3.metric("Rânduri eliminate", f"{before_rows - after_rows:,}")

    st.markdown("### Dataframe filtrat")
    st.dataframe(df_filtered, use_container_width=True, height=420)

    st.download_button(
        "⬇️ Descarcă dataset filtrat (CSV)",
        data=df_to_csv_bytes(df_filtered),
        file_name="dataset_filtrat.csv",
        mime="text/csv",
    )

    st.session_state["df_filtered"] = df_filtered

# Work on filtered dataset in the rest
df_work = st.session_state.get("df_filtered", df_original)
numeric_cols_work, cat_cols_work = detect_column_types(df_work)

# =========================
# TAB 2 - Requirement 2
# =========================
with tab2:
    st.subheader("🔎 Prezentare & Valori lipsă")

    rows, cols = df_work.shape
    total_cells = rows * cols if rows and cols else 0
    missing_cells = int(df_work.isna().sum().sum())
    missing_pct_total = (missing_cells / total_cells * 100) if total_cells else 0.0
    dup_rows = int(df_work.duplicated().sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rânduri", f"{rows:,}")
    m2.metric("Coloane", f"{cols:,}")
    m3.metric("Celule lipsă (%)", f"{missing_pct_total:.2f}%")
    m4.metric("Rânduri duplicate", f"{dup_rows:,}")

    st.divider()
    st.markdown("### Tipuri de date & lipsuri")
    rep = missing_report(df_work)
    st.dataframe(rep, use_container_width=True, height=360)

    st.markdown("### Grafic valori lipsă")
    cols_with_missing = rep[rep["Valori lipsă"] > 0].copy()
    if len(cols_with_missing) == 0:
        st.success("✅ Nu există valori lipsă în dataset-ul curent.")
    else:
        fig = px.bar(
            cols_with_missing.sort_values("Procent lipsă (%)", ascending=True),
            x="Procent lipsă (%)",
            y="Coloană",
            orientation="h",
            title="Procent valori lipsă pe coloană",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Statistici descriptive (numerice)")
    if len(numeric_cols_work) == 0:
        st.info("Nu există coloane numerice pentru statistici descriptive.")
    else:
        st.dataframe(df_work[numeric_cols_work].describe().T, use_container_width=True)

# =========================
# TAB 3 - Requirement 3
# =========================
with tab3:
    st.subheader("📈 Analiză pe coloană numerică")

    if len(numeric_cols_work) == 0:
        st.warning("Nu există coloane numerice detectate.")
    else:
        col = st.selectbox("Selectează o coloană numerică:", options=numeric_cols_work)
        bins = st.slider("Număr de bins pentru histogramă", 10, 100, 30)

        series = df_work[col].dropna()
        if series.empty:
            st.info("Coloana selectată nu are valori numerice valide (după filtrare).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Medie", f"{series.mean():.4g}")
            c2.metric("Mediană", f"{series.median():.4g}")
            c3.metric("Deviație standard", f"{series.std(ddof=1):.4g}")

            left, right = st.columns([1.2, 1], gap="large")
            with left:
                fig = px.histogram(df_work, x=col, nbins=bins, title=f"Histogramă: {col}")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                fig2 = px.box(df_work, y=col, points="outliers", title=f"Box plot: {col}")
                st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 4 - Requirement 4
# =========================
with tab4:
    st.subheader("📊 Analiză pe coloană categorică")

    cat_candidates = df_work.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # add low-cardinality numeric candidates
    for c in cat_cols_work:
        if c in df_work.columns and c not in cat_candidates:
            cat_candidates.append(c)

    if len(cat_candidates) == 0:
        st.warning("Nu există coloane categorice detectate.")
    else:
        cat_col = st.selectbox("Selectează o coloană categorică:", options=cat_candidates)
        top_n = st.slider("Top N categorii afișate (restul -> Others)", 5, 50, 15)

        vc = df_work[cat_col].astype("string").fillna("NaN").value_counts(dropna=False)
        total = int(vc.sum())

        if len(vc) > top_n:
            top = vc.head(top_n)
            others = vc.iloc[top_n:].sum()
            vc_plot = pd.concat([top, pd.Series({"Others": others})])
        else:
            vc_plot = vc

        fig = px.bar(
            x=vc_plot.index.astype(str),
            y=vc_plot.values,
            title=f"Count plot: {cat_col}",
            labels={"x": "Categorie", "y": "Frecvență"},
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        freq_df = pd.DataFrame({"Categorie": vc.index.astype(str), "Frecvență": vc.values})
        freq_df["Procent (%)"] = (freq_df["Frecvență"] / total * 100).round(2)

        st.markdown("### Tabel frecvențe (absolut + procente)")
        st.dataframe(freq_df, use_container_width=True, height=380)

# =========================
# TAB 5 - Requirement 5
# (No statsmodels used; Plotly trendline removed to avoid statsmodels dependency)
# =========================
with tab5:
    st.subheader("🧭 Corelații & Outlieri (IQR)")

    if len(numeric_cols_work) < 2:
        st.warning("Ai nevoie de cel puțin 2 coloane numerice pentru corelații.")
    else:
        st.markdown("### Matrice de corelație (Pearson)")
        corr = df_work[numeric_cols_work].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Heatmap corelații (Pearson)")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Scatter plot + coeficient Pearson")

        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("Variabila X:", options=numeric_cols_work, key="x_scatter")
        with c2:
            y_opts = [c for c in numeric_cols_work if c != x_col]
            y_col = st.selectbox("Variabila Y:", options=y_opts, key="y_scatter")

        pair = df_work[[x_col, y_col]].dropna()
        if len(pair) < 3:
            st.info("Nu sunt suficiente valori comune (non-NaN) pentru a calcula Pearson.")
        else:
            r, p = pearsonr(pair[x_col], pair[y_col])
            m1, m2 = st.columns(2)
            m1.metric("Coeficient Pearson r", f"{r:.4f}")
            m2.metric("p-value", f"{p:.4g}")

            # 🔧 No trendline="ols" (that triggers statsmodels)
            fig2 = px.scatter(pair, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}")
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.markdown("### Detecție outlieri (metoda IQR)")
        out_df = iqr_outlier_summary(df_work, numeric_cols_work)
        st.dataframe(out_df, use_container_width=True, height=360)

        st.markdown("### Vizualizare outlieri pe grafic")
        out_col = st.selectbox("Selectează coloana:", options=numeric_cols_work, key="out_col")

        s = df_work[out_col].dropna()
        if len(s) < 4:
            st.info("Prea puține valori pentru IQR pe coloana selectată.")
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mode = st.radio("Tip grafic:", options=["Box plot", "Index vs valoare"], horizontal=True)
            if mode == "Box plot":
                figb = px.box(df_work, y=out_col, points="outliers", title=f"Box plot cu outlieri: {out_col}")
                figb.add_hline(y=lower, line_dash="dash", annotation_text="Lower IQR", annotation_position="bottom right")
                figb.add_hline(y=upper, line_dash="dash", annotation_text="Upper IQR", annotation_position="top right")
                st.plotly_chart(figb, use_container_width=True)
            else:
                tmp = df_work[[out_col]].copy()
                tmp["_index_"] = np.arange(len(tmp))
                tmp["_is_outlier_"] = (tmp[out_col] < lower) | (tmp[out_col] > upper)
                figo = px.scatter(
                    tmp,
                    x="_index_",
                    y=out_col,
                    color="_is_outlier_",
                    title=f"Outlieri evidențiați (IQR): {out_col}",
                    labels={"_index_": "Index", "_is_outlier_": "Outlier"},
                )
                figo.add_hline(y=lower, line_dash="dash")
                figo.add_hline(y=upper, line_dash="dash")
                st.plotly_chart(figo, use_container_width=True)

            out_count = int(((df_work[out_col] < lower) | (df_work[out_col] > upper)).sum())
            out_pct = (out_count / len(df_work) * 100) if len(df_work) else 0.0
            st.info(f"🔎 **{out_col}** → Outlieri: **{out_count}** ( **{out_pct:.2f}%** )")

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<span class="small-note">Tip: aplică filtre în Tab 1 — restul tab-urilor lucrează pe dataset-ul filtrat.</span>', unsafe_allow_html=True)
