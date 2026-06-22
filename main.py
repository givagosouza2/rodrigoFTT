import io
import math
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


st.set_page_config(page_title="FTT Handedness Analyzer", page_icon="🖐️", layout="wide")
st.title("🖐️ Análise de dominância manual e lateralidade no FTT")
st.caption(
    "Desenho misto 2 × 2: lateralidade (destro/canhoto) × condição da mão "
    "(dominante/não dominante), com modelo misto, pós-hoc, tamanhos de efeito e índice de assimetria."
)


def normalize_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    repl = str.maketrans("ãáàâéêíóôõúç", "aaaaeeiooouc")
    s = s.translate(repl)
    return re.sub(r"\s+", " ", s)


def load_data(uploaded_file, decimal=",", sheet_name=None):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, decimal=decimal, sep=None, engine="python")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, decimal=decimal)
    if name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)
    raise ValueError("Formato não suportado.")


def coerce_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def map_laterality(series):
    def f(v):
        s = normalize_text(v)
        if pd.isna(s):
            return np.nan
        if "dest" in s or "right" in s or s in {"d", "r", "direita"}:
            return "Right-handed"
        if "canh" in s or "left" in s or s in {"c", "l", "esquerda"}:
            return "Left-handed"
        return np.nan
    return series.map(f)


def map_tested_hand(series):
    def f(v):
        s = normalize_text(v)
        if pd.isna(s):
            return np.nan
        if "direit" in s or "right" in s or s in {"d", "r"}:
            return "Right"
        if "esquerd" in s or "left" in s or s in {"e", "l"}:
            return "Left"
        return np.nan
    return series.map(f)


def map_hand_condition(series):
    def f(v):
        s = normalize_text(v)
        if pd.isna(s):
            return np.nan
        if ("nao" in s and "domin" in s) or "non-domin" in s or "nondomin" in s or s == "nd":
            return "Non-dominant"
        if "domin" in s or s == "dom":
            return "Dominant"
        return np.nan
    return series.map(f)


def derive_hand_condition(laterality, tested_hand):
    if pd.isna(laterality) or pd.isna(tested_hand):
        return np.nan
    if laterality == "Right-handed":
        return "Dominant" if tested_hand == "Right" else "Non-dominant"
    return "Dominant" if tested_hand == "Left" else "Non-dominant"


def shapiro_safe(x):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return np.nan, np.nan
    if len(x) > 5000:
        x = x.sample(5000, random_state=123)
    try:
        return stats.shapiro(x)
    except Exception:
        return np.nan, np.nan


def hedges_g(x1, x2):
    x1 = np.asarray(pd.Series(x1).dropna(), dtype=float)
    x2 = np.asarray(pd.Series(x2).dropna(), dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    sp = math.sqrt(((n1 - 1)*v1 + (n2 - 1)*v2)/(n1+n2-2))
    if sp == 0:
        return np.nan
    d = (np.mean(x1)-np.mean(x2))/sp
    df = n1+n2-2
    J = 1 - 3/(4*df-1) if df > 1 else 1
    return J*d


def cohen_dz(x1, x2):
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    mask = np.isfinite(x1) & np.isfinite(x2)
    diff = x1[mask] - x2[mask]
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return np.nan
    return np.mean(diff)/np.std(diff, ddof=1)


def descriptive_table(df, outcome):
    base = (
        df.groupby(["Laterality", "HandCondition"], observed=True)[outcome]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    qs = (
        df.groupby(["Laterality", "HandCondition"], observed=True)[outcome]
        .quantile([0.25, 0.75]).unstack().reset_index()
        .rename(columns={0.25: "q1", 0.75: "q3"})
    )
    return base.merge(qs, on=["Laterality", "HandCondition"], how="left")


def fit_model(df, outcome):
    d = df[["ID", "Laterality", "HandCondition", outcome]].dropna().copy()
    d["Laterality"] = pd.Categorical(d["Laterality"], ["Right-handed", "Left-handed"])
    d["HandCondition"] = pd.Categorical(d["HandCondition"], ["Non-dominant", "Dominant"])

    formula = (
        f'Q("{outcome}") ~ '
        'C(Laterality, Treatment(reference="Right-handed")) * '
        'C(HandCondition, Treatment(reference="Non-dominant"))'
    )

    warning = None
    try:
        model = smf.mixedlm(formula, d, groups=d["ID"], re_formula="1")
        result = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        method = "Linear mixed-effects model"
    except Exception as e:
        warning = str(e)
        model = smf.ols(formula, d)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": d["ID"]})
        method = "OLS with participant-clustered robust standard errors"

    ci = result.conf_int()
    coef = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "SE": result.bse.values,
        "statistic": result.tvalues.values,
        "p_value": result.pvalues.values,
        "CI95_low": ci.iloc[:, 0].values,
        "CI95_high": ci.iloc[:, 1].values,
    })

    term_map = {
        "Laterality": 'C(Laterality, Treatment(reference="Right-handed"))[T.Left-handed]',
        "Hand condition": 'C(HandCondition, Treatment(reference="Non-dominant"))[T.Dominant]',
        "Interaction": (
            'C(Laterality, Treatment(reference="Right-handed"))[T.Left-handed]:'
            'C(HandCondition, Treatment(reference="Non-dominant"))[T.Dominant]'
        ),
    }
    rows = []
    for effect, term in term_map.items():
        r = coef[coef["term"] == term]
        if not r.empty:
            rows.append({
                "effect": effect,
                "estimate": r["estimate"].iloc[0],
                "SE": r["SE"].iloc[0],
                "statistic": r["statistic"].iloc[0],
                "p_value": r["p_value"].iloc[0],
                "CI95_low": r["CI95_low"].iloc[0],
                "CI95_high": r["CI95_high"].iloc[0],
            })
    return result, pd.DataFrame(rows), coef, method, warning


def posthoc_tests(df, outcome, correction):
    rows = []

    for lat in ["Right-handed", "Left-handed"]:
        sub = df[df["Laterality"] == lat]
        wide = sub.pivot_table(index="ID", columns="HandCondition", values=outcome, aggfunc="mean")
        if {"Dominant", "Non-dominant"}.issubset(wide.columns):
            pair = wide[["Dominant", "Non-dominant"]].dropna()
            if len(pair) >= 2:
                diff = pair["Dominant"] - pair["Non-dominant"]
                _, pnorm = shapiro_safe(diff)
                if pd.notna(pnorm) and pnorm >= 0.05:
                    stat, p = stats.ttest_rel(pair["Dominant"], pair["Non-dominant"])
                    test = "Paired t-test"
                else:
                    stat, p = stats.wilcoxon(pair["Dominant"], pair["Non-dominant"])
                    test = "Wilcoxon signed-rank"
                rows.append({
                    "comparison": f"{lat}: Dominant vs Non-dominant",
                    "test": test,
                    "n": len(pair),
                    "mean_difference": diff.mean(),
                    "effect_size": cohen_dz(pair["Dominant"], pair["Non-dominant"]),
                    "effect_size_name": "Cohen dz",
                    "statistic": stat,
                    "p_raw": p,
                })

    for hand in ["Dominant", "Non-dominant"]:
        r = df[(df["Laterality"] == "Right-handed") & (df["HandCondition"] == hand)][outcome].dropna()
        l = df[(df["Laterality"] == "Left-handed") & (df["HandCondition"] == hand)][outcome].dropna()
        if len(r) >= 2 and len(l) >= 2:
            _, pr = shapiro_safe(r)
            _, pl = shapiro_safe(l)
            if pd.notna(pr) and pd.notna(pl) and pr >= 0.05 and pl >= 0.05:
                stat, p = stats.ttest_ind(r, l, equal_var=False)
                test = "Welch independent t-test"
            else:
                stat, p = stats.mannwhitneyu(r, l, alternative="two-sided")
                test = "Mann–Whitney U"
            rows.append({
                "comparison": f"{hand}: Right-handed vs Left-handed",
                "test": test,
                "n": len(r)+len(l),
                "mean_difference": r.mean()-l.mean(),
                "effect_size": hedges_g(r, l),
                "effect_size_name": "Hedges g",
                "statistic": stat,
                "p_raw": p,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adjusted"] = multipletests(out["p_raw"], method=correction)[1]
    return out


def asymmetry_analysis(df, outcome):
    wide = df.pivot_table(
        index=["ID", "Laterality"],
        columns="HandCondition",
        values=outcome,
        aggfunc="mean"
    ).reset_index()

    if not {"Dominant", "Non-dominant"}.issubset(wide.columns):
        return pd.DataFrame(), pd.DataFrame()

    denom = wide["Dominant"] + wide["Non-dominant"]
    wide["AsymmetryIndex_percent"] = np.where(
        denom != 0,
        ((wide["Dominant"] - wide["Non-dominant"]) / denom)*100,
        np.nan
    )
    wide["AbsoluteAsymmetry_percent"] = wide["AsymmetryIndex_percent"].abs()

    r = wide.loc[wide["Laterality"] == "Right-handed", "AsymmetryIndex_percent"].dropna()
    l = wide.loc[wide["Laterality"] == "Left-handed", "AsymmetryIndex_percent"].dropna()
    rows = []
    if len(r) >= 2 and len(l) >= 2:
        _, pr = shapiro_safe(r)
        _, pl = shapiro_safe(l)
        if pd.notna(pr) and pd.notna(pl) and pr >= 0.05 and pl >= 0.05:
            stat, p = stats.ttest_ind(r, l, equal_var=False)
            test = "Welch independent t-test"
        else:
            stat, p = stats.mannwhitneyu(r, l, alternative="two-sided")
            test = "Mann–Whitney U"
        rows.append({
            "comparison": "Asymmetry index: Right-handed vs Left-handed",
            "test": test,
            "mean_right_handed": r.mean(),
            "mean_left_handed": l.mean(),
            "difference": r.mean()-l.mean(),
            "effect_size_Hedges_g": hedges_g(r, l),
            "statistic": stat,
            "p_value": p,
        })
    return wide, pd.DataFrame(rows)


def interaction_plot(df, outcome):
    s = (
        df.groupby(["Laterality", "HandCondition"], observed=True)[outcome]
        .agg(["mean", "std", "count"]).reset_index()
    )
    s["se"] = s["std"]/np.sqrt(s["count"])
    order = ["Non-dominant", "Dominant"]
    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(7, 5))
    for lat in ["Right-handed", "Left-handed"]:
        z = s[s["Laterality"] == lat].set_index("HandCondition").reindex(order)
        ax.errorbar(x, z["mean"], yerr=z["se"], marker="o", capsize=4, label=lat)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Hand condition")
    ax.set_ylabel(outcome)
    ax.set_title(f"Interaction plot: {outcome}")
    ax.legend()
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("Configurações")
    decimal = st.selectbox("Separador decimal", [",", "."], index=0)
    correction = st.selectbox("Correção para múltiplas comparações", ["holm", "fdr_bh", "bonferroni"])
    alpha = st.number_input("Nível de significância", 0.001, 0.20, 0.05, 0.001)

uploaded = st.file_uploader("Carregue um arquivo CSV, XLSX ou XLS", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info(
        "Formato recomendado: uma linha por participante e por mão, com ID, lateralidade, "
        "mão testada ou condição da mão, e as variáveis numéricas do FTT."
    )
    st.stop()

sheet_name = None
if uploaded.name.lower().endswith((".xlsx", ".xls")):
    uploaded.seek(0)
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("Planilha", xls.sheet_names)
    uploaded.seek(0)

try:
    raw = load_data(uploaded, decimal, sheet_name)
except Exception as e:
    st.error(f"Erro ao carregar arquivo: {e}")
    st.stop()

st.subheader("Prévia")
st.dataframe(raw.head(20), use_container_width=True)

cols = list(raw.columns)
options = ["— selecionar —"] + cols

st.subheader("Mapeamento das colunas")
c1, c2, c3 = st.columns(3)

with c1:
    id_col = st.selectbox("ID do participante", options)
with c2:
    lat_col = st.selectbox("Lateralidade autorreferida", options)
with c3:
    mode = st.radio(
        "Informação da mão",
        ["Mão testada (direita/esquerda)", "Condição pronta (dominante/não dominante)"]
    )

if mode.startswith("Mão testada"):
    hand_col = st.selectbox("Coluna da mão testada", options)
    condition_col = None
else:
    condition_col = st.selectbox("Coluna da condição da mão", options)
    hand_col = None

if id_col == options[0] or lat_col == options[0]:
    st.stop()
if hand_col == options[0] if hand_col is not None else False:
    st.stop()
if condition_col == options[0] if condition_col is not None else False:
    st.stop()

data = raw.copy()
data["ID"] = data[id_col].astype(str).str.strip()
data["Laterality"] = map_laterality(data[lat_col])

if hand_col is not None:
    data["TestedHand"] = map_tested_hand(data[hand_col])
    data["HandCondition"] = [
        derive_hand_condition(a, b) for a, b in zip(data["Laterality"], data["TestedHand"])
    ]
else:
    data["HandCondition"] = map_hand_condition(data[condition_col])

n_bad_lat = data["Laterality"].isna().sum()
n_bad_hand = data["HandCondition"].isna().sum()
if n_bad_lat or n_bad_hand:
    st.warning(
        f"{n_bad_lat} linhas sem lateralidade reconhecida e {n_bad_hand} sem condição de mão reconhecida serão excluídas."
    )

data = data.dropna(subset=["ID", "Laterality", "HandCondition"]).copy()

excluded = {id_col, lat_col, hand_col, condition_col, "ID", "Laterality", "TestedHand", "HandCondition", None}
numeric_candidates = []
for c in data.columns:
    if c in excluded:
        continue
    converted = coerce_numeric(data[c])
    if converted.notna().sum() >= max(3, int(0.5*len(data))):
        numeric_candidates.append(c)

selected = st.multiselect("Parâmetros do FTT", numeric_candidates, default=numeric_candidates[:1])
if not selected:
    st.stop()

for c in selected:
    data[c] = coerce_numeric(data[c])

st.write("Distribuição dos registros")
st.dataframe(pd.crosstab(data["Laterality"], data["HandCondition"]), use_container_width=True)

complete = data.groupby("ID")["HandCondition"].nunique()
st.caption(f"Participantes com ambas as mãos: {(complete >= 2).sum()} de {complete.size}.")

all_desc, all_effects, all_posthoc, all_asym = [], [], [], []
tabs = st.tabs(selected)

for tab, outcome in zip(tabs, selected):
    with tab:
        d = data[["ID", "Laterality", "HandCondition", outcome]].dropna().copy()
        st.markdown(f"### {outcome}")

        desc = descriptive_table(d, outcome)
        desc.insert(0, "outcome", outcome)
        all_desc.append(desc)
        st.markdown("#### Estatística descritiva")
        st.dataframe(desc, use_container_width=True)

        st.markdown("#### Modelo principal")
        try:
            result, effects, coef, method, warning = fit_model(d, outcome)
            effects.insert(0, "outcome", outcome)
            all_effects.append(effects)
            st.caption(f"Método: {method}")
            if warning and method.startswith("OLS"):
                st.warning("O modelo misto não convergiu; foi usado o fallback robusto.")
            st.dataframe(effects, use_container_width=True)

            r = effects[effects["effect"] == "Interaction"]
            if not r.empty:
                p = float(r["p_value"].iloc[0])
                if p < alpha:
                    st.success(f"Interação significativa (p = {p:.4g}).")
                else:
                    st.info(f"Interação não significativa (p = {p:.4g}).")

            with st.expander("Coeficientes completos"):
                st.dataframe(coef, use_container_width=True)
                st.text(str(result.summary()))
        except Exception as e:
            st.error(f"Falha no modelo: {e}")

        st.pyplot(interaction_plot(d, outcome))

        st.markdown("#### Pós-hoc")
        ph = posthoc_tests(d, outcome, correction)
        if not ph.empty:
            ph.insert(0, "outcome", outcome)
            all_posthoc.append(ph)
            st.dataframe(ph, use_container_width=True)

        st.markdown("#### Índice de assimetria")
        asym_data, asym_test = asymmetry_analysis(d, outcome)
        if not asym_data.empty:
            st.dataframe(asym_data, use_container_width=True)
        if not asym_test.empty:
            asym_test.insert(0, "outcome", outcome)
            all_asym.append(asym_test)
            st.dataframe(asym_test, use_container_width=True)

        st.markdown("#### Normalidade por célula")
        rows = []
        for (lat, hand), sub in d.groupby(["Laterality", "HandCondition"]):
            W, p = shapiro_safe(sub[outcome])
            rows.append({"Laterality": lat, "HandCondition": hand, "n": len(sub), "Shapiro_W": W, "Shapiro_p": p})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

def combine(items):
    return pd.concat(items, ignore_index=True) if items else pd.DataFrame()

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    data.to_excel(writer, sheet_name="Dados_processados", index=False)
    combine(all_desc).to_excel(writer, sheet_name="Descritivos", index=False)
    combine(all_effects).to_excel(writer, sheet_name="Efeitos_modelo", index=False)
    combine(all_posthoc).to_excel(writer, sheet_name="Pos_hoc", index=False)
    combine(all_asym).to_excel(writer, sheet_name="Assimetria", index=False)

st.download_button(
    "⬇️ Baixar resultados em Excel",
    buffer.getvalue(),
    file_name="analise_FTT_lateralidade_dominancia.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("Formato recomendado"):
    st.markdown(
        """
        | ID | Lateralidade | Mão_testada | Número_toques | Intervalo_médio | Área_elipse |
        |---|---|---|---:|---:|---:|
        | 001 | Destro | Direita | 176 | 0.171 | 4605 |
        | 001 | Destro | Esquerda | 156 | 0.194 | 7511 |
        | 002 | Canhoto | Direita | 157 | 0.190 | 7144 |
        | 002 | Canhoto | Esquerda | 168 | 0.178 | 6982 |
        """
    )

st.info(
    "A interpretação principal deve começar pela interação lateralidade × condição da mão. "
    "Os pós-hoc ajudam a localizar diferenças quando essa interação é significativa."
)
