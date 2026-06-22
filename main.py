import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


st.set_page_config(
    page_title="FTT – Lateralidade e dominância",
    page_icon="🖐️",
    layout="wide",
)

st.title("🖐️ Efeitos da lateralidade e da dominância manual")
st.caption(
    "Aplicativo adaptado para arquivos com quatro colunas: "
    "destros dominante, destros não dominante, canhotos dominante e canhotos não dominante."
)


# ============================================================
# Leitura e preparação
# ============================================================

def read_uploaded_file(uploaded_file, decimal=",", sheet_name=None):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        uploaded_file.seek(0)
        try:
            return pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
                decimal=decimal,
                encoding="utf-8-sig",
            )
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file,
                sep=",",
                decimal=decimal,
                encoding="utf-8-sig",
            )

    if name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)

    raise ValueError("Use arquivo CSV, XLSX ou XLS.")


def to_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()

    # Aceita vírgula ou ponto decimal
    if s.str.contains(",", regex=False).any():
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def wide_to_long(
    df,
    right_dom_col,
    right_nondom_col,
    left_dom_col,
    left_nondom_col,
):
    """
    Converte as quatro colunas em formato longo.

    As linhas dos destros são pareadas entre as colunas 1 e 2.
    As linhas dos canhotos são pareadas entre as colunas 3 e 4.
    Linhas totalmente vazias dentro de um grupo são removidas.
    """

    right = pd.DataFrame({
        "Dominant": to_numeric(df[right_dom_col]),
        "Non-dominant": to_numeric(df[right_nondom_col]),
    })

    left = pd.DataFrame({
        "Dominant": to_numeric(df[left_dom_col]),
        "Non-dominant": to_numeric(df[left_nondom_col]),
    })

    # Mantém apenas participantes com as duas mãos disponíveis
    right_complete = right.dropna(subset=["Dominant", "Non-dominant"]).copy()
    left_complete = left.dropna(subset=["Dominant", "Non-dominant"]).copy()

    right_complete["ID"] = [
        f"R_{i+1:03d}" for i in range(len(right_complete))
    ]
    right_complete["Laterality"] = "Right-handed"

    left_complete["ID"] = [
        f"L_{i+1:03d}" for i in range(len(left_complete))
    ]
    left_complete["Laterality"] = "Left-handed"

    right_long = right_complete.melt(
        id_vars=["ID", "Laterality"],
        value_vars=["Dominant", "Non-dominant"],
        var_name="HandCondition",
        value_name="Outcome",
    )

    left_long = left_complete.melt(
        id_vars=["ID", "Laterality"],
        value_vars=["Dominant", "Non-dominant"],
        var_name="HandCondition",
        value_name="Outcome",
    )

    long_df = pd.concat([right_long, left_long], ignore_index=True)

    return long_df, right, left, right_complete, left_complete


# ============================================================
# Estatística
# ============================================================

def shapiro_safe(x):
    x = pd.Series(x).dropna()

    if len(x) < 3:
        return np.nan, np.nan

    try:
        return stats.shapiro(x)
    except Exception:
        return np.nan, np.nan


def hedges_g(x1, x2):
    x1 = np.asarray(pd.Series(x1).dropna(), dtype=float)
    x2 = np.asarray(pd.Series(x2).dropna(), dtype=float)

    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return np.nan

    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)

    pooled = math.sqrt(
        ((n1 - 1) * v1 + (n2 - 1) * v2) /
        (n1 + n2 - 2)
    )

    if pooled == 0:
        return np.nan

    d = (np.mean(x1) - np.mean(x2)) / pooled
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1) if df > 1 else 1

    return correction * d


def cohen_dz(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    mask = np.isfinite(x1) & np.isfinite(x2)
    difference = x1[mask] - x2[mask]

    if len(difference) < 2:
        return np.nan

    sd_difference = np.std(difference, ddof=1)

    if sd_difference == 0:
        return np.nan

    return np.mean(difference) / sd_difference


def descriptive_statistics(long_df):
    grouped = (
        long_df.groupby(
            ["Laterality", "HandCondition"],
            observed=True,
        )["Outcome"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    quartiles = (
        long_df.groupby(
            ["Laterality", "HandCondition"],
            observed=True,
        )["Outcome"]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q1", 0.75: "q3"})
    )

    return grouped.merge(
        quartiles,
        on=["Laterality", "HandCondition"],
        how="left",
    )


def fit_mixed_model(long_df):
    """
    Modelo:
    Outcome ~ Laterality * HandCondition + (1 | ID)

    Categorias de referência:
    Right-handed e Non-dominant
    """

    data = long_df.copy()

    data["Laterality"] = pd.Categorical(
        data["Laterality"],
        categories=["Right-handed", "Left-handed"],
    )

    data["HandCondition"] = pd.Categorical(
        data["HandCondition"],
        categories=["Non-dominant", "Dominant"],
    )

    formula = (
        'Outcome ~ '
        'C(Laterality, Treatment(reference="Right-handed")) * '
        'C(HandCondition, Treatment(reference="Non-dominant"))'
    )

    warning_message = None

    try:
        model = smf.mixedlm(
            formula,
            data,
            groups=data["ID"],
            re_formula="1",
        )

        result = model.fit(
            reml=False,
            method="lbfgs",
            maxiter=1000,
            disp=False,
        )

        method = "Modelo linear misto"

    except Exception as error:
        warning_message = str(error)

        model = smf.ols(formula, data)

        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": data["ID"]},
        )

        method = (
            "Regressão linear com erros-padrão robustos "
            "agrupados por participante"
        )

    confidence_interval = result.conf_int()

    coefficients = pd.DataFrame({
        "Termo": result.params.index,
        "Estimativa": result.params.values,
        "Erro-padrão": result.bse.values,
        "Estatística": result.tvalues.values,
        "p": result.pvalues.values,
        "IC95% inferior": confidence_interval.iloc[:, 0].values,
        "IC95% superior": confidence_interval.iloc[:, 1].values,
    })

    terms = {
        "Lateralidade": (
            'C(Laterality, Treatment(reference="Right-handed"))'
            '[T.Left-handed]'
        ),
        "Dominância manual": (
            'C(HandCondition, Treatment(reference="Non-dominant"))'
            '[T.Dominant]'
        ),
        "Interação lateralidade × dominância": (
            'C(Laterality, Treatment(reference="Right-handed"))'
            '[T.Left-handed]:'
            'C(HandCondition, Treatment(reference="Non-dominant"))'
            '[T.Dominant]'
        ),
    }

    effects = []

    for effect_name, term_name in terms.items():
        row = coefficients[coefficients["Termo"] == term_name]

        if not row.empty:
            effects.append({
                "Efeito": effect_name,
                "Estimativa": row["Estimativa"].iloc[0],
                "Erro-padrão": row["Erro-padrão"].iloc[0],
                "Estatística": row["Estatística"].iloc[0],
                "p": row["p"].iloc[0],
                "IC95% inferior": row["IC95% inferior"].iloc[0],
                "IC95% superior": row["IC95% superior"].iloc[0],
            })

    return (
        result,
        pd.DataFrame(effects),
        coefficients,
        method,
        warning_message,
    )


def posthoc_tests(right_complete, left_complete, correction_method):
    rows = []

    # Comparações pareadas dentro de cada lateralidade
    for group_name, group_df in [
        ("Destros", right_complete),
        ("Canhotos", left_complete),
    ]:
        dominant = group_df["Dominant"].to_numpy(dtype=float)
        nondominant = group_df["Non-dominant"].to_numpy(dtype=float)
        difference = dominant - nondominant

        _, normality_p = shapiro_safe(difference)

        if pd.notna(normality_p) and normality_p >= 0.05:
            statistic, p_value = stats.ttest_rel(
                dominant,
                nondominant,
            )
            test_name = "Teste t pareado"
        else:
            statistic, p_value = stats.wilcoxon(
                dominant,
                nondominant,
            )
            test_name = "Wilcoxon pareado"

        rows.append({
            "Comparação": f"{group_name}: dominante vs não dominante",
            "Teste": test_name,
            "n": len(group_df),
            "Média da diferença": np.mean(difference),
            "Tamanho de efeito": cohen_dz(dominant, nondominant),
            "Métrica de efeito": "Cohen dz",
            "Estatística": statistic,
            "p bruto": p_value,
        })

    # Comparações independentes entre lateralidades
    for condition in ["Dominant", "Non-dominant"]:
        right_values = right_complete[condition].dropna()
        left_values = left_complete[condition].dropna()

        _, p_right = shapiro_safe(right_values)
        _, p_left = shapiro_safe(left_values)

        if (
            pd.notna(p_right)
            and pd.notna(p_left)
            and p_right >= 0.05
            and p_left >= 0.05
        ):
            statistic, p_value = stats.ttest_ind(
                right_values,
                left_values,
                equal_var=False,
            )
            test_name = "Teste t de Welch"
        else:
            statistic, p_value = stats.mannwhitneyu(
                right_values,
                left_values,
                alternative="two-sided",
            )
            test_name = "Mann–Whitney"

        condition_pt = (
            "dominante"
            if condition == "Dominant"
            else "não dominante"
        )

        rows.append({
            "Comparação": (
                f"Destros vs canhotos: mão {condition_pt}"
            ),
            "Teste": test_name,
            "n": len(right_values) + len(left_values),
            "Média da diferença": (
                right_values.mean() - left_values.mean()
            ),
            "Tamanho de efeito": hedges_g(
                right_values,
                left_values,
            ),
            "Métrica de efeito": "Hedges g",
            "Estatística": statistic,
            "p bruto": p_value,
        })

    results = pd.DataFrame(rows)

    results["p ajustado"] = multipletests(
        results["p bruto"],
        method=correction_method,
    )[1]

    return results


def asymmetry_analysis(right_complete, left_complete):
    right = right_complete.copy()
    left = left_complete.copy()

    right["Índice de assimetria (%)"] = (
        (right["Dominant"] - right["Non-dominant"]) /
        (right["Dominant"] + right["Non-dominant"])
    ) * 100

    left["Índice de assimetria (%)"] = (
        (left["Dominant"] - left["Non-dominant"]) /
        (left["Dominant"] + left["Non-dominant"])
    ) * 100

    right["Assimetria absoluta (%)"] = (
        right["Índice de assimetria (%)"].abs()
    )

    left["Assimetria absoluta (%)"] = (
        left["Índice de assimetria (%)"].abs()
    )

    right_ai = right["Índice de assimetria (%)"].dropna()
    left_ai = left["Índice de assimetria (%)"].dropna()

    _, p_right = shapiro_safe(right_ai)
    _, p_left = shapiro_safe(left_ai)

    if (
        pd.notna(p_right)
        and pd.notna(p_left)
        and p_right >= 0.05
        and p_left >= 0.05
    ):
        statistic, p_value = stats.ttest_ind(
            right_ai,
            left_ai,
            equal_var=False,
        )
        test_name = "Teste t de Welch"
    else:
        statistic, p_value = stats.mannwhitneyu(
            right_ai,
            left_ai,
            alternative="two-sided",
        )
        test_name = "Mann–Whitney"

    comparison = pd.DataFrame([{
        "Teste": test_name,
        "Média destros": right_ai.mean(),
        "Média canhotos": left_ai.mean(),
        "Diferença entre médias": (
            right_ai.mean() - left_ai.mean()
        ),
        "Hedges g": hedges_g(right_ai, left_ai),
        "Estatística": statistic,
        "p": p_value,
    }])

    individual = pd.concat([
        right[
            [
                "ID",
                "Laterality",
                "Dominant",
                "Non-dominant",
                "Índice de assimetria (%)",
                "Assimetria absoluta (%)",
            ]
        ],
        left[
            [
                "ID",
                "Laterality",
                "Dominant",
                "Non-dominant",
                "Índice de assimetria (%)",
                "Assimetria absoluta (%)",
            ]
        ],
    ], ignore_index=True)

    return individual, comparison


def create_interaction_plot(long_df, outcome_name):
    summary = (
        long_df.groupby(
            ["Laterality", "HandCondition"],
            observed=True,
        )["Outcome"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    hand_order = ["Non-dominant", "Dominant"]
    x_positions = np.arange(len(hand_order))

    fig, ax = plt.subplots(figsize=(7, 5))

    for laterality in ["Right-handed", "Left-handed"]:
        group = (
            summary[summary["Laterality"] == laterality]
            .set_index("HandCondition")
            .reindex(hand_order)
        )

        ax.errorbar(
            x_positions,
            group["mean"],
            yerr=group["se"],
            marker="o",
            capsize=4,
            label=(
                "Destros"
                if laterality == "Right-handed"
                else "Canhotos"
            ),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Não dominante", "Dominante"])
    ax.set_xlabel("Condição da mão")
    ax.set_ylabel(outcome_name)
    ax.set_title(f"Interação: {outcome_name}")
    ax.legend()
    fig.tight_layout()

    return fig


# ============================================================
# Interface
# ============================================================

with st.sidebar:
    st.header("Configurações")

    decimal_separator = st.selectbox(
        "Separador decimal",
        [",", "."],
        index=0,
    )

    correction_method = st.selectbox(
        "Correção dos pós-hoc",
        ["holm", "fdr_bh", "bonferroni"],
        index=0,
    )

    alpha = st.number_input(
        "Nível de significância",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.001,
    )

uploaded_file = st.file_uploader(
    "Carregue o arquivo de quatro colunas",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info(
        "O arquivo deve conter quatro colunas, nesta lógica: "
        "destros dominante, destros não dominante, "
        "canhotos dominante e canhotos não dominante."
    )
    st.stop()

sheet_name = None

if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
    uploaded_file.seek(0)
    excel_file = pd.ExcelFile(uploaded_file)

    sheet_name = st.selectbox(
        "Planilha do Excel",
        excel_file.sheet_names,
    )

    uploaded_file.seek(0)

try:
    raw_data = read_uploaded_file(
        uploaded_file,
        decimal=decimal_separator,
        sheet_name=sheet_name,
    )
except Exception as error:
    st.error(f"Erro ao carregar o arquivo: {error}")
    st.stop()

if raw_data.shape[1] < 4:
    st.error(
        "O arquivo precisa ter pelo menos quatro colunas."
    )
    st.stop()

st.subheader("Prévia do arquivo")
st.dataframe(raw_data.head(20), use_container_width=True)

columns = list(raw_data.columns)

st.subheader("Mapeamento das quatro colunas")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    right_dom_col = st.selectbox(
        "1. Destros — mão dominante",
        columns,
        index=0,
    )

with col2:
    right_nondom_col = st.selectbox(
        "2. Destros — mão não dominante",
        columns,
        index=min(1, len(columns) - 1),
    )

with col3:
    left_dom_col = st.selectbox(
        "3. Canhotos — mão dominante",
        columns,
        index=min(2, len(columns) - 1),
    )

with col4:
    left_nondom_col = st.selectbox(
        "4. Canhotos — mão não dominante",
        columns,
        index=min(3, len(columns) - 1),
    )

selected_columns = {
    right_dom_col,
    right_nondom_col,
    left_dom_col,
    left_nondom_col,
}

if len(selected_columns) < 4:
    st.error("Cada condição deve usar uma coluna diferente.")
    st.stop()

default_outcome_name = Path(uploaded_file.name).stem

outcome_name = st.text_input(
    "Nome do parâmetro analisado",
    value=default_outcome_name,
)

(
    long_data,
    right_raw,
    left_raw,
    right_complete,
    left_complete,
) = wide_to_long(
    raw_data,
    right_dom_col,
    right_nondom_col,
    left_dom_col,
    left_nondom_col,
)

if len(right_complete) < 2 or len(left_complete) < 2:
    st.error(
        "São necessários pelo menos dois participantes completos "
        "em cada grupo."
    )
    st.stop()

st.success(
    f"Foram identificados {len(right_complete)} destros e "
    f"{len(left_complete)} canhotos com dados pareados das duas mãos."
)

excluded_right = (
    len(right_raw) -
    len(right_complete)
)

excluded_left = (
    len(left_raw) -
    len(left_complete)
)

if excluded_right > 0 or excluded_left > 0:
    st.warning(
        f"Foram excluídas {excluded_right} linhas incompletas dos destros "
        f"e {excluded_left} linhas incompletas dos canhotos. "
        "Linhas vazias ao final das colunas dos canhotos são esperadas "
        "quando os grupos têm tamanhos diferentes."
    )

with st.expander("Ver dados convertidos para formato longo"):
    display_long = long_data.copy()
    display_long["Laterality"] = display_long["Laterality"].replace({
        "Right-handed": "Destro",
        "Left-handed": "Canhoto",
    })
    display_long["HandCondition"] = display_long["HandCondition"].replace({
        "Dominant": "Dominante",
        "Non-dominant": "Não dominante",
    })
    display_long = display_long.rename(
        columns={"Outcome": outcome_name}
    )
    st.dataframe(display_long, use_container_width=True)

# ============================================================
# Resultados
# ============================================================

st.divider()
st.header("Resultados")

st.subheader("1. Estatística descritiva")

descriptive = descriptive_statistics(long_data)

descriptive["Laterality"] = descriptive["Laterality"].replace({
    "Right-handed": "Destros",
    "Left-handed": "Canhotos",
})

descriptive["HandCondition"] = descriptive["HandCondition"].replace({
    "Dominant": "Dominante",
    "Non-dominant": "Não dominante",
})

st.dataframe(descriptive, use_container_width=True)

st.subheader("2. Modelo principal")

try:
    (
        model_result,
        main_effects,
        full_coefficients,
        model_method,
        model_warning,
    ) = fit_mixed_model(long_data)

    st.caption(f"Método utilizado: {model_method}")

    if model_warning and model_method.startswith("Regressão"):
        st.warning(
            "O modelo misto não convergiu. Foi usada regressão linear "
            "com erros-padrão robustos agrupados por participante."
        )

    st.dataframe(main_effects, use_container_width=True)

    interaction_row = main_effects[
        main_effects["Efeito"] ==
        "Interação lateralidade × dominância"
    ]

    if not interaction_row.empty:
        interaction_p = interaction_row["p"].iloc[0]

        if interaction_p < alpha:
            st.success(
                "A interação lateralidade × dominância manual foi "
                f"estatisticamente significativa (p = {interaction_p:.4g}). "
                "A diferença entre mão dominante e não dominante depende "
                "da lateralidade."
            )
        else:
            st.info(
                "A interação lateralidade × dominância manual não foi "
                f"estatisticamente significativa (p = {interaction_p:.4g})."
            )

    with st.expander("Coeficientes completos do modelo"):
        st.dataframe(
            full_coefficients,
            use_container_width=True,
        )
        st.text(str(model_result.summary()))

except Exception as error:
    st.error(f"Não foi possível ajustar o modelo: {error}")
    main_effects = pd.DataFrame()
    full_coefficients = pd.DataFrame()

st.pyplot(
    create_interaction_plot(
        long_data,
        outcome_name,
    )
)

st.subheader("3. Comparações pós-hoc")

posthoc = posthoc_tests(
    right_complete,
    left_complete,
    correction_method,
)

st.dataframe(posthoc, use_container_width=True)

st.subheader("4. Índice de assimetria")

asymmetry_individual, asymmetry_comparison = asymmetry_analysis(
    right_complete,
    left_complete,
)

st.dataframe(
    asymmetry_comparison,
    use_container_width=True,
)

with st.expander("Índices individuais de assimetria"):
    st.dataframe(
        asymmetry_individual,
        use_container_width=True,
    )

st.subheader("5. Normalidade das diferenças pareadas")

normality_rows = []

for group_name, group_df in [
    ("Destros", right_complete),
    ("Canhotos", left_complete),
]:
    differences = (
        group_df["Dominant"] -
        group_df["Non-dominant"]
    )

    statistic, p_value = shapiro_safe(differences)

    normality_rows.append({
        "Grupo": group_name,
        "n": len(differences),
        "Shapiro–Wilk W": statistic,
        "p": p_value,
    })

normality = pd.DataFrame(normality_rows)

st.dataframe(normality, use_container_width=True)

# ============================================================
# Exportação
# ============================================================

st.divider()
st.header("Exportação")

export_buffer = io.BytesIO()

with pd.ExcelWriter(
    export_buffer,
    engine="openpyxl",
) as writer:
    raw_data.to_excel(
        writer,
        sheet_name="Dados_originais",
        index=False,
    )

    long_export = long_data.rename(
        columns={"Outcome": outcome_name}
    )

    long_export.to_excel(
        writer,
        sheet_name="Dados_longos",
        index=False,
    )

    descriptive.to_excel(
        writer,
        sheet_name="Descritivos",
        index=False,
    )

    main_effects.to_excel(
        writer,
        sheet_name="Efeitos_modelo",
        index=False,
    )

    full_coefficients.to_excel(
        writer,
        sheet_name="Coeficientes",
        index=False,
    )

    posthoc.to_excel(
        writer,
        sheet_name="Pos_hoc",
        index=False,
    )

    asymmetry_individual.to_excel(
        writer,
        sheet_name="Assimetria_individual",
        index=False,
    )

    asymmetry_comparison.to_excel(
        writer,
        sheet_name="Assimetria_comparacao",
        index=False,
    )

    normality.to_excel(
        writer,
        sheet_name="Normalidade",
        index=False,
    )

st.download_button(
    "⬇️ Baixar resultados completos em Excel",
    data=export_buffer.getvalue(),
    file_name=(
        "analise_lateralidade_dominancia_"
        f"{outcome_name.replace(' ', '_')}.xlsx"
    ),
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
)

st.info(
    "A interpretação deve começar pela interação lateralidade × "
    "dominância manual. As comparações pós-hoc mostram a diferença "
    "entre as mãos dentro de cada grupo e as diferenças entre destros "
    "e canhotos em cada condição."
)
