from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Analyse nutritionnelle USDA",
    page_icon="🥗",
    layout="wide",
)


DATA_FILE = Path(__file__).parent / "USDA_National_Nutrient_DataBase.xlsx"


@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE, sheet_name="ABBREV")


@st.cache_data
def prepare_acp(data, min_completion):
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    excluded_columns = {"NDB_No", "GmWt_1", "GmWt_2", "Refuse_Pct"}
    acp_columns = [
        column
        for column in numeric_columns
        if column not in excluded_columns
        and data[column].notna().mean() >= min_completion
    ]
    acp_data = data[acp_columns]
    imputed = SimpleImputer(strategy="median").fit_transform(acp_data)
    scaled = StandardScaler().fit_transform(imputed)
    pca = PCA().fit(scaled)
    coordinates = pca.transform(scaled)
    loadings = pd.DataFrame(
        pca.components_.T[:, :2] * np.sqrt(pca.explained_variance_[:2]),
        index=acp_columns,
        columns=["PC1", "PC2"],
    )
    return acp_columns, pca, coordinates, loadings


def explain_step(title, objective, result):
    st.subheader(title)
    st.write(f"**À quoi ça sert :** {objective}")
    st.info(f"**Ce qui en ressort :** {result}")


st.title("Analyse nutritionnelle des produits USDA")
st.caption("Exploration des caractéristiques, associations et profils produits par ACP")

if not DATA_FILE.exists():
    st.error(f"Fichier introuvable : {DATA_FILE.name}")
    st.stop()

data = load_data()
numeric_columns = data.select_dtypes(include="number").columns.tolist()
missing_rate = data.isna().mean().sort_values(ascending=False)

with st.sidebar:
    st.header("Paramètres")
    min_completion = st.slider(
        "Taux minimal de données disponibles pour l'ACP",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        format="%.0f",
    )
    st.write(f"Variables numériques : **{len(numeric_columns)}**")
    st.write(f"Produits : **{len(data):,}**")

acp_columns, pca, coordinates, loadings = prepare_acp(data, min_completion)

tab_overview, tab_quality, tab_relations, tab_pca = st.tabs(
    ["1. Comprendre les données", "2. Qualité et descriptif", "3. Associations", "4. ACP et profils"]
)

with tab_overview:
    explain_step(
        "Étape 1 - Importer les données",
        "Lire la feuille ABBREV du fichier USDA et identifier les produits et leurs nutriments.",
        f"La base contient {len(data):,} produits et {data.shape[1]} variables. Chaque ligne représente un produit et les valeurs nutritionnelles sont généralement exprimées pour 100 g.",
    )
    st.dataframe(data.head(10), use_container_width=True)
    st.subheader("Structure des variables")
    st.write("Les variables comprennent l'énergie, les macronutriments, les minéraux, les vitamines, les acides gras et le cholestérol.")
    st.dataframe(
        pd.DataFrame({"variable": data.columns, "type": data.dtypes.astype(str).values}),
        use_container_width=True,
        hide_index=True,
    )

with tab_quality:
    explain_step(
        "Étape 2 - Vérifier et décrire la base",
        "Mesurer les valeurs manquantes et résumer les distributions avant les calculs statistiques.",
        f"Certaines variables sont très complètes, tandis que plusieurs vitamines ont des données manquantes. Pour l'ACP, les valeurs absentes sont remplacées par la médiane de leur variable; {len(acp_columns)} variables respectent le seuil choisi.",
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Valeurs manquantes")
        missing_display = (missing_rate.head(15) * 100).sort_values()
        st.bar_chart(missing_display.rename("Pourcentage manquant"))
    with right:
        st.subheader("Statistiques principales")
        main_nutrients = [
            column for column in [
                "Energ_Kcal", "Protein_(g)", "Lipid_Tot_(g)", "Carbohydrt_(g)",
                "Fiber_TD_(g)", "Sugar_Tot_(g)", "Sodium_(mg)", "Calcium_(mg)",
            ] if column in data
        ]
        st.dataframe(data[main_nutrients].describe().T.round(2), use_container_width=True)

    plot_columns = [
        column for column in ["Energ_Kcal", "Protein_(g)", "Lipid_Tot_(g)", "Carbohydrt_(g)"]
        if column in data
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, column in zip(axes.ravel(), plot_columns):
        sns.histplot(data[column].dropna(), bins=40, kde=True, ax=axis, color="#176b87")
        axis.set_title(column)
        axis.set_xlabel("Valeur pour 100 g")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with tab_relations:
    explain_step(
        "Étape 3 - Étudier les associations",
        "Calculer les corrélations de Pearson pour voir si deux nutriments évoluent ensemble.",
        "Une corrélation proche de +1 indique une association positive forte; une corrélation proche de -1 indique une association inverse. Une corrélation proche de 0 signale une relation linéaire faible.",
    )
    correlation = data[acp_columns].corr()
    figure, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axis)
    axis.set_title("Corrélations entre variables nutritionnelles")
    st.pyplot(figure)
    plt.close(figure)

    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .rename("corrélation")
        .sort_values(key=np.abs, ascending=False)
        .head(12)
        .to_frame()
    )
    st.subheader("Associations les plus fortes")
    st.dataframe(pairs.round(3), use_container_width=True)

with tab_pca:
    explained = pca.explained_variance_ratio_
    explain_step(
        "Étape 4 - Construire le profil global par ACP",
        "Réduire plusieurs variables nutritionnelles à quelques axes synthétiques, après imputation et standardisation.",
        f"Le premier axe explique {explained[0] * 100:.1f} % de la variabilité et les deux premiers axes expliquent {(explained[:2].sum()) * 100:.1f} %. Les variables les plus éloignées de l'origine structurent le plus les profils.",
    )
    explained_table = pd.DataFrame({
        "axe": [f"PC{i}" for i in range(1, len(explained) + 1)],
        "variance_expliquee_%": explained * 100,
        "variance_cumulee_%": explained.cumsum() * 100,
    })
    st.dataframe(explained_table.head(10).round(2), use_container_width=True, hide_index=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(range(1, len(explained) + 1), explained * 100, marker="o", color="#d95f02")
    axes[0].set_title("Variance expliquée par axe")
    axes[0].set_xlabel("Composante")
    axes[0].set_ylabel("Pourcentage")
    axes[1].plot(range(1, len(explained) + 1), explained.cumsum() * 100, marker="o", color="#1b9e77")
    axes[1].set_title("Variance cumulée")
    axes[1].set_xlabel("Nombre de composantes")
    axes[1].set_ylabel("Pourcentage")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Variables qui construisent les deux premiers axes")
    st.dataframe(loadings.assign(force=loadings.pow(2).sum(axis=1).pow(0.5)).sort_values("force", ascending=False).round(3), use_container_width=True)

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.axvline(0, color="grey", linewidth=0.8)
    for variable, row in loadings.iterrows():
        axis.arrow(0, 0, row["PC1"], row["PC2"], color="#176b87", alpha=0.7, head_width=0.025, length_includes_head=True)
        axis.text(row["PC1"] * 1.07, row["PC2"] * 1.07, variable, fontsize=8)
    axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), title="Cercle des corrélations", xlabel="PC1", ylabel="PC2")
    st.pyplot(figure)
    plt.close(figure)

    coordinates_df = pd.DataFrame(coordinates[:, :2], columns=["PC1", "PC2"])
    coordinates_df["produit"] = data["Shrt_Desc"].values
    coordinates_df["energie"] = data["Energ_Kcal"].values
    st.subheader("Projection des produits")
    st.scatter_chart(coordinates_df, x="PC1", y="PC2", color="energie")
    st.write("Chaque point représente un produit. Les produits proches ont des compositions nutritionnelles similaires; les points éloignés correspondent à des profils différents.")

    st.subheader("Produits les plus atypiques")
    selected_axis = st.selectbox("Choisir un axe", ["PC1", "PC2"])
    extreme_rows = coordinates_df.loc[
        coordinates_df[selected_axis].abs().nlargest(10).index,
        ["produit", selected_axis, "energie"],
    ].sort_values(selected_axis)
    st.dataframe(extreme_rows, use_container_width=True, hide_index=True)

st.divider()
st.caption("Interprétation: l'ACP décrit des profils nutritionnels; elle ne prouve pas une relation de causalité entre les nutriments.")