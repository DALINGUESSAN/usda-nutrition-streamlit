from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).parent
FIGURE_DIR = PROJECT_DIR / "figures_rapport"
FIGURE_DIR.mkdir(exist_ok=True)

data = pd.read_excel(PROJECT_DIR / "USDA_National_Nutrient_DataBase.xlsx", sheet_name="ABBREV")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"

numeric_columns = data.select_dtypes(include="number").columns.tolist()
excluded = {"NDB_No", "GmWt_1", "GmWt_2", "Refuse_Pct"}
acp_columns = [
    column for column in numeric_columns
    if column not in excluded and data[column].notna().mean() >= 0.70
]
imputed = SimpleImputer(strategy="median").fit_transform(data[acp_columns])
scaled = StandardScaler().fit_transform(imputed)
pca = PCA().fit(scaled)
coordinates = pca.transform(scaled)

main_columns = [
    "Energ_Kcal", "Protein_(g)", "Lipid_Tot_(g)", "Carbohydrt_(g)"
]
figure, axes = plt.subplots(2, 2, figsize=(10, 7))
for axis, column in zip(axes.ravel(), main_columns):
    sns.histplot(data[column].dropna(), bins=40, kde=True, ax=axis, color="#176b87")
    axis.set_title(column)
    axis.set_xlabel("Valeur pour 100 g")
    axis.set_ylabel("Nombre de produits")
figure.suptitle("Distribution de l'énergie et des macronutriments", y=1.02)
figure.savefig(FIGURE_DIR / "distributions.png")
plt.close(figure)

missing = data.isna().mean().sort_values(ascending=False).head(15).sort_values() * 100
figure, axis = plt.subplots(figsize=(9, 6))
missing.plot.barh(ax=axis, color="#d95f02")
axis.set_title("Variables présentant le plus de valeurs manquantes")
axis.set_xlabel("Pourcentage de valeurs manquantes")
axis.set_ylabel("")
figure.savefig(FIGURE_DIR / "valeurs_manquantes.png")
plt.close(figure)

correlation = data[acp_columns].corr()
figure, axis = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axis)
axis.set_title("Matrice de corrélation des variables nutritionnelles")
figure.savefig(FIGURE_DIR / "correlations.png")
plt.close(figure)

figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100, marker="o", color="#d95f02")
axes[0].set_title("Variance expliquée")
axes[0].set_xlabel("Composante")
axes[0].set_ylabel("Pourcentage")
axes[1].plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum() * 100, marker="o", color="#1b9e77")
axes[1].set_title("Variance cumulée")
axes[1].set_xlabel("Nombre de composantes")
axes[1].set_ylabel("Pourcentage")
figure.savefig(FIGURE_DIR / "variance_acp.png")
plt.close(figure)

loadings = pd.DataFrame(
    pca.components_.T[:, :2] * np.sqrt(pca.explained_variance_[:2]),
    index=acp_columns,
    columns=["PC1", "PC2"],
)
figure, axis = plt.subplots(figsize=(9, 7))
axis.axhline(0, color="grey", linewidth=0.8)
axis.axvline(0, color="grey", linewidth=0.8)
for variable, row in loadings.iterrows():
    axis.arrow(0, 0, row["PC1"], row["PC2"], color="#176b87", alpha=0.7, head_width=0.025, length_includes_head=True)
    axis.text(row["PC1"] * 1.07, row["PC2"] * 1.07, variable, fontsize=7)
axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), title="Cercle des corrélations", xlabel="PC1", ylabel="PC2")
figure.savefig(FIGURE_DIR / "cercle_correlations.png")
plt.close(figure)

figure, axis = plt.subplots(figsize=(10, 6))
scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=data["Energ_Kcal"], cmap="viridis", s=10, alpha=0.45)
figure.colorbar(scatter, ax=axis, label="Energie (kcal / 100 g)")
axis.set_title("Projection des produits dans le plan factoriel")
axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)" )
axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)" )
figure.savefig(FIGURE_DIR / "projection_produits.png")
plt.close(figure)

print(f"Figures créées dans {FIGURE_DIR}")