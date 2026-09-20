# Analyse nutritionnelle des produits USDA

Application Streamlit pour analyser la composition nutritionnelle des produits de la base USDA.

## Démo en ligne

[Ouvrir l'application Streamlit](https://usda-nutrition-app-9awedg4cm2qbog5la5j8ci.streamlit.app/)

## Objectifs

- Décrire les caractéristiques nutritionnelles des produits.
- Étudier les associations entre les nutriments avec les corrélations.
- Construire un profil synthétique des produits avec une analyse en composantes principales (ACP).

## Contenu de l'application

L'application présente quatre étapes :

1. compréhension de la structure de la base ;
2. qualité des données et statistiques descriptives ;
3. associations entre variables nutritionnelles ;
4. ACP, cercle des corrélations et projection des produits.

Les valeurs manquantes utilisées par l'ACP sont remplacées par la médiane, puis les variables sont standardisées.

## Installation et lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

La base `USDA_National_Nutrient_DataBase.xlsx` doit rester dans le même dossier que `app.py`.

## Données

La feuille `ABBREV` contient les produits et leurs valeurs nutritionnelles, généralement exprimées pour 100 g.