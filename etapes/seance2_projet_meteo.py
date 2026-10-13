"""Checkpoint de la séance 2 : chargement, nettoyage et agrégation mensuelle.

Version linéaire (pas encore découpée en fonctions), conservée pour montrer
l'état du projet en fin de séance 2. La version finale est ../projet_meteo.py.

Exécution : python etapes/seance2_projet_meteo.py
"""

from pathlib import Path

import pandas as pd

FICHIER = Path(__file__).parent.parent / "Fil rouge donnees.csv.csv"

# --- Étape 1 : chargement du fichier complet ---
# sep=";" : séparateur Météo-France ; "mq" = valeur manquante (sinon rr3 serait
# lue en texte) ; date lue en texte pour la convertir avec un format explicite
df_brut = pd.read_csv(FICHIER, sep=";", na_values="mq",
                      dtype={"numer_sta": str, "date": str})
# Le ";" final de chaque ligne crée une colonne vide "Unnamed: 59"
df_brut = df_brut.loc[:, ~df_brut.columns.str.startswith("Unnamed")]
print("Dimensions du fichier brut :", df_brut.shape)

attendu = 365 * 8  # un relevé toutes les 3 h -> 8 par jour
print(f"Lignes attendues : {attendu}, chargées : {len(df_brut)}, "
      f"manquantes : {attendu - len(df_brut)}")
print("Types des colonnes utiles :")
print(df_brut[["date", "t", "pmer", "u", "ff", "rr3"]].dtypes)

# --- Étape 2 : sélection des colonnes utiles ---
# 6 colonnes sur 59 : température, pression, humidité, vent, pluie et date
df = df_brut[["date", "t", "pmer", "u", "ff", "rr3"]].copy()

# --- Étape 3 : conversion et plausibilité ---
df["t_c"] = df["t"] - 273.15          # Kelvin -> Celsius, nouvelle colonne
df["pmer_hpa"] = df["pmer"] / 100     # Pa -> hPa
print(f"\nTempérature : min {df['t_c'].min():.1f} °C, max {df['t_c'].max():.1f} °C")
# -0.1 mm = "trace" de pluie (< 0.1 mm) dans la convention SYNOP
print("Valeurs de rr3 à -0.1 (traces) :", (df["rr3"] == -0.1).sum())
df["rr3"] = df["rr3"].mask(df["rr3"] == -0.1, 0.0)

# --- Étape 4 : valeurs manquantes ---
colonnes = ["t_c", "pmer_hpa", "u", "ff", "rr3"]
manquants = df[colonnes].isna().sum()
print("\nValeurs manquantes par colonne :")
print(pd.DataFrame({"nb": manquants, "pct": (100 * manquants / len(df)).round(2)}))

# t_c (cible) : on supprime plutôt que d'inventer la valeur étudiée
df = df.dropna(subset=["t_c"])
# pression, humidité, vent : continus, on interpole entre relevés voisins
df[["pmer_hpa", "u", "ff"]] = df[["pmer_hpa", "u", "ff"]].interpolate(limit_direction="both")
# pluie : 80 % de zéros, 0 est la valeur la plus probable
df["rr3"] = df["rr3"].fillna(0.0)
print("Valeurs manquantes restantes :", df[colonnes].isna().sum().sum())

# --- Étape 5 : date, mois et agrégation mensuelle ---
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S")
df["mois"] = df["date"].dt.month
df["heure"] = df["date"].dt.hour

mensuel = df.groupby("mois")["t_c"].agg(["mean", "min", "max"]).round(2)
print("\nTempérature par mois (°C) :")
print(mensuel)

# --- Pour aller plus loin ---
print("\nCycle diurne (température moyenne par heure UTC) :")
print(df.groupby("heure")["t_c"].mean().round(2))
