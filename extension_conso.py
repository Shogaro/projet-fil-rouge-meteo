"""Extension optionnelle — la température explique-t-elle la consommation électrique ?

Croise la température journalière de Lille-Lesquin (SYNOP, Météo-France) avec la
consommation électrique journalière de la France (éCO2mix, RTE) sur 2023, puis
applique la même démarche que pour la température : préparation, découpage
chronologique, entraînement, évaluation contre une référence naïve.

Données attendues
-----------------
- le fichier SYNOP du projet principal (voir projet_meteo.py) ;
- un export éCO2mix national (séparateur ";") avec au moins les colonnes
  date, heure, consommation (MW, pas de 15 min, valeurs aux quarts d'heure
  :15/:45 vides par construction car la donnée définitive est semi-horaire).
  Source : jeu « eco2mix-national-cons-def », odre.opendatasoft.com.

Exécution
---------
    python extension_conso.py

Choix
-----
- Échelle du jour : les deux jeux n'ont pas la même fréquence (3 h / 30 min).
  Température : moyenne des relevés du jour, en heure de Paris (les relevés
  SYNOP sont en UTC, les dates éCO2mix en heure locale). Jours avec moins de
  6 relevés sur 8 écartés : une moyenne de 3 relevés nocturnes serait biaisée.
  Consommation : énergie du jour en GWh (puissance moyenne en MW × 24 h / 1000).
- Clé de jointure : la date (type date, granularité jour) des deux côtés.
- Variantes : la consommation monte quand il fait froid (chauffage) mais ne
  baisse plus au-delà d'un seuil, d'où les degrés-jours de chauffage
  max(0, 15 °C − T) ; et un indicateur week-end (activité économique réduite).
- Limite assumée : on explique une consommation *nationale* par la température
  d'*une seule* station, utilisée comme indicateur du froid en France.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import projet_meteo as pm

FICHIER_CONSO = Path(__file__).parent / "eco2mix_national_2023.csv"
DOSSIER_FIGURES = pm.DOSSIER_FIGURES
FUSEAU = "Europe/Paris"
MIN_RELEVES_JOUR = 6
SEUIL_CHAUFFAGE = 15.0  # °C, température sous laquelle on chauffe


def charger_conso(chemin):
    """Charge l'export éCO2mix et vérifie ses colonnes.

    Renvoie
    -------
    DataFrame avec date (datetime.date) et consommation (MW, NaN aux quarts d'heure).

    Lève
    ----
    FileNotFoundError, ValueError (fichier vide ou colonnes absentes).
    """
    try:
        # utf-8-sig : l'export commence par un BOM qui polluerait le 1er nom de colonne
        df = pd.read_csv(chemin, sep=";", encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier éCO2mix introuvable : '{chemin}'.") from None
    except pd.errors.EmptyDataError:
        raise ValueError(f"Le fichier éCO2mix '{chemin}' est vide.") from None
    pm.verifier_colonnes(df, ["date", "heure", "consommation"])
    if df.empty:
        raise ValueError(f"Le fichier éCO2mix '{chemin}' ne contient aucune donnée.")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    return df


def conso_journaliere(conso):
    """Agrège la consommation à l'échelle du jour.

    Les quarts d'heure :15/:45 sont vides par construction et ignorés par la
    moyenne. Un jour avec moins de 40 demi-heures renseignées sur 48 est écarté.

    Renvoie
    -------
    DataFrame indexé par date : conso_gwh, nb_mesures.
    """
    jour = conso.groupby("date")["consommation"].agg(["mean", "count"])
    jour = jour[jour["count"] >= 40]
    return pd.DataFrame({"conso_gwh": jour["mean"] * 24 / 1000,
                         "nb_mesures": jour["count"]})


def temperature_journaliere(meteo, min_releves=MIN_RELEVES_JOUR):
    """Température moyenne par jour local (heure de Paris).

    Paramètres
    ----------
    meteo : DataFrame nettoyé du projet (date en UTC, t_c).
    min_releves : nombre minimal de relevés pour garder un jour.

    Renvoie
    -------
    (DataFrame indexé par date : t_jour, nb_releves ; nombre de jours écartés)
    """
    jour_local = meteo["date"].dt.tz_localize("UTC").dt.tz_convert(FUSEAU).dt.date
    jour = meteo.groupby(jour_local)["t_c"].agg(["mean", "count"])
    jour.index.name = "date"
    garde = jour["count"] >= min_releves
    res = pd.DataFrame({"t_jour": jour["mean"], "nb_releves": jour["count"]})[garde]
    return res, int((~garde).sum())


def joindre(temp_jour, conso_jour):
    """Jointure interne sur la date ; ajoute les variables explicatives dérivées.

    Renvoie
    -------
    DataFrame trié par date avec t_jour, conso_gwh, dju, week_end et une
    colonne ``date`` en datetime (pour réutiliser les fonctions du projet).
    """
    jointure = temp_jour.join(conso_jour, how="inner").sort_index()
    jointure["dju"] = (SEUIL_CHAUFFAGE - jointure["t_jour"]).clip(lower=0)
    jointure["date"] = pd.to_datetime(jointure.index)
    jointure["week_end"] = (jointure["date"].dt.dayofweek >= 5).astype(int)
    return jointure.reset_index(drop=True)


def modele_conso(df, variables):
    """Préparation, découpage chronologique 80/20, entraînement, évaluation.

    Réutilise preparer_xy, separer, entrainer et evaluer du projet principal.

    Renvoie
    -------
    dict comme pm.lancer_modele (modèle, jeux, prédictions, scores).
    """
    X, y, dates = pm.preparer_xy(df, variables, cible="conso_gwh", verbeux=False)
    jeux = pm.separer(X, y, dates)
    modele = pm.entrainer(jeux["X_train"], jeux["y_train"])
    naive = np.full(len(jeux["y_test"]), jeux["y_train"].mean())
    return {
        **jeux,
        "modele": modele,
        "pred_test": modele.predict(jeux["X_test"]),
        "train": pm.evaluer(jeux["y_train"], modele.predict(jeux["X_train"])),
        "test": pm.evaluer(jeux["y_test"], modele.predict(jeux["X_test"])),
        "naive": pm.evaluer(jeux["y_test"], naive),
    }


def afficher(res, titre, moyenne_test):
    """Affiche coefficients et scores (en GWh et en % de la consommation moyenne)."""
    print(f"\n--- {titre} ---")
    print(f"  Ordonnée à l'origine : {res['modele'].intercept_:.0f} GWh")
    for nom, coef in zip(res["X_train"].columns, res["modele"].coef_):
        print(f"  coef {nom:<9} : {coef:+.1f} GWh")
    for jeu in ("train", "test", "naive"):
        s = res[jeu]
        print(f"  {jeu:<6} MAE = {s['MAE']:6.1f} GWh ({100 * s['MAE'] / moyenne_test:4.1f} %) | "
              f"biais = {s['biais']:+6.1f} GWh | R² = {s['R2']:.3f}")


def figure_nuage_conso(df, res_lin, res_dju, dossier):
    """Nuage consommation / température avec les deux modèles ajustés."""
    t = np.linspace(df["t_jour"].min(), df["t_jour"].max(), 200)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    couleurs = np.where(df["week_end"] == 1, "tab:orange", "tab:blue")
    ax.scatter(df["t_jour"], df["conso_gwh"], c=couleurs, s=12)
    ax.plot(t, res_lin["modele"].predict(pd.DataFrame({"t_jour": t})),
            "k--", label="droite : conso ~ température")
    dju = np.clip(SEUIL_CHAUFFAGE - t, 0, None)
    ax.plot(t, res_dju["modele"].predict(pd.DataFrame({"dju": dju})),
            "r-", label=f"degrés-jours (seuil {SEUIL_CHAUFFAGE:.0f} °C)")
    ax.scatter([], [], c="tab:blue", s=12, label="jour de semaine")
    ax.scatter([], [], c="tab:orange", s=12, label="week-end")
    ax.set_xlabel("Température moyenne journalière à Lille (°C)")
    ax.set_ylabel("Consommation électrique France (GWh/jour)")
    ax.set_title("Consommation électrique journalière en fonction de la température (2023)")
    ax.legend()
    ax.grid(alpha=0.3)
    return pm.sauver_figure(fig, dossier, "10_conso_temperature.png")


def figure_erreurs_conso(res_lin, res_complet, dossier):
    """Consommation réelle et prédite sur la période de test."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    d = res_lin["d_test"]
    ax1.plot(d, res_lin["y_test"], "k", label="réel")
    ax1.plot(d, res_lin["pred_test"], "--", label="droite (température)")
    ax1.plot(d, res_complet["pred_test"], label="degrés-jours + week-end")
    ax1.set_ylabel("GWh/jour")
    ax1.set_title("Consommation journalière sur la période de test")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(d, res_lin["pred_test"] - res_lin["y_test"].values, "--", label="droite")
    ax2.plot(d, res_complet["pred_test"] - res_complet["y_test"].values,
             label="degrés-jours + week-end")
    ax2.axhline(0, color="red", lw=0.8)
    ax2.set_ylabel("Erreur prédit - réel (GWh)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return pm.sauver_figure(fig, dossier, "11_conso_erreurs.png")


def main():
    """Enchaîne découverte, agrégation journalière, jointure, modèles et figures."""
    print("=== Découverte du fichier éCO2mix ===")
    conso = charger_conso(FICHIER_CONSO)
    print(f"{len(conso)} lignes, colonnes : {list(conso.columns)}")
    quarts = conso["heure"].str[-2:].isin(["15", "45"])
    print(f"Valeurs vides : {conso['consommation'].isna().sum()} "
          f"(dont {conso.loc[quarts, 'consommation'].isna().sum()} aux quarts d'heure :15/:45, "
          f"{conso.loc[~quarts, 'consommation'].isna().sum()} aux :00/:30)")
    print(f"Période : {conso['date'].min()} -> {conso['date'].max()}, "
          f"{conso['date'].nunique()} jours")

    print("\n=== Passage à l'échelle du jour ===")
    conso_jour = conso_journaliere(conso)
    meteo = pm.nettoyer(pm.preparer_colonnes(pm.charger_donnees(pm.FICHIER_DONNEES)))
    temp_jour, ecartes = temperature_journaliere(meteo)
    print(f"Consommation : {len(conso_jour)} jours "
          f"(de {conso_jour['conso_gwh'].min():.0f} à {conso_jour['conso_gwh'].max():.0f} GWh/jour)")
    print(f"Température : {len(temp_jour)} jours gardés, {ecartes} jour(s) écarté(s) "
          f"(< {MIN_RELEVES_JOUR} relevés)")

    print("\n=== Jointure ===")
    df = joindre(temp_jour, conso_jour)
    attendu = len(set(temp_jour.index) & set(conso_jour.index))
    print(f"Lignes attendues : {attendu} (jours présents des deux côtés), obtenues : {len(df)}")
    manquants = sorted(set(conso_jour.index) - set(temp_jour.index))
    print(f"Jours sans température exploitable ({len(manquants)}) : "
          f"{', '.join(str(j) for j in manquants)}")
    print(f"Corrélation température / consommation : {df['t_jour'].corr(df['conso_gwh']):.3f}")
    print(f"Corrélation degrés-jours / consommation : {df['dju'].corr(df['conso_gwh']):.3f}")
    print("Consommation moyenne semaine / week-end : "
          f"{df.groupby('week_end')['conso_gwh'].mean().round(0).tolist()} GWh")

    print("\n=== Modélisation (découpage chronologique 80/20) ===")
    variantes = {
        "Droite : conso ~ température": ["t_jour"],
        "Degrés-jours de chauffage": ["dju"],
        "Degrés-jours + week-end": ["dju", "week_end"],
    }
    resultats = {nom: modele_conso(df, v) for nom, v in variantes.items()}
    premier = next(iter(resultats.values()))
    print(f"  Entraînement : {len(premier['y_train'])} jours "
          f"({premier['d_train'].min():%Y-%m-%d} -> {premier['d_train'].max():%Y-%m-%d})")
    print(f"  Test         : {len(premier['y_test'])} jours "
          f"({premier['d_test'].min():%Y-%m-%d} -> {premier['d_test'].max():%Y-%m-%d})")
    moyenne_test = premier["y_test"].mean()
    for nom, res in resultats.items():
        afficher(res, nom, moyenne_test)

    print("\nComparaison avec le modèle de température (projet principal) : "
          "voir NOTES.md.")
    figures = [
        figure_nuage_conso(df, resultats["Droite : conso ~ température"],
                           resultats["Degrés-jours de chauffage"], DOSSIER_FIGURES),
        figure_erreurs_conso(resultats["Droite : conso ~ température"],
                             resultats["Degrés-jours + week-end"], DOSSIER_FIGURES),
    ]
    print("\nFigures enregistrées :")
    for f in figures:
        print(f"  {f.relative_to(Path(__file__).parent)} ({f.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as erreur:
        print(f"Erreur : {erreur}", file=sys.stderr)
        sys.exit(1)
