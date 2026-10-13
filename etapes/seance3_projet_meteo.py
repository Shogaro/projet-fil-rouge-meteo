"""Checkpoint de la séance 3 : exploration visuelle et script structuré.

Version conservée pour montrer l'état du projet en fin de séance 3 (sans la
régression de la séance 4). La version finale est ../projet_meteo.py.

Exécution : python etapes/seance3_projet_meteo.py [fichier.csv]
Les figures sont écrites dans etapes/figures_seance3/.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # aucune fenêtre : les figures sont uniquement enregistrées
import matplotlib.pyplot as plt
import pandas as pd

# --- Constantes du projet (définies à un seul endroit) ---
FICHIER_DONNEES = Path(__file__).parent.parent / "Fil rouge donnees.csv.csv"
DOSSIER_FIGURES = Path(__file__).parent / "figures_seance3"
PAS_HEURES = 3  # un relevé SYNOP toutes les 3 heures

COLONNES_BRUTES = ["date", "t", "pmer", "u", "ff", "rr3"]
COLONNES_METEO = ["t_c", "pmer_hpa", "u", "ff", "rr3"]

# Bornes de plausibilité pour Lille (valeurs très larges : on ne cherche que
# les erreurs grossières, pas les records)
BORNES = {
    "t_c": (-40.0, 50.0),
    "pmer_hpa": (900.0, 1100.0),
    "u": (0.0, 100.0),
    "ff": (0.0, 75.0),
    "rr3": (0.0, 200.0),
}

# =============================================================================
# Séance 2 — chargement et nettoyage
# =============================================================================

def charger_donnees(chemin):
    """Charge le CSV SYNOP brut.

    Paramètres
    ----------
    chemin : str ou Path
        Chemin du fichier Météo-France (séparateur ";", manquants "mq").

    Renvoie
    -------
    DataFrame
        Données brutes, sans la colonne vide créée par le ";" final.

    Lève
    ----
    FileNotFoundError si le fichier n'existe pas, ValueError s'il est vide.
    """
    chemin = Path(chemin)
    try:
        # date lue en texte pour la convertir ensuite avec un format explicite
        df = pd.read_csv(chemin, sep=";", na_values="mq",
                         dtype={"numer_sta": str, "date": str})
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Fichier de données introuvable : '{chemin}'. "
            "Vérifiez son nom et son emplacement."
        ) from None
    except pd.errors.EmptyDataError:
        raise ValueError(f"Le fichier '{chemin}' est vide.") from None

    if df.empty:
        raise ValueError(f"Le fichier '{chemin}' ne contient aucune ligne de données.")

    # Chaque ligne se termine par ";" : pandas crée une colonne "Unnamed: xx" vide
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def verifier_colonnes(df, colonnes):
    """Vérifie que toutes les colonnes demandées existent.

    Paramètres
    ----------
    df : DataFrame
    colonnes : list[str]

    Lève
    ----
    ValueError en listant les colonnes absentes.
    """
    absentes = [c for c in colonnes if c not in df.columns]
    if absentes:
        raise ValueError(f"Colonne(s) attendue(s) absente(s) du fichier : {absentes}")


def preparer_colonnes(df):
    """Sélectionne les colonnes utiles et ajoute les colonnes converties.

    On ne garde que la date et les grandeurs utilisées dans la suite du projet :
    le fichier brut a ~60 colonnes (nébulosité, visibilité...) inutiles ici.
    Les colonnes d'origine (t, pmer) sont conservées à côté des colonnes
    converties pour pouvoir contrôler la conversion.

    Paramètres
    ----------
    df : DataFrame brut (non modifié).

    Renvoie
    -------
    DataFrame
        Nouveau DataFrame avec date (datetime), t_c (°C), pmer_hpa (hPa),
        u (%), ff (m/s), rr3 (mm), mois et heure.
    """
    verifier_colonnes(df, COLONNES_BRUTES)
    res = df[COLONNES_BRUTES].copy()

    try:
        res["date"] = pd.to_datetime(res["date"], format="%Y%m%d%H%M%S")
    except ValueError:
        raise ValueError(
            "La colonne 'date' ne respecte pas le format attendu AAAAMMJJHHMMSS."
        ) from None

    res["t_c"] = res["t"] - 273.15         # Kelvin -> Celsius
    res["pmer_hpa"] = res["pmer"] / 100    # Pa -> hPa
    # Dans les fichiers SYNOP, -0.1 mm signifie "trace" (< 0.1 mm, non mesurable)
    res["rr3"] = res["rr3"].mask(res["rr3"] == -0.1, 0.0)
    res["mois"] = res["date"].dt.month
    res["heure"] = res["date"].dt.hour
    return res


def verifier_plausibilite(df, bornes=BORNES):
    """Vérifie que les valeurs non manquantes restent dans des bornes plausibles.

    Paramètres
    ----------
    df : DataFrame contenant les colonnes de ``bornes``.
    bornes : dict colonne -> (min, max)

    Lève
    ----
    ValueError en indiquant la colonne, le nombre de valeurs hors bornes et un
    exemple de ligne fautive.
    """
    for col, (vmin, vmax) in bornes.items():
        hors = df[col].notna() & ~df[col].between(vmin, vmax)
        if hors.any():
            exemple = df.loc[hors].iloc[0]
            raise ValueError(
                f"{hors.sum()} valeur(s) implausible(s) dans '{col}' "
                f"(attendu entre {vmin} et {vmax}), par ex. {exemple[col]} "
                f"le {exemple['date']}."
            )


def rapport_manquants(df, colonnes):
    """Compte les valeurs manquantes par colonne.

    Renvoie
    -------
    DataFrame
        Colonnes ``nb`` (nombre de NaN) et ``pct`` (proportion en %).
    """
    nb = df[colonnes].isna().sum()
    return pd.DataFrame({"nb": nb, "pct": (100 * nb / len(df)).round(2)})


def creneaux_absents(df):
    """Liste les créneaux de 3 h absents du fichier (lignes entières manquantes).

    Renvoie
    -------
    DatetimeIndex des dates attendues mais absentes.
    """
    attendu = pd.date_range(df["date"].min(), df["date"].max(), freq=f"{PAS_HEURES}h")
    return attendu.difference(df["date"])


def nettoyer(df):
    """Traite les valeurs manquantes avec une stratégie adaptée à chaque colonne.

    - t_c (cible) : lignes supprimées. On ne fabrique pas la valeur qu'on
      cherche à analyser et à prédire.
    - pmer_hpa, u, ff : interpolation dans le temps. Ces grandeurs varient
      peu en 3 h, la moyenne des relevés voisins est une bonne estimation
      (une moyenne annuelle serait absurde pour un trou en janvier).
    - rr3 : remplacé par 0. La pluie est très intermittente (environ 80 % des
      relevés sont à 0) : interpoler inventerait de la pluie entre deux
      averses, 0 est la valeur la plus probable.

    Paramètres
    ----------
    df : DataFrame issu de ``preparer_colonnes`` (non modifié).

    Renvoie
    -------
    DataFrame nettoyé, trié par date, sans NaN sur COLONNES_METEO.
    """
    res = df.sort_values("date").dropna(subset=["t_c"]).copy()
    continues = ["pmer_hpa", "u", "ff"]
    res[continues] = res[continues].interpolate(limit_direction="both")
    res["rr3"] = res["rr3"].fillna(0.0)
    return res.reset_index(drop=True)


def agregation_mensuelle(df):
    """Température moyenne/min/max et cumul de pluie par mois.

    Renvoie
    -------
    DataFrame indexé par le numéro de mois.
    """
    return df.groupby("mois").agg(
        t_moy=("t_c", "mean"),
        t_min=("t_c", "min"),
        t_max=("t_c", "max"),
        u_moy=("u", "mean"),
        pluie_mm=("rr3", "sum"),
        nb_releves=("t_c", "size"),
    ).round(2)


def agregation_horaire(df):
    """Température moyenne par heure de la journée (cycle diurne).

    Renvoie
    -------
    DataFrame indexé par l'heure (0, 3, ..., 21).
    """
    return df.groupby("heure")["t_c"].agg(["mean", "min", "max"]).round(2)


def manquants_par_mois(df, colonnes):
    """Proportion (%) de valeurs manquantes par mois, avant nettoyage.

    Renvoie
    -------
    DataFrame mois x colonnes.
    """
    return (df[colonnes].isna().groupby(df["mois"]).mean() * 100).round(2)


# =============================================================================
# Séance 3 — exploration visuelle
# =============================================================================

def sauver_figure(fig, dossier, nom):
    """Enregistre une figure puis la ferme (savefig AVANT close, sinon image vide).

    Renvoie
    -------
    Path du fichier écrit.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / nom
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
    return chemin


def figure_temperature(df, mensuel, dossier):
    """Température brute (tous les relevés) et moyenne mensuelle.

    La série est réindexée sur la grille complète de 3 h : les créneaux
    absents apparaissent comme des trous au lieu d'être reliés par un trait.
    """
    serie = df.set_index("date")["t_c"].asfreq(f"{PAS_HEURES}h")
    milieux_mois = pd.to_datetime(
        [f"{df['date'].dt.year.iloc[0]}-{m:02d}-15" for m in mensuel.index]
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(serie.index, serie.values, lw=0.5, color="tab:blue")
    ax1.set_title("Température à Lille-Lesquin, tous les relevés (pas de 3 h)")
    ax1.set_ylabel("Température (°C)")
    ax1.grid(alpha=0.3)

    ax2.plot(milieux_mois, mensuel["t_moy"], marker="o", label="moyenne")
    ax2.fill_between(milieux_mois, mensuel["t_min"], mensuel["t_max"],
                     alpha=0.2, label="min - max")
    ax2.set_title("Température mensuelle")
    ax2.set_ylabel("Température (°C)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return sauver_figure(fig, dossier, "01_temperature.png")


def figure_precipitations(df, mensuel, dossier):
    """Répartition des cumuls sur 3 h (hors zéros) et cumul mensuel.

    Pour le mois le plus pluvieux, on somme : la moyenne par relevé dépend du
    nombre de relevés et ne répond pas à « combien est-il tombé ? ».
    """
    pluie = df.loc[df["rr3"] > 0, "rr3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(pluie, bins=40, color="tab:blue", edgecolor="white")
    ax1.set_yscale("log")
    ax1.set_title(f"Précipitations sur 3 h, relevés pluvieux uniquement\n"
                  f"({len(pluie)} relevés > 0 sur {len(df)})")
    ax1.set_xlabel("Cumul sur 3 h (mm)")
    ax1.set_ylabel("Nombre de relevés (échelle log)")

    ax2.bar(mensuel.index, mensuel["pluie_mm"], color="tab:cyan")
    ax2.set_xticks(mensuel.index)
    ax2.set_title("Cumul mensuel de précipitations")
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Cumul (mm)")
    return sauver_figure(fig, dossier, "02_precipitations.png")


def matrice_correlation(df, colonnes=COLONNES_METEO):
    """Matrice de corrélation de Pearson entre les colonnes météo."""
    return df[colonnes].corr().round(3)


def figure_correlations(corr, dossier):
    """Carte de chaleur de la matrice de corrélation."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Coefficient de corrélation")
    ax.set_title("Corrélations entre variables météo")
    return sauver_figure(fig, dossier, "03_correlations.png")


def figure_nuages(df, dossier):
    """Nuages de points température / humidité et température / pression.

    Les points sont colorés par mois : la relation change selon la saison,
    ce que le coefficient global ne montre pas.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, label in [(axes[0], "u", "Humidité relative (%)"),
                           (axes[1], "pmer_hpa", "Pression au niveau de la mer (hPa)")]:
        sc = ax.scatter(df[col], df["t_c"], c=df["mois"], cmap="twilight", s=6)
        ax.set_xlabel(label)
        ax.set_ylabel("Température (°C)")
        ax.set_title(f"Température en fonction de {col}")
        ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=axes[1], label="Mois")
    return sauver_figure(fig, dossier, "04_nuages.png")


def figure_boxplot_mois(df, dossier):
    """Distribution de la température par mois (boîtes à moustaches)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    groupes = [df.loc[df["mois"] == m, "t_c"] for m in range(1, 13)]
    ax.boxplot(groupes, tick_labels=range(1, 13))
    ax.set_title("Distribution de la température par mois")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Température (°C)")
    ax.grid(alpha=0.3, axis="y")
    return sauver_figure(fig, dossier, "05_boxplot_mois.png")


def figure_temp_humidite(mensuel, dossier):
    """Température et humidité moyennes mensuelles sur deux axes verticaux."""
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(mensuel.index, mensuel["t_moy"], marker="o", color="tab:red")
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("Température moyenne (°C)", color="tab:red")
    ax1.set_xticks(mensuel.index)
    ax2 = ax1.twinx()
    ax2.plot(mensuel.index, mensuel["u_moy"], marker="s", color="tab:blue")
    ax2.set_ylabel("Humidité moyenne (%)", color="tab:blue")
    ax1.set_title("Température et humidité moyennes mensuelles")
    return sauver_figure(fig, dossier, "06_temp_humidite.png")


def figure_cycle_diurne(horaire, dossier):
    """Température moyenne selon l'heure de la journée."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horaire.index, horaire["mean"], marker="o")
    ax.set_xticks(horaire.index)
    ax.set_title("Cycle diurne de la température (moyenne annuelle)")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Température moyenne (°C)")
    ax.grid(alpha=0.3)
    return sauver_figure(fig, dossier, "07_cycle_diurne.png")


# =============================================================================
# Programme principal
# =============================================================================

def main(chemin=FICHIER_DONNEES):
    """Enchaîne chargement, nettoyage, agrégations et figures d'exploration."""
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=== Séance 2 : chargement et nettoyage ===")
    brut = charger_donnees(chemin)
    print(f"Dimensions du fichier brut : {brut.shape}")

    df = preparer_colonnes(brut)
    absents = creneaux_absents(df)
    print(f"Lignes attendues (pas de {PAS_HEURES} h) : {len(df) + len(absents)}, "
          f"chargées : {len(df)}, créneaux absents : {len(absents)}")
    if len(absents):
        jours = pd.Series(absents.date).value_counts().sort_index()
        print(f"  Jours concernés : {len(jours)} (ex. {', '.join(str(j) for j in jours.index[:5])})")

    print(f"Température : min {df['t_c'].min():.1f} °C, max {df['t_c'].max():.1f} °C")
    verifier_plausibilite(df)

    print("\nValeurs manquantes avant nettoyage :")
    print(rapport_manquants(df, COLONNES_METEO))
    print("\nManquants par mois (%) avant nettoyage, mois concernés :")
    par_mois = manquants_par_mois(df, COLONNES_METEO)
    print(par_mois[par_mois.sum(axis=1) > 0])

    propre = nettoyer(df)
    restants = rapport_manquants(propre, COLONNES_METEO)["nb"].sum()
    print(f"\nAprès nettoyage : {len(propre)} lignes, {restants} valeur(s) manquante(s) restante(s)")

    mensuel = agregation_mensuelle(propre)
    print("\nAgrégation mensuelle :")
    print(mensuel)
    horaire = agregation_horaire(propre)
    print("\nCycle diurne (température par heure UTC) :")
    print(horaire)

    print("\n=== Séance 3 : exploration visuelle ===")
    pluie = propre["rr3"]
    print(f"Précipitations 3 h : min {pluie.min()} mm, max {pluie.max()} mm, "
          f"{100 * (pluie == 0).mean():.1f} % de zéros")
    print(f"Mois le plus pluvieux : {mensuel['pluie_mm'].idxmax()} "
          f"({mensuel['pluie_mm'].max():.1f} mm)")
    corr = matrice_correlation(propre)
    print("\nCorrélations avec la température :")
    print(corr["t_c"].drop("t_c").sort_values(key=abs, ascending=False))

    figures = [
        figure_temperature(propre, mensuel, DOSSIER_FIGURES),
        figure_precipitations(propre, mensuel, DOSSIER_FIGURES),
        figure_correlations(corr, DOSSIER_FIGURES),
        figure_nuages(propre, DOSSIER_FIGURES),
        figure_boxplot_mois(propre, DOSSIER_FIGURES),
        figure_temp_humidite(mensuel, DOSSIER_FIGURES),
        figure_cycle_diurne(horaire, DOSSIER_FIGURES),
    ]

    print("\nFigures enregistrées :")
    for f in figures:
        print(f"  {f.relative_to(Path(__file__).parent.parent)} ({f.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    fichier = sys.argv[1] if len(sys.argv) > 1 else FICHIER_DONNEES
    try:
        main(fichier)
    except (FileNotFoundError, ValueError) as erreur:
        print(f"Erreur : {erreur}", file=sys.stderr)
        sys.exit(1)
