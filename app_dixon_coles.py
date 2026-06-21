"""
Model Dixon-Cole per a Porra World Cup 2026
---------------------------------------------
App Streamlit que reprodueix la lògica del notebook dixon_cole_model_v2.ipynb:
carrega de dades, neteja de noms, càlcul de lambda per Poisson ponderat,
correcció Dixon-Coles, regla de decisió signe-primer, i visualització Altair.

Per executar:
    streamlit run app_dixon_coles.py

Fitxers que ha de trobar a la mateixa carpeta (o pujar des de la barra lateral):
    - context_results_2021.csv
    - fifa_ranking_live_2026-06-10.csv
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.stats import poisson

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
st.set_page_config(page_title="Dixon-Cole · World Cup 2026", layout="wide")

GLOBAL_AVG_GOALS = 1.35       # mitjana històrica de gols per equip en partits internacionals
MIN_MATCHES = 8               # llindar per considerar l'històric "suficient"
HOME_ADV_FACTOR = 1.12        # bonus únic de camp (només per als amfitrions reals)
HOSTS = {"United States", "Mexico", "Canada"}
MAX_GOALS = 7                 # mida de la matriu de marcadors (0..MAX_GOALS-1 gols)
RHO_BOUNDS = (-0.15, -0.02)   # rang defensable segons la literatura Dixon-Coles

PATH_RESULTS_DEFAULT = "context_results_2021.csv"
PATH_RANKING_DEFAULT = "fifa_ranking_live_2026-06-10.csv"

# Mateixos 29 partits de la jornada 1 del Mundial que ja teníeu al notebook.
# is_host=True només té efecte real per a USA / Mexico / Canada com a local.
PARTIDS_PER_DEFECTE = [
    ("2026-06-11", "Mexico", "South Africa", True),
    ("2026-06-11", "Korea Republic", "Czechia", False),
    ("2026-06-12", "Canada", "Bosnia and Herzegovina", True),
    ("2026-06-12", "USA", "Paraguay", True),
    ("2026-06-13", "Haiti", "Scotland", False),
    ("2026-06-13", "Australia", "Türkiye", False),
    ("2026-06-13", "Brazil", "Morocco", False),
    ("2026-06-13", "Qatar", "Switzerland", False),
    ("2026-06-14", "Côte d'Ivoire", "Ecuador", False),
    ("2026-06-14", "Germany", "Curaçao", False),
    ("2026-06-14", "Netherlands", "Japan", False),
    ("2026-06-14", "Sweden", "Tunisia", False),
    ("2026-06-15", "Saudi Arabia", "Uruguay", False),
    ("2026-06-15", "Spain", "Cabo Verde", False),
    ("2026-06-15", "IR Iran", "New Zealand", False),
    ("2026-06-15", "Belgium", "Egypt", False),
    ("2026-06-16", "France", "Senegal", False),
    ("2026-06-16", "Iraq", "Norway", False),
    ("2026-06-16", "Argentina", "Algeria", False),
    ("2026-06-16", "Austria", "Jordan", False),
    ("2026-06-17", "Ghana", "Panama", False),
    ("2026-06-17", "England", "Croatia", False),
    ("2026-06-17", "Portugal", "Congo DR", False),
    ("2026-06-17", "Uzbekistan", "Colombia", False),
    ("2026-06-18", "Czechia", "South Africa", False),
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", False),
    ("2026-06-18", "Canada", "Qatar", True),
    ("2026-06-18", "Mexico", "Korea Republic", True),
    ("2026-06-19", "Brazil", "Haiti", False),
    ("2026-06-19", "Scotland", "Morocco", False),
    ("2026-06-19", "Türkiye", "Paraguay", False),
    ("2026-06-19", "USA", "Australia", True),
    ("2026-06-20", "Germany", "Côte d'Ivoire", False),
    ("2026-06-20", "Ecuador", "Curaçao", False),
    ("2026-06-20", "Netherlands", "Sweden", False),
    ("2026-06-20", "Tunisia", "Japan", False),
    ("2026-06-21", "Uruguay", "Cabo Verde", False),
    ("2026-06-21", "Spain", "Saudi Arabia", False),
    ("2026-06-21", "Belgium", "IR Iran", False),
    ("2026-06-21", "New Zealand", "Egypt", False),
    ("2026-06-22", "Norway", "Senegal", False),
    ("2026-06-22", "France", "Iraq", False),
    ("2026-06-22", "Argentina", "Austria", False),
    ("2026-06-22", "Jordan", "Algeria", False),
    ("2026-06-23", "England", "Ghana", False),
    ("2026-06-23", "Panama", "Croatia", False),
    ("2026-06-23", "Portugal", "Uzbekistan", False),
    ("2026-06-23", "Colombia", "Congo DR", False),
    ("2026-06-24", "Scotland", "Brazil", False),
    ("2026-06-24", "Morocco", "Haiti", False),
    ("2026-06-24", "Switzerland", "Canada", False),
    ("2026-06-24", "Bosnia and Herzegovina", "Qatar", False),
    ("2026-06-24", "Czechia", "Mexico", False),
    ("2026-06-24", "South Africa", "Korea Republic", False),
    ("2026-06-25", "Curaçao", "Côte d'Ivoire", False),
    ("2026-06-25", "Ecuador", "Germany", False),
    ("2026-06-25", "Japan", "Sweden", False),
    ("2026-06-25", "Tunisia", "Netherlands", False),
    ("2026-06-25", "Türkiye", "USA", False),
    ("2026-06-25", "Paraguay", "Australia", False),
    ("2026-06-26", "Norway", "France", False),
    ("2026-06-26", "Senegal", "Iraq", False),
    ("2026-06-26", "Egypt", "IR Iran", False),
    ("2026-06-26", "New Zealand", "Belgium", False),
    ("2026-06-26", "Cabo Verde", "Saudi Arabia", False),
    ("2026-06-26", "Uruguay", "Spain", False),
    ("2026-06-27", "Panama", "England", False),
    ("2026-06-27", "Croatia", "Ghana", False),
    ("2026-06-27", "Algeria", "Austria", False),
    ("2026-06-27", "Jordan", "Argentina", False),
    ("2026-06-27", "Colombia", "Portugal", False),
    ("2026-06-27", "Congo DR", "Uzbekistan", False),
]

# =============================================================================
# NETEJA DE NOMS
# =============================================================================
# Una única direcció canònica (cap al format de context_results_2021.csv) per
# evitar el mapping circular que teníem abans (p. ex. "Cape Verde"->"Cabo Verde"
# i "Cabo Verde"->"Cape Verde" alhora, que es contradeien).
ALIASES = {
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Korea Republic": "South Korea",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "CuraÃ§ao": "Curaçao",   # mojibake real present a context_results_2021.csv
    "Curacao": "Curaçao",
    "USA": "United States",
    "Bosnia": "Bosnia and Herzegovina",
}


def canon(name: str) -> str:
    name = str(name).strip()
    return ALIASES.get(name, name)


# =============================================================================
# CÀRREGA DE DADES (cacheada perquè no es recalculi a cada interacció)
# =============================================================================
@st.cache_data(show_spinner="Carregant històric i rànquing FIFA...")
def load_data(results_bytes_or_path, ranking_bytes_or_path):
    df_results = pd.read_csv(results_bytes_or_path)
    df_results["date"] = pd.to_datetime(df_results["date"])
    df_results = df_results.dropna(subset=["home_score", "away_score"]).copy()
    df_results["home_score"] = df_results["home_score"].astype(int)
    df_results["away_score"] = df_results["away_score"].astype(int)
    df_results["home_team"] = df_results["home_team"].map(canon)
    df_results["away_team"] = df_results["away_team"].map(canon)

    df_ranking = pd.read_csv(ranking_bytes_or_path)
    df_ranking["country_full"] = df_ranking["country_full"].map(canon)
    ranking_points = dict(zip(df_ranking["country_full"], df_ranking["total_points"]))

    return df_results, ranking_points


@st.cache_data(show_spinner=False)
def estimate_rho(df_results: pd.DataFrame) -> float:
    """
    Estima rho per moment-matching: compara la taxa de taules observada amb
    la que prediria un Poisson independent amb les mitjanes globals de gols.

    Nota honesta: és un estimador agregat senzill (no l'MLE conjunt complet
    de Dixon-Coles), i a nivell agregat el senyal és feble i pot sortir amb
    el signe "equivocat" per soroll d'heterogeneïtat entre partits molt
    desiguals. Per això s'acota al rang típic de la literatura
    (entre -0.15 i -0.02) en lloc de deixar-lo saturar a 0 o canviar de signe.
    """
    draws = df_results[df_results["home_score"] == df_results["away_score"]]
    observed_draw_rate = len(draws) / len(df_results)

    lam_avg = df_results["home_score"].mean()
    mu_avg = df_results["away_score"].mean()
    expected_draw_rate = sum(poisson.pmf(k, lam_avg) * poisson.pmf(k, mu_avg) for k in range(15))

    raw_rho = -(observed_draw_rate - expected_draw_rate) / (expected_draw_rate + 1e-9)
    return float(np.clip(raw_rho, RHO_BOUNDS[0], RHO_BOUNDS[1]))


# =============================================================================
# MOTOR ESTADÍSTIC
# =============================================================================
def get_team_stats(team_name, df_history, ranking_dict, target_date):
    """Força ofensiva/defensiva ponderada per recència, tipus de partit i força del rival."""
    df_past = df_history[df_history["date"] < target_date]
    home_matches = df_past[df_past["home_team"] == team_name].copy()
    away_matches = df_past[df_past["away_team"] == team_name].copy()

    home_matches["goals_scored"], home_matches["goals_conceded"] = home_matches["home_score"], home_matches["away_score"]
    home_matches["opp_team"] = home_matches["away_team"]
    away_matches["goals_scored"], away_matches["goals_conceded"] = away_matches["away_score"], away_matches["home_score"]
    away_matches["opp_team"] = away_matches["home_team"]

    df_team = pd.concat([home_matches, away_matches])
    if len(df_team) == 0:
        return GLOBAL_AVG_GOALS, GLOBAL_AVG_GOALS, 0

    days_ago = (target_date - df_team["date"]).dt.days
    time_weight = np.exp(-days_ago / 730)  # vida mitjana ~24 mesos
    match_weight = np.where(df_team["tournament"] == "Friendly", 0.5, 1.0)
    df_team["weight"] = time_weight * match_weight

    avg_ranking_points = np.mean(list(ranking_dict.values()))
    # Un equipo que ni siquiera aparece en el ranking FIFA (microfederaciones caribeñas,
    # equipos no afiliados a FIFA, etc.) casi seguro es MÁS DÉBIL que la media, no de fuerza
    # media. Usar la media aquí infla artificialmente las goleadas contra estos rivales.
    fallback_unranked = np.percentile(list(ranking_dict.values()), 5)
    df_team["opp_points"] = df_team["opp_team"].map(ranking_dict).fillna(fallback_unranked)
    df_team["opp_factor"] = df_team["opp_points"] / avg_ranking_points

    weighted_offense = np.average(df_team["goals_scored"] * df_team["opp_factor"], weights=df_team["weight"])
    weighted_defense = np.average(df_team["goals_conceded"] / df_team["opp_factor"], weights=df_team["weight"])
    return weighted_offense, weighted_defense, len(df_team)


def calculate_lambdas(home_team, away_team, df_history, ranking_dict, match_date, is_host_country):
    off_h, def_h, n_h = get_team_stats(home_team, df_history, ranking_dict, match_date)
    off_a, def_a, n_a = get_team_stats(away_team, df_history, ranking_dict, match_date)

    data_driven_h = (off_h * def_a) / GLOBAL_AVG_GOALS
    data_driven_a = (off_a * def_h) / GLOBAL_AVG_GOALS
    base_h, base_a = data_driven_h, data_driven_a

    home_advantage = HOME_ADV_FACTOR if (is_host_country and home_team in HOSTS) else 1.0

    return base_h * home_advantage, base_a, n_h, n_a


def dixon_coles_adjustment(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return max(0.0, 1 - lam * mu * rho)
    if x == 0 and y == 1:
        return max(0.0, 1 + lam * rho)
    if x == 1 and y == 0:
        return max(0.0, 1 + mu * rho)
    if x == 1 and y == 1:
        return max(0.0, 1 - rho)
    return 1.0


def score_matrix(lam, mu, rho, max_goals=MAX_GOALS):
    M = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            M[x, y] = poisson.pmf(x, lam) * poisson.pmf(y, mu) * dixon_coles_adjustment(x, y, lam, mu, rho)
    return M / M.sum()


def predict_match(home, away, lam, mu, rho):
    if np.isnan(lam) or lam <= 0:
        lam = GLOBAL_AVG_GOALS
    if np.isnan(mu) or mu <= 0:
        mu = GLOBAL_AVG_GOALS

    M = score_matrix(lam, mu, rho)
    p1, px, p2 = float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())

    probs = {"1": p1, "X": px, "2": p2}
    sign = max(probs, key=probs.get)
    region = np.tril(M, -1) if sign == "1" else (np.diag(np.diag(M)) if sign == "X" else np.triu(M, 1))
    hg, ag = np.unravel_index(np.argmax(region), region.shape)

    sign_prob = probs[sign]
    confianca = "Alta" if sign_prob >= 0.55 else ("Mitjana" if sign_prob >= 0.40 else "Baixa")

    return {
        "Partit": f"{home} vs {away}",
        "Marcador suggerit": f"{hg}-{ag}",
        "λ (local-visitant)": f"{lam:.2f}–{mu:.2f}",
        "Confiança": confianca,
        "% 1X2 (1/X/2)": f"{p1*100:.0f} / {px*100:.0f} / {p2*100:.0f}",
        "_lam": lam, "_mu": mu, "_p1": p1, "_px": px, "_p2": p2,
    }


# =============================================================================
# VISUALITZACIÓ ALTAIR (mateix format que al notebook)
# =============================================================================
def build_match_charts(home_team, away_team, lambda_h, lambda_a, rho, max_goals=MAX_GOALS):
    M = score_matrix(lambda_h, lambda_a, rho, max_goals)
    goals = list(range(max_goals))

    p1, px, p2 = float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())
    sign = "1" if p1 >= px and p1 >= p2 else ("X" if px >= p1 and px >= p2 else "2")
    region = np.tril(M, -1) if sign == "1" else (np.diag(np.diag(M)) if sign == "X" else np.triu(M, 1))
    bi, bj = np.unravel_index(np.argmax(region), region.shape)
    conf = "Alta" if max(p1, px, p2) >= 0.55 else ("Mitjana" if max(p1, px, p2) >= 0.40 else "Baixa")

    # --- 1. Distribució de Poisson ---
    df_poisson = pd.DataFrame(
        [{"gols": k, "prob": float(poisson.pmf(k, lambda_h)), "equip": home_team} for k in goals]
        + [{"gols": k, "prob": float(poisson.pmf(k, lambda_a)), "equip": away_team} for k in goals]
    )
    chart_poisson = (
        alt.Chart(df_poisson)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("gols:O", title="Gols", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("prob:Q", title="Probabilitat", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "equip:N",
                scale=alt.Scale(domain=[home_team, away_team], range=["#378ADD", "#D85A30"]),
                legend=alt.Legend(title=None, orient="top", labelFontSize=11),
            ),
            xOffset="equip:N",
            tooltip=[
                alt.Tooltip("equip:N", title="Equip"),
                alt.Tooltip("gols:O", title="Gols"),
                alt.Tooltip("prob:Q", title="P", format=".1%"),
            ],
        )
        .properties(title="Distribució de Poisson", width=220, height=280)
    )

    # --- 2. Matriu (heatmap) de marcadors ---
    df_heat = pd.DataFrame(
        [
            {"local": i, "visitant": j, "prob": float(M[i, j]), "marcador": f"{i}–{j}", "suggerit": bool(i == bi and j == bj)}
            for i in goals
            for j in goals
        ]
    )
    max_prob = float(df_heat["prob"].max())
    base = alt.Chart(df_heat)

    rect = base.mark_rect().encode(
        x=alt.X("visitant:O", title="Gols visitant", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("local:O", title="Gols local", sort="descending"),
        color=alt.Color("prob:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="P", format=".1%", labelFontSize=10)),
        tooltip=[alt.Tooltip("marcador:N", title="Marcador"), alt.Tooltip("prob:Q", title="Probabilitat", format=".2%")],
    )
    text = base.mark_text(fontSize=9).encode(
        x=alt.X("visitant:O"),
        y=alt.Y("local:O", sort="descending"),
        text=alt.Text("prob:Q", format=".1%"),
        color=alt.condition(alt.datum.prob > max_prob * 0.5, alt.value("white"), alt.value("#333333")),
    )
    border = (
        base.mark_rect(filled=False, stroke="#D85A30", strokeWidth=2.5)
        .encode(x=alt.X("visitant:O"), y=alt.Y("local:O", sort="descending"))
        .transform_filter(alt.datum.suggerit)
    )
    chart_heat = (rect + text + border).properties(
        title=f"Matriu de marcador · suggerit: {bi}–{bj} ({conf})", width=280, height=280
    )

    # --- 3. Probabilitats 1X2 ---
    df_sign = pd.DataFrame({"resultat": ["Local (1)", "Empat (X)", "Visitant (2)"], "prob": [p1, px, p2]})
    chart_sign = (
        alt.Chart(df_sign)
        .mark_bar(opacity=0.85, size=40)
        .encode(
            x=alt.X("resultat:N", title=None, sort=["Local (1)", "Empat (X)", "Visitant (2)"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("prob:Q", title="Probabilitat", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "resultat:N",
                scale=alt.Scale(domain=["Local (1)", "Empat (X)", "Visitant (2)"], range=["#378ADD", "#888780", "#D85A30"]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("resultat:N", title="Resultat"), alt.Tooltip("prob:Q", title="P", format=".1%")],
        )
        .properties(title="Probabilitats 1X2", width=160, height=280)
    )

    return (
        alt.hconcat(
            chart_poisson,
            chart_heat,
            chart_sign,
            title=alt.TitleParams(
                text=f"{home_team} vs {away_team}  ·  λ {lambda_h:.2f}–{lambda_a:.2f}",
                fontSize=13,
                fontWeight="bold",
                anchor="middle",
            ),
            spacing=20,
        )
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=11)
    )


# =============================================================================
# INTERFÍCIE
# =============================================================================
# =============================================================================
# INTERFÍCIE
# =============================================================================
st.title("Model Dixon-Cole per a Porra World Cup 2026")
st.caption(
    "Model de Poisson bivariant amb correcció Dixon-Coles, ajustat amb dades històriques "
    "(2021–2026) i rànquing FIFA segons finals Maig 2026."
)
st.caption(
    " Autor: Albert Maristany per a porra oficial de Marcel Gómez - Mundial 2026"
)

try:
    df_results, ranking_points = load_data(PATH_RESULTS_DEFAULT, PATH_RANKING_DEFAULT)
except FileNotFoundError:
    st.error(
        f"No trobo `{PATH_RESULTS_DEFAULT}` ni `{PATH_RANKING_DEFAULT}` al repositori. "
        "Comprova que els dos CSV estiguin a l'arrel del repo (mateixa carpeta que aquest fitxer)."
    )
    st.stop()

RHO = estimate_rho(df_results)

rows = []
for fecha_str, eq_home, eq_away, is_host in PARTIDS_PER_DEFECTE:
    eq_home_c, eq_away_c = canon(eq_home), canon(eq_away)
    match_date = pd.to_datetime(fecha_str)
    lam, mu, n_h, n_a = calculate_lambdas(eq_home_c, eq_away_c, df_results, ranking_points, match_date, is_host)
    rows.append(predict_match(eq_home, eq_away, lam, mu, RHO))
df_pred = pd.DataFrame(rows)

tab_prediccions, tab_model = st.tabs(["📊 Prediccions", "🧮 Model i metodologia"])

# =============================================================================
# PESTANYA 1 — PREDICCIONS (per defecte, orientada a l'usuari)
# =============================================================================
with tab_prediccions:
    st.subheader("Totes les prediccions — Fase de grups")
    display_cols = ["Partit", "Marcador suggerit", "λ (local-visitant)", "Confiança", "% 1X2 (1/X/2)"]
    st.dataframe(df_pred[display_cols], width="stretch", hide_index=True)

    st.divider()
    st.subheader("Tria un partit per veure'n el detall")
    st.caption("Canvia el partit aquí sota per explorar la distribució de Poisson, la matriu de marcadors i el 1X2.")

    partit_triat = st.selectbox("Partit:", df_pred["Partit"].tolist(), label_visibility="collapsed")
    fila = df_pred[df_pred["Partit"] == partit_triat].iloc[0]

    chart = build_match_charts(
        partit_triat.split(" vs ")[0],
        partit_triat.split(" vs ")[1],
        fila["_lam"],
        fila["_mu"],
        RHO,
    )
    st.altair_chart(chart, width="content")

    st.caption(
        f"Marcador suggerit: **{fila['Marcador suggerit']}** · Confiança: **{fila['Confiança']}** · "
        f"1X2: **{fila['% 1X2 (1/X/2)']}**"
    )

# =============================================================================
# PESTANYA 2 — MODEL I RAONAMENT
# =============================================================================
with tab_model:
    st.markdown(
        """
**Hipòtesi inicial.** La proposta de l'anàlisi és que un enfocament científic, rigurós i data-driven,
augmentat amb eines d'IA, sobre un problema de predicció de dades pot guanyar a qualsevol "ball-knowledge".
A 21 de juny de 2026 (38 partits jugats de fase de grups), això no es compleix.
"""
    )

    st.markdown(
        """
**Role-playing agents.** S'ha començat simulant una taula rodona entre 3 experts amb IA:
un/a científic/a de dades sènior, un/a estadístic/a sènior i un/a expert/a en prediccions esportives.
Els 3 han debatut (1) si era possible guanyar a rivals humans en una porra d'un mundial, i (2) quin
era l'approach més efectiu. S'ha conclòs que la variant de Dixon-Coles de la distribució de Poisson
era el "state of the art" i la millor opció (resultat/energia). Es complementa un model construït
amb dades de partits des de 2021 amb el rànquing de la FIFA a finals de maig de 2026 per a les seleccions
amb pocs resultats de partits.
"""
    )

    st.markdown(
        """
**Notebooks amb Python i visualitzacions amb Altair.** S'ha construït el model (amb l'ajuda de models
d'IA gratuïts, poc potents) en notebooks de Jupyter, s'ha optimitzat i després s'ha projectat a una
app web amb Streamlit.
"""
    )

    st.markdown(
        """
**Idea de base de la metodologia.** Els gols d'un partit de futbol se solen modelar com a variables de
Poisson: cadascun dels dos equips marca un nombre de gols $X \\sim \\text{Poisson}(\\lambda)$, on
$\\lambda$ és l'esperança de gols. El repte és estimar $\\lambda_{local}$ i $\\lambda_{visitant}$ a partir
de l'històric, i corregir el biaix que el Poisson independent introdueix als marcadors baixos.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Càlcul de λ (cel·la `get_team_stats` / `calculate_lambdas`)**")
        st.markdown(
            """
Per a cada equip, es prenen tots els seus partits **anteriors a la data del partit a predir**
(per evitar fuga d'informació) i es pondera cada partit per:

- **Recència** — decaïment exponencial amb vida mitjana ≈ 24 mesos.
- **Tipus de partit** — els amistosos pesen la meitat que els competitius.
- **Força del rival** — un gol contra un equip top val més que un gol contra un equip dèbil
  (factor calculat amb els `total_points` del rànquing FIFA).
"""
        )
        st.latex(r"\text{ofensiva}_i = \frac{\sum_t w_t \cdot \text{gols}_t \cdot \text{opp\_factor}_t}{\sum_t w_t}")
        st.latex(r"\lambda_{local} = \dfrac{\text{ofensiva}_{local} \times \text{defensa}_{visitant}}{\bar{g}} \times \gamma_{camp}")

    with col2:
        st.markdown("**Correcció Dixon-Coles (cel·la `dixon_coles_adjustment`)**")
        st.markdown(
            "El Poisson independent infravalora sistemàticament els marcadors baixos "
            "(0-0, 1-0, 0-1, 1-1). La correcció $\\tau$ de Dixon-Coles ajusta exactament aquestes "
            "quatre caselles amb un paràmetre $\\rho$:"
        )
        st.latex(
            r"""
            \tau(x,y)=\begin{cases}
            1-\lambda\mu\rho & x{=}0,y{=}0\\
            1+\lambda\rho & x{=}0,y{=}1\\
            1+\mu\rho & x{=}1,y{=}0\\
            1-\rho & x{=}1,y{=}1\\
            1 & \text{altrament}
            \end{cases}
            """
        )
        st.markdown(f"En aquesta execució, $\\rho$ s'estima a partir de les pròpies dades i queda acotat "
                    f"al rang `{RHO_BOUNDS}` (rang típic a la literatura): **ρ = {RHO:.3f}**.")

    st.markdown(
        """
**Regla de decisió (cel·la `predict_match`).** Primer es calcula la probabilitat de cada signe
1X2 sumant la part corresponent de la matriu de marcadors; **després**, i només dins del signe
guanyador, es busca el marcador exacte més probable. Evita el biaix de triar el marcador més
probable en absolut (que sol distorsionar el signe).

**Avantatge de camp.** Només s'aplica un factor multiplicatiu (`+12%` a λ_local) quan el local és
de veritat un dels tres amfitrions (Estats Units, Mèxic o Canadà); la resta de partits es tracten
com a camp neutral.
"""
    )

