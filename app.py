# -*- coding: utf-8 -*-
"""
Service d'analyse statistique — Cabinet HC
==============================================================
Ce petit serveur tourne en local sur votre ordinateur (http://localhost:8420)
et prend en charge TOUS les calculs et décisions statistiques sur les
séries de ventes (tendance, croissance, prévisions, saisonnalité).

L'IA (Claude/Gemini), elle, ne reçoit plus que les chiffres déjà calculés
ici : son rôle est uniquement de les mettre en mots, de structurer
l'argumentaire et d'assurer la cohérence du rapport — jamais de "deviner"
une tendance ou une prévision elle-même.

Comment ça marche :
- Vous lancez ce service une fois (voir demarrer.bat), il reste actif
  en arrière-plan tant que votre ordinateur est allumé.
- L'outil (le fichier index.html) l'interroge automatiquement à chaque
  analyse de ventes (Packs 1, 2, 3). Si ce service n'est pas lancé,
  l'outil bascule automatiquement sur son moteur de secours local
  (moins poussé, mais l'outil ne casse jamais).

Méthode choisie automatiquement selon la quantité de données disponibles
(c'est la bonne pratique statistique : plus une série est courte, plus la
méthode doit être simple pour éviter de "voir" des motifs qui n'existent pas) :
- Moins de 8 points   -> régression linéaire simple (tendance de fond uniquement)
- 8 à 23 points        -> lissage exponentiel double (méthode de Holt : tendance + inertie)
- 24 points ou plus    -> décomposition saisonnière additive + lissage de Holt sur la
                          série "dé-saisonnalisée" (permet de distinguer un vrai
                          changement de tendance d'un simple effet saisonnier récurrent)
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)


# ------------------------------------------------------------------
# CORS manuel (pas de dépendance flask-cors à installer) : nécessaire
# car l'outil est ouvert en tant que fichier local (origine "null" pour
# le navigateur), donc on autorise explicitement toutes les origines.
# ------------------------------------------------------------------
@app.after_request
def ajouter_entetes_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


import re


# ------------------------------------------------------------------
# EXTRACTION DE FAITS CHIFFRÉS DANS UN TEXTE BRUT (Diagnostic de Performance)
# ------------------------------------------------------------------
# Objectif : repérer mécaniquement, dans des notes de terrain en vrac,
# les montants, pourcentages, dates et quantités déjà présents dans le
# texte — pour les donner à l'IA comme repères vérifiés, en plus du
# texte complet. Ceci ne remplace PAS la compréhension du texte par
# l'IA (repérer le sens, les causes, les enjeux reste son rôle) : c'est
# un simple filet de sécurité mécanique contre un chiffre qui passerait
# inaperçu dans un texte long et désordonné.
# ------------------------------------------------------------------

MOIS_FR = 'janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre'

RE_MONTANT = re.compile(
    r'(?P<valeur>\d{1,3}(?:[ .]\d{3})*(?:,\d+)?)\s*(?P<devise>FCFA|F\s?CFA|CFA|F\b|francs?|€|euros?|\$|USD|dollars?)',
    re.IGNORECASE,
)
RE_POURCENTAGE = re.compile(r'(?P<valeur>\d{1,3}(?:[.,]\d+)?)\s*%')
RE_DATE = re.compile(
    r'(?:(?P<jour>\d{1,2})\s+)?\b(?P<mois>' + MOIS_FR + r')\b\s*(?P<annee>\d{4})?',
    re.IGNORECASE,
)
RE_DATE_NUM = re.compile(r'\b(?P<j>\d{1,2})[/\-](?P<m>\d{1,2})[/\-](?P<a>\d{2,4})\b')
RE_QUANTITE = re.compile(
    r'(?P<valeur>\d{1,3}(?:[ .]\d{3})*)\s*(?P<unite>clients?|employés?|employes?|salariés?|salaries?|unités?|unites?|pièces?|pieces?|commandes?|jours?|semaines?|mois|kg|kilos?|sacs?|cartons?|litres?|points?\s+de\s+vente|magasins?|boutiques?)',
    re.IGNORECASE,
)


def contexte_autour(texte, debut, fin, marge=35):
    d = max(0, debut - marge)
    f = min(len(texte), fin + marge)
    extrait = texte[d:f].replace('\n', ' ').strip()
    return ('…' if d > 0 else '') + extrait + ('…' if f < len(texte) else '')


def extraire_faits(texte):
    faits = []

    for m in RE_MONTANT.finditer(texte):
        faits.append({
            'type': 'montant',
            'texte_trouve': m.group(0).strip(),
            'contexte': contexte_autour(texte, m.start(), m.end()),
            'position': m.start(),
        })

    for m in RE_POURCENTAGE.finditer(texte):
        faits.append({
            'type': 'pourcentage',
            'texte_trouve': m.group(0).strip(),
            'contexte': contexte_autour(texte, m.start(), m.end()),
            'position': m.start(),
        })

    for m in RE_DATE.finditer(texte):
        faits.append({
            'type': 'date',
            'texte_trouve': m.group(0).strip(),
            'contexte': contexte_autour(texte, m.start(), m.end()),
            'position': m.start(),
        })

    for m in RE_DATE_NUM.finditer(texte):
        faits.append({
            'type': 'date',
            'texte_trouve': m.group(0).strip(),
            'contexte': contexte_autour(texte, m.start(), m.end()),
            'position': m.start(),
        })

    for m in RE_QUANTITE.finditer(texte):
        faits.append({
            'type': 'quantite',
            'texte_trouve': m.group(0).strip(),
            'contexte': contexte_autour(texte, m.start(), m.end()),
            'position': m.start(),
        })

    faits.sort(key=lambda f: f['position'])
    for f in faits:
        del f['position']
    return faits


@app.route('/extraire-faits', methods=['OPTIONS'])
def preflight_extraction():
    return ('', 204)


@app.route('/extraire-faits', methods=['POST'])
def route_extraire_faits():
    payload = request.get_json(force=True, silent=True) or {}
    texte = (payload.get('texte') or '').strip()
    if not texte:
        return jsonify({'faits': []})
    if len(texte) > 20000:
        texte = texte[:20000]
    faits = extraire_faits(texte)
    return jsonify({
        'nombre_faits_detectes': len(faits),
        'faits': faits,
    })


@app.route('/analyser-serie', methods=['OPTIONS'])
def preflight():
    return ('', 204)


@app.route('/sante', methods=['GET'])
def sante():
    """Permet à l'outil de vérifier rapidement si le service tourne."""
    return jsonify({'statut': 'ok', 'service': 'Cabinet HC - Analyse Python'})


def regression_lineaire(valeurs):
    """Régression linéaire simple (numpy.polyfit) + R² (qualité d'ajustement)."""
    n = len(valeurs)
    x = np.arange(n)
    y = np.array(valeurs, dtype=float)
    if n < 2:
        return {'pente': 0.0, 'ordonnee_origine': float(y[0]) if n else 0.0, 'r2': 0.0}
    pente, ordonnee = np.polyfit(x, y, 1)
    y_pred = pente * x + ordonnee
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'pente': float(pente), 'ordonnee_origine': float(ordonnee), 'r2': float(r2)}


def croissance(valeurs):
    """Croissance globale (premier -> dernier point) et croissance moyenne par période."""
    n = len(valeurs)
    if n < 2:
        return {'globale_pct': 0.0, 'moyenne_par_periode_pct': 0.0}
    premier, dernier = valeurs[0], valeurs[-1]
    globale = ((dernier - premier) / premier * 100) if premier else 0.0
    variations = []
    for i in range(1, n):
        if valeurs[i - 1]:
            variations.append((valeurs[i] - valeurs[i - 1]) / valeurs[i - 1] * 100)
    moyenne = float(np.mean(variations)) if variations else 0.0
    return {'globale_pct': float(globale), 'moyenne_par_periode_pct': moyenne}


def holt_lineaire(valeurs, alpha=0.3, beta=0.1, n_prevision=3):
    """
    Lissage exponentiel double (méthode de Holt) : capture un niveau ET une
    tendance qui s'ajustent progressivement, plus robuste qu'une simple
    droite sur des séries courtes à moyennes, sans halluciner de saisonnalité
    qu'on n'a pas assez de données pour confirmer.
    """
    valeurs = list(valeurs)
    niveau = valeurs[0]
    tendance = valeurs[1] - valeurs[0] if len(valeurs) > 1 else 0.0
    for i in range(1, len(valeurs)):
        dernier_niveau = niveau
        niveau = alpha * valeurs[i] + (1 - alpha) * (niveau + tendance)
        tendance = beta * (niveau - dernier_niveau) + (1 - beta) * tendance
    prevision = [float(niveau + (h + 1) * tendance) for h in range(n_prevision)]
    return {'niveau_final': float(niveau), 'tendance_par_periode': float(tendance), 'prevision_prochaines_periodes': prevision}


def decomposition_saisonniere_additive(valeurs, periode=12, n_prevision=3):
    """
    Décomposition classique additive (valeur = tendance + saisonnalité + résidu),
    calculée avec une moyenne mobile centrée — la méthode de référence pour isoler
    un effet saisonnier récurrent d'une vraie tendance de fond. Nécessite au moins
    2 cycles complets (24 points pour une saisonnalité mensuelle) pour être fiable.
    """
    s = pd.Series(valeurs, dtype=float)
    tendance = s.rolling(window=periode, center=True).mean()
    detrend = s - tendance
    indices_saisonniers = detrend.groupby(s.index % periode).transform('mean')
    # Normalisation : la somme des coefficients saisonniers sur un cycle doit être ~0
    moyenne_saisonniere = indices_saisonniers.groupby(s.index % periode).mean() if False else None
    coefs = [float(detrend[(s.index % periode) == m].mean()) if not detrend[(s.index % periode) == m].dropna().empty else 0.0 for m in range(periode)]
    coefs_centres = [c - (sum(coefs) / periode) for c in coefs]

    desaisonnalise = [float(v) - coefs_centres[i % periode] for i, v in enumerate(valeurs)]
    holt_sur_desaisonnalise = holt_lineaire(desaisonnalise, n_prevision=n_prevision)
    prevision_avec_saison = [
        holt_sur_desaisonnalise['prevision_prochaines_periodes'][h] + coefs_centres[(len(valeurs) + h) % periode]
        for h in range(n_prevision)
    ]
    amplitude_saisonniere = float(max(coefs_centres) - min(coefs_centres))
    return {
        'coefficients_saisonniers_par_position': [round(c, 2) for c in coefs_centres],
        'amplitude_saisonniere': amplitude_saisonniere,
        'tendance_desaisonnalisee_par_periode': holt_sur_desaisonnalise['tendance_par_periode'],
        'prevision_prochaines_periodes': [float(p) for p in prevision_avec_saison],
    }


def calculer_diagnostic(valeurs, n_prevision=3):
    n = len(valeurs)
    if n < 2:
        return {'erreur': "Pas assez de données exploitables pour une analyse statistique (2 points minimum)."}

    reg = regression_lineaire(valeurs)
    croiss = croissance(valeurs)
    avg_value = float(np.mean(valeurs))

    # Même seuil que l'ancien outil (Cabinet HC) : pente relative à la valeur moyenne
    relative_slope = (reg['pente'] / abs(avg_value)) if avg_value else 0.0
    if relative_slope > 0.01:
        tendance_qualitative = 'Hausse'
    elif relative_slope < -0.01:
        tendance_qualitative = 'Baisse'
    else:
        tendance_qualitative = 'Stable'

    if n < 8:
        methode = 'regression_lineaire_simple'
        raison = f"Série de seulement {n} points : une régression linéaire simple est la méthode la plus fiable (pas assez de données pour une tendance non-linéaire ou une saisonnalité)."
        details_methode = {}
    elif n < 24:
        methode = 'lissage_exponentiel_holt'
        raison = f"Série de {n} points : suffisant pour un lissage exponentiel avec tendance (méthode de Holt), mais pas assez pour confirmer une saisonnalité (il faudrait au moins 24 points, soit 2 cycles annuels complets)."
        details_methode = holt_lineaire(valeurs, n_prevision=n_prevision)
    else:
        methode = 'decomposition_saisonniere_additive'
        raison = f"Série de {n} points (au moins 2 cycles complets) : une décomposition saisonnière additive permet de distinguer la vraie tendance de fond d'un effet saisonnier récurrent."
        details_methode = decomposition_saisonniere_additive(valeurs, n_prevision=n_prevision)

    return {
        'n_points': n,
        'methode_choisie': methode,
        'raison_du_choix': raison,
        'tendance_qualitative': tendance_qualitative,
        'valeur_moyenne': avg_value,
        'tendance_lineaire': reg,
        'croissance': croiss,
        'resultats_methode_choisie': details_methode,
    }


@app.route('/analyser-serie', methods=['POST'])
def analyser_serie():
    payload = request.get_json(force=True, silent=True) or {}
    serie = payload.get('serie') or []
    valeurs = [p.get('value') for p in serie if isinstance(p.get('value'), (int, float))]
    if not valeurs and payload.get('valeurs'):
        # Format simplifié : {"valeurs": [1200, 1300, ...]} — pratique depuis Make/no-code
        valeurs = [float(v) for v in payload.get('valeurs', []) if isinstance(v, (int, float, str)) and str(v).strip() != '']
    n_prevision = payload.get('n_prevision', 3)
    try:
        n_prevision = max(1, min(int(n_prevision), 12))
    except (TypeError, ValueError):
        n_prevision = 3

    resultat = calculer_diagnostic(valeurs, n_prevision)
    if 'erreur' in resultat:
        return jsonify(resultat), 400
    return jsonify(resultat)


@app.route('/analyser-fichier', methods=['OPTIONS'])
def preflight_fichier():
    return ('', 204)


@app.route('/analyser-fichier', methods=['POST'])
def analyser_fichier():
    """
    Reçoit directement le texte BRUT d'un fichier CSV dans le corps de la requête
    (Content-Type: text/plain ou text/csv — pas de JSON), colonnes Date, Montant
    (voir Gabarit_Diagnostic_Flash.csv). Fait tout le travail ici : parsing,
    détection de colonne, calcul de tendance. Pensé pour un appel HTTP direct
    depuis Make, sans logique de parsing côté no-code.
    Paramètre optionnel en query string : ?n_prevision=3
    """
    csv_text = request.get_data(as_text=True) or ''
    if not csv_text.strip():
        return jsonify({'erreur': "Aucun contenu de fichier reçu (corps de requête vide)."}), 400

    try:
        n_prevision = max(1, min(int(request.args.get('n_prevision', 3)), 12))
    except (TypeError, ValueError):
        n_prevision = 3

    try:
        import io
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        return jsonify({'erreur': f"Impossible de lire le fichier comme un CSV valide : {e}"}), 400

    # Détection de la colonne de montants : nom explicite, sinon colonne la plus numérique
    montant_col = None
    for col in df.columns:
        if re.search(r'montant|vente|revenu|valeur|^ca$|sales|amount|revenue|chiffre', str(col), re.IGNORECASE):
            montant_col = col
            break
    if montant_col is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            montant_col = numeric_cols[0]

    if montant_col is None:
        return jsonify({'erreur': "Aucune colonne de montants détectée dans le fichier. Vérifiez le gabarit (colonnes 'Date' et 'Montant')."}), 400

    valeurs = pd.to_numeric(df[montant_col], errors='coerce').dropna().tolist()
    resultat = calculer_diagnostic(valeurs, n_prevision)
    if 'erreur' in resultat:
        return jsonify(resultat), 400
    resultat['colonne_utilisee'] = str(montant_col)
    return jsonify(resultat)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8420))
    print('=' * 60)
    print(f' Service d\'analyse H-Engine — actif sur le port {port}')
    print('=' * 60)
    app.run(host='0.0.0.0', port=port)
