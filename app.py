from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

# ── Feature lists — Must match your exported CSVs exactly ────────────────────

OUTFIELD_FEATURES = [
    'height(cm)', 'weight(kg)', 'avg_goals_per_game',
    'team_index', 'showman_index', 'top_speed',
    'runner_index', 'aggression_index'
]

GK_FEATURES = [
    'saves_per_90', 'clean_sheets_per_90', 'goals_conceded_per_90',
    'punches_made_per_90', 'height(cm)', 'weight(kg)', 'saves_on_penalty'
]

# Slider key (from frontend) → CSV column name (in dataset)
OUTFIELD_KEY_MAP = {
    'height': 'height(cm)', 'weight': 'weight(kg)', 'goals_per_game': 'avg_goals_per_game',
    'team_index': 'team_index', 'showman': 'showman_index', 'top_speed': 'top_speed',
    'runner_index': 'runner_index', 'aggression': 'aggression_index'
}

GK_KEY_MAP = {
    'saves_per_90': 'saves_per_90', 'clean_sheets_p90': 'clean_sheets_per_90',
    'goals_conc_p90': 'goals_conceded_per_90', 'punches_p90': 'punches_made_per_90',
    'height': 'height(cm)', 'weight': 'weight(kg)', 'pen_saves': 'saves_on_penalty'
}

# ── Load separate CSVs (No more NaNs!) ────────────────────────────────────────

try:
    outfield_data = pd.read_csv('outfield_data.csv')
    gk_data = pd.read_csv('gk_data.csv')
    print(f"✅ Data Loaded: {len(outfield_data)} outfielders, {len(gk_data)} goalkeepers")
except FileNotFoundError:
    print("❌ Error: 'outfield_data.csv' or 'gk_data.csv' not found. Export them from your notebook first!")

# ── Fit Scalers and KNN Models ────────────────────────────────────────────────

scaler_out = StandardScaler().fit(outfield_data[OUTFIELD_FEATURES])
out_scaled = scaler_out.transform(outfield_data[OUTFIELD_FEATURES])
knn_out = NearestNeighbors(n_neighbors=5).fit(out_scaled)

scaler_gk = StandardScaler().fit(gk_data[GK_FEATURES])
gk_scaled = scaler_gk.transform(gk_data[GK_FEATURES])
knn_gk = NearestNeighbors(n_neighbors=5).fit(gk_scaled)

# ── Helper: Match % Calculation ───────────────────────────────────────────────

def compute_match_pct(user_input_df, player_row, features, source_df):
    similarity_sum = 0
    breakdown = []

    for f in features:
        target_val = float(user_input_df.iloc[0][f])
        player_val = float(player_row[f])
        min_f, max_f = source_df[f].min(), source_df[f].max()
        
        feat_range = max_f - min_f if max_f != min_f else 1
        diff = abs(target_val - player_val) / feat_range
        sim = max(0, 1 - diff)
        similarity_sum += sim

        breakdown.append({
            'feature': f,
            'player_val': round(player_val, 2),
            'target_val': round(target_val, 2),
            'pct': int(sim * 100)
        })

    match_pct = round((similarity_sum / len(features)) * 100, 1)
    return match_pct, breakdown

# ── /recommend endpoint ───────────────────────────────────────────────────────

@app.route('/recommend', methods=['POST'])
def recommend():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'No JSON body provided'}), 400

    position = body.get('position', '').lower()
    vals = body.get('values', {})

    # 🛡️ ERROR CHECK: Validate Slider Keys
    required_keys = OUTFIELD_KEY_MAP.keys() if position == 'outfielder' else GK_KEY_MAP.keys()
    missing_keys = [k for k in required_keys if k not in vals]
    
    if missing_keys:
        return jsonify({
            'error': f'Missing slider values for {position}',
            'missing_keys': missing_keys
        }), 400

    try:
        if position == 'outfielder':
            # Map keys and maintain order
            user_row = [vals[k] for k in OUTFIELD_KEY_MAP.keys()]
            user_df = pd.DataFrame([user_row], columns=OUTFIELD_FEATURES)
            
            user_scaled = scaler_out.transform(user_df)
            _, indices = knn_out.kneighbors(user_scaled)
            matched = outfield_data.iloc[indices[0]]

            players = []
            for _, row in matched.iterrows():
                m_pct, bdown = compute_match_pct(user_df, row, OUTFIELD_FEATURES, outfield_data)
                players.append({
                    'name': row['player_name'], 'team': row['team'], 'archetype': row['archetype_name'],
                    'match': m_pct, 'age': int(row['age']), 'minutes_played': int(row['minutes_played']),
                    'passing_accuracy': round(float(row['passing_accuracy(%)']), 1),
                    'dribbles': int(row['dribbles']), 'balls_recovered': int(row['balls_recovered']),
                    'breakdown': bdown
                })

        elif position == 'goalkeeper':
            user_row = [vals[k] for k in GK_KEY_MAP.keys()]
            user_df = pd.DataFrame([user_row], columns=GK_FEATURES)
            
            user_scaled = scaler_gk.transform(user_df)
            _, indices = knn_gk.kneighbors(user_scaled)
            matched = gk_data.iloc[indices[0]]

            players = []
            for _, row in matched.iterrows():
                m_pct, bdown = compute_match_pct(user_df, row, GK_FEATURES, gk_data)
                players.append({
                    'name': row['player_name'], 'team': row['team'], 'archetype': row['archetype_name'],
                    'match': m_pct, 'age': int(row['age']), 'minutes_played': int(row['minutes_played']),
                    'matches_appearance': int(row['matches_appearance']),
                    'passing_accuracy': round(float(row['passing_accuracy(%)']), 1),
                    'balls_recovered': int(row['balls_recovered']),
                    'breakdown': bdown
                })
        else:
            return jsonify({'error': f'Invalid position: {position}'}), 400

        return jsonify({
            'team': players[0]['team'],
            'team_reason': f"Your top match, {players[0]['name']}, plays for {players[0]['team']}.",
            'players': players
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/')
def index():
    return send_from_directory('.', 'fanfinder (6).html')
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)