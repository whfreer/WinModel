import pandas as pd
import joblib

# ----------------------------
# Load the trained model and scaler
# ----------------------------

best_model = joblib.load('win_probability_model_weighted_time.pkl')
scaler = joblib.load('feature_scaler_weighted_time.pkl')

# ----------------------------
# Define features and prediction function
# ----------------------------

features = [
    'time_remaining', 'time_remaining_x3', 'time_remaining_x5',
    'score_diff', 'down', 'distance',
    'yards_to_goal_line', 'off_timeouts_remaining', 'def_timeouts_remaining'
]

def predict_win_probability_5min(scenario_dict):
    # Add weighted time features
    scenario_dict['time_remaining_x3'] = scenario_dict['time_remaining'] * 3
    scenario_dict['time_remaining_x5'] = scenario_dict['time_remaining'] * 5

    # Create DataFrame and apply scaler
    X_input = pd.DataFrame([scenario_dict])[features]
    X_input_scaled = scaler.transform(X_input)

    # Predict win probability
    win_prob = best_model.predict_proba(X_input_scaled)[0][1]
    return win_prob






# ----------------------------
# Example usage
# ----------------------------

example = {
    'time_remaining': 300,         # 1:15 left
    'score_diff': 0,             # trailing by 4
    'down': 1,
    'distance': 10,
    'yards_to_goal_line': 10,
    'off_timeouts_remaining': 3,
    'def_timeouts_remaining': 3
}

prob = predict_win_probability_5min(example)
print(f"Win probability: {prob:.2%}")
