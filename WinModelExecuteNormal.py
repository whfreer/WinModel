import joblib
import pandas as pd

# ----------------------------
# Load model and scaler once
# ----------------------------
model = joblib.load('win_probability_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# ----------------------------
# Define features used in training
# ----------------------------
features = [
    'time_remaining', 'score_diff', 'down', 'distance',
    'yards_to_goal_line', 'off_timeouts_remaining', 'def_timeouts_remaining'
]

# ----------------------------
# Prediction function
# ----------------------------
def predict_win_probability(scenario):
    """
    Predicts win probability based on the input scenario.

    Parameters:
        scenario (dict): Dictionary containing values for:
            - time_remaining (int): Seconds left in the game
            - score_diff (int): Offense score - defense score
            - down (int): Current down (1–4)
            - distance (float): Yards to first down
            - yards_to_goal_line (float): Distance to end zone
            - off_timeouts_remaining (int): Offense timeouts remaining
            - def_timeouts_remaining (int): Defense timeouts remaining

    Returns:
        float: Predicted win probability (between 0 and 1)
    """
    X_input = pd.DataFrame([scenario])[features]
    X_scaled = scaler.transform(X_input)
    win_prob = model.predict_proba(X_scaled)[0][1]
    return win_prob


scenario = {
    'time_remaining': 1000,
    'score_diff': 4,
    'down': 4,
    'distance': 2,
    'yards_to_goal_line': 12,
    'off_timeouts_remaining': 1,
    'def_timeouts_remaining': 2
}

win_prob = predict_win_probability(scenario)
print(f"Win Probability: {win_prob:.2%}")


