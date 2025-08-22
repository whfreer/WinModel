import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib  # For saving the model

# ----------------------------
# Load and preprocess the data
# ----------------------------

columns_to_import = [
    'quarter', 'down', 'clock', 'def_score', 'off_score', 'distance',
    'off_timeouts_remaining', 'def_timeouts_remaining', 'yards_to_goal_line',
    'offensive_win_loss'
]

df = pd.read_csv('2024_fbs_season (1).csv', usecols=columns_to_import)
df = df[df['down'] > 0].dropna()

# Convert time to seconds remaining in game
def convert_to_seconds(time_str, quarter):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return (4 - quarter) * 15 * 60 + minutes * 60 + seconds
    except:
        return np.nan

df['time_remaining'] = df.apply(lambda row: convert_to_seconds(row['clock'], row['quarter']), axis=1)
df = df[df['time_remaining'] <= 300]

# Convert win/loss to binary
df['offensive_win_loss'] = df['offensive_win_loss'].map({'W': 1, 'L': 0})

# Compute score differential
df['score_diff'] = df['off_score'] - df['def_score']

# Drop rows with missing values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ----------------------------
# Feature Engineering
# ----------------------------

# Weight time_remaining heavily
df['time_remaining_x3'] = df['time_remaining'] * 3
df['time_remaining_x5'] = df['time_remaining'] * 5

# ----------------------------
# Define features and labels
# ----------------------------

features = [
    'time_remaining', 'time_remaining_x3', 'time_remaining_x5',
    'score_diff', 'down', 'distance',
    'yards_to_goal_line', 'off_timeouts_remaining', 'def_timeouts_remaining'
]

X = df[features]
y = df['offensive_win_loss']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Train/test split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Define and tune MLPClassifier
# ----------------------------

param_grid = {
    'hidden_layer_sizes': [
        (100,), (150,), (100, 50), (150, 100),
        (150, 100, 50), (200, 150, 100)
    ],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.005],
    'max_iter': [500]
}

mlp = MLPClassifier(random_state=42)

grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# ----------------------------
# Evaluate model
# ----------------------------

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# Save model and scaler
# ----------------------------

joblib.dump(best_model, 'win_probability_model_weighted_time.pkl')
joblib.dump(scaler, 'feature_scaler_weighted_time.pkl')

# ----------------------------
# Predict custom scenario
# ----------------------------

def predict_win_probability(scenario_dict):
    # Add time weighting
    scenario_dict['time_remaining_x3'] = scenario_dict['time_remaining'] * 3
    scenario_dict['time_remaining_x5'] = scenario_dict['time_remaining'] * 5

    X_input = pd.DataFrame([scenario_dict])[features]
    X_input_scaled = scaler.transform(X_input)
    win_prob = best_model.predict_proba(X_input_scaled)[0][1]
    return win_prob

# Example:
example = {
    'time_remaining': 120,        # 2 minutes
    'score_diff': -3,
    'down': 4,
    'distance': 5,
    'yards_to_goal_line': 35,
    'off_timeouts_remaining': 2,
    'def_timeouts_remaining': 3
}

prob = predict_win_probability(example)
print(f"Win probability: {prob:.2%}")
