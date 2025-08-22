import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
columns_to_import = ['quarter', 'down', 'distance', 'yards_to_goal_line', 'expected_points', 'score_differential',
                     'off_timeouts_remaining', 'def_timeouts_remaining', 'home_team', 'offense']
df = pd.read_csv('2024_fbs_season (1).csv', usecols=columns_to_import)

df = df[df['down'] > 0].dropna()
df['is_home_team'] = df.apply(lambda row: 1 if row['offense'] == row['home_team'] else 0, axis=1)

# Define features and target
base_features = ['down', 'distance', 'yards_to_goal_line', 'score_differential', 'is_home_team']
target = 'expected_points'

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(df[base_features])
feature_names = poly.get_feature_names_out(base_features)

y = df[target]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print formula
print("Expected Points (EPA) Formula:")
print(f"EPA = {intercept:.4f} ", end="")
for feature, coef in zip(feature_names, coefficients):
    if coef != 0:
        print(f"+ ({coef:.4f} * {feature})", end=" ")
print()

# Evaluate model performance
score = model.score(X_test, y_test)
print(f"\nR^2 (R-squared) on test data: {score:.4f}")
