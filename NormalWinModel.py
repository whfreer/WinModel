import os
import pandas as pd
import numpy as np
from scipy.stats import norm

# Function to calculate Expected Points (EPA) based on down, distance, field position, score differential, and home team status
def calculate_epa(down, distance, yards_to_goal_line, score_differential, is_home_team):
    epa = (7.1498 + (-0.3230 * down) + (-0.2493 * distance) + (-0.0314 * yards_to_goal_line) +
           (0.0005 * score_differential) + (0.2231 * is_home_team) + (-0.1326 * (down ** 2)) +
           (0.0209 * down * distance) + (-0.0046 * down * yards_to_goal_line) +
           (-0.0001 * down * score_differential) + (0.0263 * down * is_home_team) +
           (0.0033 * (distance ** 2)) + (0.0008 * distance * yards_to_goal_line) +
           (-0.0000 * distance * score_differential) + (-0.0007 * distance * is_home_team) +
           (-0.0004 * (yards_to_goal_line ** 2)) + (0.0000 * yards_to_goal_line * score_differential) +
           (0.0112 * yards_to_goal_line * is_home_team) + (0.0000 * (score_differential ** 2)) +
           (-0.0002 * score_differential * is_home_team) + (0.2231 * (is_home_team ** 2)))
    return epa

# Function to calculate win probability from your team's perspective
def calculate_win_probability(your_team_margin, epa, home_vegas_line, minutes_remaining, game_stdev=15.422):
    # Adjust the margin by adding EP (your team's perspective)
    current_expected_margin = your_team_margin + epa

    # Calculate fraction of time remaining
    fraction_remaining = minutes_remaining / 60
    adjusted_stdev = game_stdev / np.sqrt(60/minutes_remaining)
    adjusted_mean = -home_vegas_line * fraction_remaining

    # Calculate win probability using the normal distribution
    win_prob = (1 - norm.cdf(current_expected_margin + 0.5, adjusted_mean, adjusted_stdev)) + \
               (0.5 * (norm.cdf(current_expected_margin + 0.5, adjusted_mean, adjusted_stdev) - \
                       norm.cdf(current_expected_margin - 0.5, adjusted_mean, adjusted_stdev)))
    return win_prob

# Example inputs
down = 1
distance = 10
yards_to_goal_line = 10
score_differential = -2
is_home_team = 1  # Assuming this is the home team
home_vegas_line = -15
your_team_margin = -2
minutes_remaining = 2

# Calculate EPA with the updated formula
epa = calculate_epa(down, distance, yards_to_goal_line, score_differential, is_home_team)

# Calculate win probability using the adjusted margin for your team's perspective (EPA adjusted)
win_probability = calculate_win_probability(your_team_margin, epa, home_vegas_line, minutes_remaining)

# Print results
print(f"EPA: {epa:.4f}")
print(f"Adjusted Expected Margin (from your team's perspective): {your_team_margin + epa:.4f}")
print(f"Win Probability (from your team's perspective): {win_probability:.4f}")
