import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# Function to plot EPA with two different condition sets
def plot_epa(variable_name, var_range, conditions1, conditions2):
    epa_values1 = [calculate_epa(**{**conditions1, variable_name: v}) for v in var_range]
    epa_values2 = [calculate_epa(**{**conditions2, variable_name: v}) for v in var_range]

    plt.figure(figsize=(8, 5))
    plt.plot(var_range, epa_values1, label='Condition Set 1', marker='o')
    plt.plot(var_range, epa_values2, label='Condition Set 2', marker='s')

    plt.xlabel(variable_name)
    plt.ylabel("Expected Points (EPA)")
    plt.title(f"EPA vs {variable_name}")
    plt.legend()

    # Set axis limits - Change these values as needed
    plt.xlim(min(var_range), max(var_range))  # Adjust X-axis range
    plt.ylim(min(min(epa_values1), min(epa_values2)) - 0.5,
             max(max(epa_values1), max(epa_values2)) + 0.5)  # Adjust Y-axis range

    plt.grid()
    plt.show()


# Example variable to vary and its range - Change variable and range as needed
variable_to_vary = 'distance'  # Change to 'distance', 'down', etc.
variable_range = np.linspace(0, 100, 100)  # Adjust the range appropriately

# Define two condition sets - Modify as needed
conditions_set_1 = {'down': 1, 'distance': 10, 'yards_to_goal_line': 80, 'score_differential': -7, 'is_home_team': 0}
conditions_set_2 = {'down': 4, 'distance': 2, 'yards_to_goal_line': 80, 'score_differential': -7, 'is_home_team': 0}

# Generate the plot
plot_epa(variable_to_vary, variable_range, conditions_set_1, conditions_set_2)
