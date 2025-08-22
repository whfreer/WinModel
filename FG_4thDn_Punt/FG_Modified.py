import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
columns_to_import = ['quarter', 'down', 'clock', 'def_score', 'off_score', 'distance',
                     'off_timeouts_remaining', 'def_timeouts_remaining', 'yards_to_goal_line',
                     'offensive_win_loss', 'special_teams_type', 'kick_result', 'touchdown']

df = pd.read_csv('2024_fbs_season (1).csv', usecols=columns_to_import)

# Filter for field goals
df = df[df['special_teams_type'] == 'FIELD GOAL']

# Classify outcomes
def classify_kick(row):
    result = str(row['kick_result'])
    td = pd.notna(row['touchdown'])  # Check if touchdown is not NaN
    if result.startswith('MADE'):
        return 'MADE'
    elif result.startswith('BLOCKED') and td:
        return 'BLOCKED_TD'
    elif result.startswith('BLOCKED'):
        return 'BLOCKED'
    else:
        return 'MISSED'

df['kick_outcome'] = df.apply(classify_kick, axis=1)
df['FG_distance'] = df['yards_to_goal_line']

# Group by distance and outcome
grouped = df.groupby(['FG_distance', 'kick_outcome']).size().unstack(fill_value=0)

# Ensure all outcome columns are present
for col in ['MADE', 'MISSED', 'BLOCKED', 'BLOCKED_TD']:
    if col not in grouped.columns:
        grouped[col] = 0

# Calculate total and percentages
grouped['total'] = grouped[['MADE', 'MISSED', 'BLOCKED', 'BLOCKED_TD']].sum(axis=1)
grouped['MADE_pct'] = grouped['MADE'] / grouped['total'] * 100
grouped['MISSED_pct'] = grouped['MISSED'] / grouped['total'] * 100
grouped['BLOCKED_pct'] = grouped['BLOCKED'] / grouped['total'] * 100
grouped['BLOCKED_TD_pct'] = grouped['BLOCKED_TD'] / grouped['total'] * 100

# ===== Plot 1: Bar chart of counts =====
grouped[['MADE', 'MISSED', 'BLOCKED', 'BLOCKED_TD']].plot(
    kind='bar', stacked=True, figsize=(14, 6), colormap='viridis'
)
plt.title('Field Goal Attempt Outcomes by Distance')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Number of Attempts')
plt.legend(title='Kick Outcome')
plt.tight_layout()
plt.show()

# ===== Plot 2: Line chart of raw percentages =====
plt.figure(figsize=(14, 7))
plt.plot(grouped.index, grouped['MADE_pct'], label='MADE %', color='green', marker='o')
plt.plot(grouped.index, grouped['MISSED_pct'], label='MISSED %', color='red', marker='o')
plt.plot(grouped.index, grouped['BLOCKED_pct'], label='BLOCKED %', color='gray', marker='o')
plt.plot(grouped.index, grouped['BLOCKED_TD_pct'], label='BLOCKED FOR TD %', color='black', linestyle='--', marker='x')
plt.title('Field Goal Outcome Percentages by Distance (Unsmoothed)')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Percentage of Attempts')
plt.legend(title='Kick Outcome')
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Plot 3: Line chart of smoothed percentages =====
grouped['MADE_pct_smoothed'] = grouped['MADE_pct'].rolling(window=5, center=True).mean()
grouped['MISSED_pct_smoothed'] = grouped['MISSED_pct'].rolling(window=5, center=True).mean()
grouped['BLOCKED_pct_smoothed'] = grouped['BLOCKED_pct'].rolling(window=5, center=True).mean()
grouped['BLOCKED_TD_pct_smoothed'] = grouped['BLOCKED_TD_pct'].rolling(window=5, center=True).mean()

plt.figure(figsize=(14, 7))
plt.plot(grouped.index, grouped['MADE_pct_smoothed'], label='MADE % (Smoothed)', color='green', marker='o')
plt.plot(grouped.index, grouped['MISSED_pct_smoothed'], label='MISSED % (Smoothed)', color='red', marker='o')
plt.plot(grouped.index, grouped['BLOCKED_pct_smoothed'], label='BLOCKED % (Smoothed)', color='gray', marker='o')
plt.plot(grouped.index, grouped['BLOCKED_TD_pct_smoothed'], label='BLOCKED FOR TD % (Smoothed)', color='black', linestyle='--', marker='x')
plt.title('Smoothed Field Goal Outcome Percentages by Distance')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Smoothed Percentage of Attempts')
plt.legend(title='Kick Outcome')
plt.grid(True)
plt.tight_layout()
plt.show()
