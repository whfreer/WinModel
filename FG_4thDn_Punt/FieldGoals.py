import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
columns_to_import = ['quarter', 'down', 'clock', 'def_score', 'off_score', 'distance',
                     'off_timeouts_remaining', 'def_timeouts_remaining', 'yards_to_goal_line',
                     'offensive_win_loss','special_teams_type','kick_result','touchdown']
df = pd.read_csv('2024_fbs_season (1).csv', usecols=columns_to_import)

# Filter for field goals only
df = df[df['special_teams_type'] == 'FIELD GOAL']

# Categorize kick outcome: MADE, MISSED, or BLOCKED
def classify_kick(x):
    x = str(x)
    if x.startswith('MADE'):
        return 'MADE'
    elif x.startswith('BLOCKED'):
        return 'BLOCKED'
    else:
        return 'MISSED'

df['kick_outcome'] = df['kick_result'].apply(classify_kick)

# Assign field goal distance
df['FG_distance'] = df['yards_to_goal_line']

# Group by distance and outcome
grouped = df.groupby(['FG_distance', 'kick_outcome']).size().unstack(fill_value=0)

# Optional: sort by distance
grouped = grouped.sort_index()

# Compute made percentage (excluding BLOCKED)
grouped['made_pct'] = grouped['MADE'] / (grouped['MADE'] + grouped['MISSED']) * 100

# Plot stacked bar chart for MADE, MISSED, BLOCKED
plt.figure(figsize=(12, 6))
grouped[['MADE', 'MISSED', 'BLOCKED']].plot(kind='bar', stacked=True, figsize=(14, 6), colormap='viridis')
plt.title('Field Goal Attempts by Distance (Including BLOCKED)')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Number of Attempts')
plt.tight_layout()
plt.show()

# Plot made percentage (unsmoothed)
plt.figure(figsize=(12, 6))
plt.plot(grouped.index, grouped['made_pct'], marker='o', linestyle='-', color='green')
plt.title('Field Goal Made Percentage by Distance')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Made Percentage')
plt.grid(True)
plt.tight_layout()
plt.show()

# Smoothed made percentage
grouped['made_pct_smoothed'] = grouped['made_pct'].rolling(window=5, center=True).mean()

plt.figure(figsize=(12, 6))
plt.plot(grouped.index, grouped['made_pct_smoothed'], marker='o', linestyle='-', color='blue')
plt.title('Smoothed Field Goal Made Percentage by Distance')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Made Percentage (Smoothed)')
plt.grid(True)
plt.tight_layout()
plt.show()


# Ensure all outcome columns are present
for col in ['MADE', 'MISSED', 'BLOCKED']:
    if col not in grouped.columns:
        grouped[col] = 0

# Calculate total attempts per distance
grouped['total'] = grouped['MADE'] + grouped['MISSED'] + grouped['BLOCKED']

# Calculate percentage columns
grouped['MADE_pct'] = grouped['MADE'] / grouped['total'] * 100
grouped['MISSED_pct'] = grouped['MISSED'] / grouped['total'] * 100
grouped['BLOCKED_pct'] = grouped['BLOCKED'] / grouped['total'] * 100

# Plot line chart with percentages
plt.figure(figsize=(14, 7))
plt.plot(grouped.index, grouped['MADE_pct'], label='MADE %', color='green', marker='o')
plt.plot(grouped.index, grouped['MISSED_pct'], label='MISSED %', color='red', marker='o')
plt.plot(grouped.index, grouped['BLOCKED_pct'], label='BLOCKED %', color='gray', marker='o')

plt.title('Field Goal Outcome Percentages by Distance')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Percentage of Attempts')
plt.legend(title='Kick Outcome')
plt.grid(True)
plt.tight_layout()
plt.show()


# Ensure all outcome columns are present
for col in ['MADE', 'MISSED', 'BLOCKED']:
    if col not in grouped.columns:
        grouped[col] = 0

# Calculate total attempts per distance
grouped['total'] = grouped['MADE'] + grouped['MISSED'] + grouped['BLOCKED']

# Compute raw percentages
grouped['MADE_pct'] = grouped['MADE'] / grouped['total'] * 100
grouped['MISSED_pct'] = grouped['MISSED'] / grouped['total'] * 100
grouped['BLOCKED_pct'] = grouped['BLOCKED'] / grouped['total'] * 100

# Smooth percentages using rolling average (window=5)
grouped['MADE_pct_smoothed'] = grouped['MADE_pct'].rolling(window=5, center=True).mean()
grouped['MISSED_pct_smoothed'] = grouped['MISSED_pct'].rolling(window=5, center=True).mean()
grouped['BLOCKED_pct_smoothed'] = grouped['BLOCKED_pct'].rolling(window=5, center=True).mean()

# Plot smoothed percentage lines
plt.figure(figsize=(14, 7))
plt.plot(grouped.index, grouped['MADE_pct_smoothed'], label='MADE % (Smoothed)', color='green', marker='o')
plt.plot(grouped.index, grouped['MISSED_pct_smoothed'], label='MISSED % (Smoothed)', color='red', marker='o')
plt.plot(grouped.index, grouped['BLOCKED_pct_smoothed'], label='BLOCKED % (Smoothed)', color='gray', marker='o')

plt.title('Smoothed Field Goal Outcome Percentages by Distance')
plt.xlabel('Yards to Goal Line')
plt.ylabel('Smoothed Percentage of Attempts')
plt.legend(title='Kick Outcome')
plt.grid(True)
plt.tight_layout()
plt.show()




# Export smoothed data to CSV
smoothed_df = grouped[['made_pct_smoothed']].dropna().reset_index()
smoothed_df.to_csv('smoothed_field_goal_pct.csv', index=False)
