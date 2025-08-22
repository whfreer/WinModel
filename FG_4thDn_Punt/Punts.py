import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
columns_to_import = [
    'quarter', 'down', 'clock', 'def_score', 'off_score', 'distance', 'drive_end_event',
    'off_timeouts_remaining', 'def_timeouts_remaining', 'yards_to_goal_line', 'fumble_recovery',
    'offensive_win_loss', 'special_teams_type', 'kick_result', 'touchdown', 'kick_yards',
    'return_yards', 'first_down_gained', 'penalty_yards'
]

df = pd.read_csv('2024_fbs_season (1).csv', usecols=columns_to_import)

# Filter for punts
punts = df[(df['down'] == 4) & (df['special_teams_type'] == 'PUNT')].copy()

# Define punt outcome categories
def classify_punt(row):
    result = str(row['kick_result']).strip().upper()
    if result == 'TOUCHBACK':
        return 'Touchback'
    elif result == 'RECOVERED BY KICKING TEAM':
        return 'Fumble'
    elif pd.notna(row['touchdown']):
        return 'Touchdown'
    elif result == 'RETURNED':
        return 'Return'
    elif pd.notna(row['penalty_yards']) and row.get('first_down_gained') == 1:
        return 'Penalty 1st Down'
    elif result == 'BLOCKED':
        return 'Blocked'
    elif result in ['OUT OF BOUNDS', 'DOWNED', 'FAIR CATCH']:
        return 'Normal Punt'
    else:
        return

punts['punt_outcome'] = punts.apply(classify_punt, axis=1)

# Bin yards to goal line in 5-yard bins
punts['yard_bin'] = (punts['yards_to_goal_line'] // 5 * 5).astype(int)

# Count outcomes per yard bin
outcome_counts = punts.groupby(['yard_bin', 'punt_outcome']).size().unstack(fill_value=0)

# Normalize to get percentages
outcome_percentages = outcome_counts.div(outcome_counts.sum(axis=1), axis=0) * 100

# Plot stacked bar chart of outcome percentages
outcome_percentages.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab10')
plt.ylabel('Percentage of Outcomes')
plt.xlabel('Yards to Goal Line (binned)')
plt.title('Punt Outcome Percentages by Field Position')
plt.legend(title='Punt Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()

# Plot line chart of outcome percentages
plt.figure(figsize=(14, 8))
for outcome in outcome_percentages.columns:
    plt.plot(outcome_percentages.index, outcome_percentages[outcome], marker='o', label=outcome)

plt.ylabel('Percentage of Outcomes')
plt.xlabel('Yards to Goal Line (binned)')
plt.title('Line Chart of Punt Outcome Percentages by Field Position')
plt.legend(title='Punt Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Average punt distance by yard bin
kick_dist_by_yard = punts.groupby('yard_bin')['kick_yards'].mean()

plt.figure(figsize=(12, 6))
plt.plot(kick_dist_by_yard.index, kick_dist_by_yard.values, marker='o', linestyle='-', color='darkblue')
plt.title('Average Punt Distance by Field Position')
plt.xlabel('Yards to Goal Line (binned)')
plt.ylabel('Average Kick Distance (yards)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Average return yards per return by yard bin
return_punts = punts[punts['punt_outcome'] == 'Return'].copy()
avg_return_yards = return_punts.groupby('yard_bin')['return_yards'].mean()

plt.figure(figsize=(12, 6))
plt.plot(avg_return_yards.index, avg_return_yards.values, marker='o', linestyle='-', color='green')
plt.title('Average Return Yards by Field Position')
plt.xlabel('Yards to Goal Line (binned)')
plt.ylabel('Average Return Yards')
plt.grid(True)
plt.tight_layout()
plt.show()

# Save data to CSV
combined_data = pd.DataFrame({
    'Average Kick Distance': kick_dist_by_yard,
    'Average Return Yards': avg_return_yards
}).reset_index()

combined_data.to_csv('punt_analysis_data.csv', index=False)



# Calculate percentage of each punt outcome per yard_bin (5-yard group)
outcome_counts_by_bin = punts.groupby(['yard_bin', 'punt_outcome']).size().unstack(fill_value=0)
outcome_percentages_by_bin = outcome_counts_by_bin.div(outcome_counts_by_bin.sum(axis=1), axis=0) * 100

# Reset index and save to CSV
outcome_percentages_by_bin.reset_index().to_csv('punt_outcome_percentages_by_yard_bin.csv', index=False)
