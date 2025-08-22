import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
columns_to_import = ['quarter', 'down', 'clock', 'def_score', 'off_score', 'distance',
                     'off_timeouts_remaining', 'def_timeouts_remaining', 'yards_to_goal_line',
                     'offensive_win_loss', 'first_down_gained', 'special_teams_type',
                     'drive_end', 'touchdown']

df = pd.read_csv('2024_fbs_season (1).csv', usecols=columns_to_import)

# Filter: 4th down, no special teams, distance ≤ 15
df_4th = df[(df['down'] == 4) &
            (df['special_teams_type'].isna()) &
            (df['distance'] <= 20)]

# Drop missing values
df_4th = df_4th.dropna(subset=['distance', 'yards_to_goal_line'])

# Outcome flags
df_4th['conversion'] = df_4th['first_down_gained'].fillna(0).astype(int)
df_4th['turnover'] = df_4th['drive_end'].isin(['INTERCEPTION', 'FUMBLE']).astype(int)
df_4th['turnover_td'] = ((df_4th['turnover'] == 1) & df_4th['touchdown'].notna()).astype(int)
df_4th['turnover_on_downs'] = ((df_4th['conversion'] == 0) & (df_4th['turnover'] == 0)).astype(int)
df_4th['unsuccessful'] = (df_4th['conversion'] == 0).astype(int)

# Bin distance to nearest yard
df_4th['distance_bin'] = df_4th['distance'].round(0).astype(int)

# Group by distance
summary = df_4th.groupby('distance_bin').agg(
    attempts=('conversion', 'count'),
    conversions=('conversion', 'sum'),
    unsuccessful=('unsuccessful', 'sum'),
    turnovers=('turnover', 'sum'),
    turnover_tds=('turnover_td', 'sum'),
    turnover_on_downs=('turnover_on_downs', 'sum')
).reset_index()

# Percentages
summary['conversion_pct'] = summary['conversions'] / summary['attempts'] * 100
summary['unsuccessful_pct'] = summary['unsuccessful'] / summary['attempts'] * 100
summary['turnover_pct'] = summary['turnovers'] / summary['attempts'] * 100
summary['turnover_td_pct'] = summary['turnover_tds'] / summary['attempts'] * 100
summary['turnover_on_downs_pct'] = summary['turnover_on_downs'] / summary['attempts'] * 100

# --- Line Chart ---
plt.figure(figsize=(14, 7))
plt.plot(summary['distance_bin'], summary['conversion_pct'], marker='o', label='Conversion %', color='green')
plt.plot(summary['distance_bin'], summary['turnover_pct'], marker='x', label='Turnover %', color='red')
plt.plot(summary['distance_bin'], summary['turnover_td_pct'], marker='D', label='Turnover TD %', color='darkred')
plt.plot(summary['distance_bin'], summary['turnover_on_downs_pct'], marker='s', label='Turnover on Downs %', color='orange')
plt.plot(summary['distance_bin'], summary['unsuccessful_pct'], marker='^', label='Unsuccessful %', color='gray')

plt.title('4th Down Outcomes by Distance to Go')
plt.xlabel('Distance to Go (yards)')
plt.ylabel('Percentage of Plays')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Stacked Bar Chart (Percentages) ---
stacked_df = summary[['distance_bin', 'conversion_pct', 'turnover_on_downs_pct',
                      'turnover_pct', 'turnover_td_pct']].copy()

stacked_df = stacked_df.rename(columns={
    'conversion_pct': 'Conversions',
    'turnover_on_downs_pct': 'Turnover on Downs',
    'turnover_pct': 'Turnovers',
    'turnover_td_pct': 'Turnover TDs'
})

stacked_df = stacked_df.set_index('distance_bin')

stacked_df.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Stacked 4th Down Outcomes by Distance to Go')
plt.xlabel('Distance to Go (yards)')
plt.ylabel('Percentage of Plays')
plt.legend(title='Outcome')
plt.tight_layout()
plt.show()

# --- Export to Excel ---
summary.to_excel('4th_down_outcome_breakdown.xlsx', index=False)




