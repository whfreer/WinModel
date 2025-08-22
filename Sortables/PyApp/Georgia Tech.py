import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_team_4th_down_conversions(team_abbr, csv_path='2024_fbs_season (1).csv', window_size=5):
    # Load relevant columns
    columns_to_import = ['quarter', 'down', 'clock', 'def_score', 'off_score', 'distance',
                         'off_timeouts_remaining', 'def_timeouts_remaining', 'yards_to_goal_line',
                         'offensive_win_loss', 'first_down_gained', 'home_team', 'away_team', 'special_teams_type']

    df = pd.read_csv(csv_path, usecols=columns_to_import)

    # Filter for selected team
    df = df[(df['home_team'] == team_abbr) | (df['away_team'] == team_abbr)]

    # 4th down non-special teams only
    df_4th = df[(df['down'] == 4) & (df['special_teams_type'].isna())]
    df_4th = df_4th.dropna(subset=['distance'])
    df_4th['distance'] = df_4th['distance'].round(1)
    df_4th['conversion'] = df_4th['first_down_gained'].fillna(0).astype(int)

    # Group and compute stats
    grouped = df_4th.groupby('distance').agg(
        conversions=('conversion', 'sum'),
        attempts=('conversion', 'count')
    )
    grouped['conversion_pct'] = grouped['conversions'] / grouped['attempts']
    grouped['smoothed_pct'] = grouped['conversion_pct'].rolling(window=window_size, center=True).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(grouped.index, grouped['conversion_pct'], marker='o', linestyle=':', color='gray', label='Raw %')
    plt.plot(grouped.index, grouped['smoothed_pct'], color='blue', linewidth=2,
             label=f'Smoothed % (window={window_size})')
    plt.plot(grouped.index, grouped['attempts'] / grouped['attempts'].max(), linestyle='--', color='lightgreen',
             label='Relative Attempt Count')

    plt.title(f'{team_abbr} 4th Down Conversion % vs Distance')
    plt.xlabel('Distance to Go (yards)')
    plt.ylabel('Conversion Percentage')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save CSV
    out_file = f'{team_abbr}_4th_down_conversion_vs_distance.csv'
    grouped.reset_index()[['distance', 'smoothed_pct', 'conversion_pct', 'attempts']].to_csv(out_file, index=False)
    print(f'Saved output to {out_file}')

plot_team_4th_down_conversions('FLUN')