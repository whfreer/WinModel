from PUNT_FG_GO import go_for_it_decision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# Parameters
time = 1200
score_diff = 3
down = 4
kicker_range = 55
thresholds = [0.025, 0.10, 0.20]

# Ranges
yards_to_goals = np.arange(1, 100)  # Yards to goal line
yards_to_go = np.arange(1, 21)      # Yards to go (1st down distance)

# Grid size
max_shift = yards_to_go[-1]
full_length = len(yards_to_goals) + max_shift
go_for_it_score_grid = np.full((len(yards_to_go), full_length), np.nan)

# Fill the grid by checking how many thresholds recommend "Go for It"
for i, distance in enumerate(yards_to_go):
    for j, ytg in enumerate(yards_to_goals):
        if ytg >= distance:
            shifted_col = j - distance + max_shift
            yards_between = ytg - distance
            if yards_between <= 90:
                go_count = 0
                fallback_counts = {'Punt': 0, 'Field Goal': 0}
                for t in thresholds:
                    _, rec = go_for_it_decision(
                        time, score_diff, down, distance, ytg, kicker_range, t
                    )
                    if rec == 'Go for It':
                        go_count += 1
                    else:
                        fallback_counts[rec] += 1

                if go_count == 3:
                    go_for_it_score_grid[i, shifted_col] = 3  # Dark green
                elif go_count == 2:
                    go_for_it_score_grid[i, shifted_col] = 2  # Medium green
                elif go_count == 1:
                    go_for_it_score_grid[i, shifted_col] = 1  # Light green
                else:
                    # Majority fallback (Punt = 0, FG = 0.5)
                    if fallback_counts['Field Goal'] > fallback_counts['Punt']:
                        go_for_it_score_grid[i, shifted_col] = 0.5  # Yellow
                    else:
                        go_for_it_score_grid[i, shifted_col] = 0  # Red

# Final x-axis range
x_values = np.arange(-max_shift, len(yards_to_goals))
valid_mask = (x_values >= 0) & (x_values <= 89)
x_values = x_values[valid_mask]
go_for_it_score_grid = go_for_it_score_grid[:, valid_mask]

# Split left and right of 50
left_mask = x_values <= 50
right_mask = x_values > 50
x_left, x_right = x_values[left_mask], x_values[right_mask]
left_half = go_for_it_score_grid[:, left_mask]
right_half = go_for_it_score_grid[:, right_mask]

# Custom green shades for go-for-it certainty
custom_cmap = ListedColormap(['red', 'yellow', 'lightgreen', 'mediumseagreen', 'forestgreen'])
norm = BoundaryNorm([-0.1, 0.25, 0.75, 1.75, 2.75, 3.1], custom_cmap.N)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(16, 12), sharey=True)

yticks = np.arange(len(yards_to_go))
yticklabels = yards_to_go

# Plot each half
for ax, data, x_vals, title in zip(
    axs,
    [left_half, right_half],
    [x_left, x_right],
    ['Recommended 4th Down Decision (Opponent Territory)', 'Recommended 4th Down Decision (Own Territory)']
):
    sns.heatmap(
        data,
        ax=ax,
        cmap=custom_cmap,
        norm=norm,
        cbar=False,
        linewidths=1,
        linecolor='black'
    )
    ax.set_xticks(np.arange(0, data.shape[1], 5) + 0.5)
    ax.set_xticklabels(x_vals[::5], fontsize=10)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Yards Between First Down Marker and Goal Line', fontsize=12)
    ax.set_ylabel('Yards to Go', fontsize=12)

# Colorbar to the side
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
    cax=cbar_ax,
    ticks=[0, 0.5, 1, 2, 3]
)
cbar.ax.set_yticklabels(['Punt', 'FG', 'CoinFlip Go', 'Likely Go', 'Absolute Go'])
cbar.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
