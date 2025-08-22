from WinModelExecuteNormal import *
from WinModelExecute5min import *
import numpy as np



def go_for_it_decision(time_remaining,score_diff,down,distance,yards_to_goal_line, kicker_range, gft):
    clutch = False
    punt_expected = 0
    fg_expected = 0
    go_for_it_expected = 0
    if time_remaining <= 300:
        clutch = True
    if clutch:
        if yards_to_goal_line >= 35:
            punt_expected = punt_outcomes(time_remaining, score_diff, down, distance, yards_to_goal_line, clutch)
        go_for_it_expected = go_for_it_outcomes(time_remaining, score_diff, down, distance, yards_to_goal_line, clutch)
        if kicker_range >= yards_to_goal_line + 17:
            fg_expected = FG_outcomes(time_remaining,score_diff,down,distance,yards_to_goal_line, clutch)
    else:
        if yards_to_goal_line >= 35:
            punt_expected = punt_outcomes(time_remaining, score_diff, down, distance, yards_to_goal_line, clutch)
        go_for_it_expected = go_for_it_outcomes(time_remaining, score_diff, down, distance, yards_to_goal_line, clutch)
        if kicker_range >= yards_to_goal_line + 17:
            fg_expected = FG_outcomes(time_remaining,score_diff,down,distance,yards_to_goal_line, clutch)
    options = {
        'Punt': punt_expected,
        'Field Goal': fg_expected,
        'Go for It': go_for_it_expected
    }
    # Define threshold percentage (e.g., 10% better than others)
    go_for_it_threshold = gft

    # Find the best non-go-for-it option
    other_options = {k: v for k, v in options.items() if k != 'Go for It'}
    best_other_value = max(other_options.values())

    # Apply threshold logic
    if options['Go for It'] > (1 + go_for_it_threshold) * best_other_value:
        recommended = 'Go for It'
    else:
        # If Go for It isn't sufficiently better, recommend best of the others
        recommended = max(other_options, key=other_options.get)

    # Print each expected value
    print(f"Punt Expected Value: {punt_expected:.3f}")
    print(f"Field Goal Expected Value: {fg_expected:.3f}")
    print(f"Go For It Expected Value: {go_for_it_expected:.3f}")
    print(f"Recommended Decision: {recommended}")
    return options, recommended




def punt_outcomes(time_remaining,score_diff,down,distance,x, clutch):
    return_distance = 0.06*x + 3.3892
    punt_distance = (0.00012729 * (x**3)) - (0.0315 * (x**2)) + (2.5448*x) - 23.139
    return_percent = (-4.16287e-8 * x**6 + 1.58596e-5 * x**5 - 0.00241825 * x**4 + 0.187957 * x**3 - 7.81974 * x**2 + 165.551 * x - 1400.04)/100
    touchback_percent = (7.81974e-9 * x**6 - 2.80791e-6 * x**5 + 3.97550e-04 * x**4 - 0.0275 * x**3 + 0.9571 * x**2 - 15.958 * x + 130.63)/100
    fumble = 0.01
    if x < 70:
        blocked = 0.0075
        td = 0.005
    else:
        blocked = 0.015
        td = 0.01
    downed_percent = 1 - return_percent - touchback_percent - fumble - blocked - td
    return_wp = (1 - wpc(time_remaining-8,-score_diff,1,10,100 - (x-punt_distance+return_distance), clutch)) * return_percent
    touchback_wp = (1 - wpc(time_remaining-8,-score_diff,1,10,80, clutch)) * touchback_percent
    fumble_wp = fumble * wpc(time_remaining-8,score_diff,1,10,(x-punt_distance), clutch)
    blocked_wp = blocked *(1- wpc(time_remaining-8,-score_diff,1,10,100-x-10, clutch))
    td_wp = td * wpc(time_remaining-15,score_diff-7,1,10,80, clutch)
    downed_wp = downed_percent * (1-wpc(time_remaining-8,-score_diff,1,10,(100 - (x-punt_distance)), clutch))
    punt_expected_win_percent = return_wp + touchback_wp + fumble_wp + blocked_wp + td_wp + downed_wp
    return punt_expected_win_percent


def FG_outcomes(time_remaining,score_diff,down,distance,x, clutch):
    fg_percent = (-3e-5 * x ** 4 + 0.0025 * x ** 3 - 0.0749 * x ** 2- 0.357 * x + 97.349)/100
    miss_percent = 1 - fg_percent
    make_wp = (1-wpc(time_remaining-6,-(score_diff+3),1,10,80, clutch))*fg_percent
    miss_wp = (1-wpc(time_remaining-6,-score_diff,1,10,100-x, clutch))*miss_percent
    return make_wp + miss_wp
def go_for_it_outcomes(time_remaining,score_diff,down,distance,x, clutch):
    convert_percent = 0.75 * np.exp(-0.128 * distance)
    miss_percent = 1 - convert_percent
    make_wp = (wpc(time_remaining - 6, score_diff, 1, 10, x-distance, clutch)) * convert_percent
    miss_wp = (1 - wpc(time_remaining - 6, -score_diff, 1, 10, 100 - x, clutch)) * miss_percent
    return make_wp + miss_wp
def wpc(time_remaining,score_diff,down,distance,yards_to_goal_line, clutch):
    scene = {
        'time_remaining': time_remaining,  # 1:15 left
        'score_diff': score_diff,  # trailing by 4
        'down': down,
        'distance': distance,
        'yards_to_goal_line': yards_to_goal_line,
        'off_timeouts_remaining': 3,
        'def_timeouts_remaining': 3
    }
    if clutch:
        wp = predict_win_probability_5min(scene)
    else:
        wp = predict_win_probability(scene)
    return wp



go_for_it_decision(60,-2,4,10,10,55,0)