# coding: utf-8
import os
import datetime
import numpy as np
import pandas as pd

# base_folder = os.path.abspath(".")
# os.chdir(base_folder)
# os.chdir(os.getcwd() + "/new_data")

# ### Loading all files
# I have 5 files that are loaded.
# 1. **home_csv -->** Allows me to indicate home and away!
# 2. **team_csv -->** Team statistics (Team level)
# 3. **more_stats_csv -->** Advanced Team Statistics (from 1996, currently not in use, Team level)
# 4. **game_date_csv -->** Game dates and potential attendance (Team level)
# 5. **win_loss_csv**
# ## Main Script


def load_all_files(year):
    home_csv = pd.read_csv("more_home_away_{}.csv".format(year))
    teams_csv = pd.read_csv("team_{}.csv".format(year), usecols=['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS'])

    # This has issues... Need to think if dropping these numbers are better
    more_stats_csv = pd.read_csv("more_team_stats_{}.csv".format(year))
    game_date_csv = pd.read_csv("game_date_{}.csv".format(year))

    # This has issues too... Need to re-calculate them myself...
    win_loss_csv = pd.read_csv("more_wins_losses_{}.csv".format(year))
    return home_csv, teams_csv, more_stats_csv, game_date_csv, win_loss_csv


# Create Variable:** HOME/AWAY variable for each team
def create_home_away_var(table):
    """Create home or away variable. Provides """
    home = table[["GAME_ID", "HOME_TEAM_ID", "SEASON", "LIVE_PERIOD"]].rename(columns={"HOME_TEAM_ID": "TEAM_ID"})
    home["Home"] = "Home"

    away = table[["GAME_ID", "VISITOR_TEAM_ID", "SEASON", "LIVE_PERIOD"]].rename(columns={"VISITOR_TEAM_ID": "TEAM_ID"})
    away["Home"] = "Away"
    return home.append(away)


# Create Variable:** Convert GAME_DATE to datetime
def create_date_variable(main_table):
    game_date = main_table.copy()
    game_date["GAME_DATE"] = [i.split(", ", 1)[1] for i in game_date.GAME_DATE]
    game_date = game_date[["GAME_DATE", "GAME_ID"]]
    game_date["GAME_DATE"] = [datetime.datetime.strptime(i, "%B %d, %Y").date() for i in game_date["GAME_DATE"]]
    return game_date


# Add game statistics
def add_game_stats(table, table1):
    """ Creates Game Count variable! Needs to create home-away variable first! """
    return pd.merge(table, table1, on=["TEAM_ID", "GAME_ID"])


# #### Creates Game counter variable.
def create_game_count_var(input_table):
    """ Creates Game Count variable! Needs to create home-away variable first! """
    input_table["G"] = 1
    input_table["G"] = input_table.sort_values(["TEAM_NAME", "GAME_DATE"]).groupby("TEAM_ID")["G"].transform(pd.Series.cumsum)
    return input_table.sort_values(["TEAM_ID", "G"])


# Creates a variable for number of days from previous game
def create_days_from_previous_games_var(main_table):
    main_table['p_games'] = main_table["GAME_DATE"] - main_table["GAME_DATE"].shift(1)
    main_table["p_games"] = [str(i).split(" ")[0] for i in main_table["p_games"]]
    main_table["p_games"] = pd.to_numeric(main_table["p_games"], errors=coerce)
    main_table.loc[main_table["p_games"] < 0, "p_games"] = np.nan
    return main_table

# Function to stack competing team stats side by side!
# - Currently, the stats are stored by per team in a single table. However, I each row in the table to reflect each game, with home and away game stats shown in 1 row to faciliate my analysis.


def create_opp_stats(table):
    """ Function duplicates all games, and merges them to form home and away records in each row.
        This directly doubles the number of games stats that I have in the table, as now I have
        both team_a and team_b from each game merged side by side. s"""
    opp_merge = table.copy()
    opp_merge.Home = table.Home.replace("Home", "Away_2").replace("Away", "Home").replace("Away_2", "Away")

    return pd.merge(table.drop(["PLUS_MINUS", "SEASON", "LIVE_PERIOD"], axis=1),
                    opp_merge.drop(["PLUS_MINUS"], axis=1),
                    on=["Home", "GAME_ID", "GAME_DATE"], how="left")


# #### Create Win-Loss variables!
# - Currently, I am just focusing on predicting win-loss of each game, base on home and away team statistics. This is the function that creates my win-loss variable. **Future iterations of my prediction models can consider for estimating expected points scored by the home and away team**, and using those estimates to calculate the points spread between the 2 teams. Would definitely be very intereting to see which model gives better predictions!


def create_win_loss_vars(main_table):
    main_table["WL_x"] = 0
    main_table["W_x"] = 0
    main_table["L_x"] = 0
    main_table["W_y"] = 0
    main_table["L_y"] = 0
    main_table.loc[main_table["PTS_x"] > main_table["PTS_y"], "WL_x"] = 1
    main_table.loc[main_table["PTS_x"] > main_table["PTS_y"], "W_x"] = 1
    main_table.loc[main_table["PTS_x"] < main_table["PTS_y"], "L_x"] = 1
    main_table.loc[main_table["PTS_x"] < main_table["PTS_y"], "W_y"] = 1
    main_table.loc[main_table["PTS_x"] > main_table["PTS_y"], "L_y"] = 1
    return main_table


# #### Creation of average and shooting percentage statistics
# - **Per game statistics:** Divide cumulative game stats by total games thus far (Requires n-th games played in season variable).
# - **Shooting percentage statistics:** Cumulatively divide total shot made by total shots attempted.

# #### Averages!
# **Important note:** The initial table are built on the stats from team_x and team_y (aka team_x_opp). Hence, calculation of averages have to be done on the cumulated game_dates from **G_x** only! *__Counter example is when I previously used G_y to calculate the averages for team_y, and where team_y was on G_y == 2, but team_x is on G_x == 1. Hence, the numbers to represent how strong team_x has been against their opponents only had 1 game, but I actually used 2 games.__*


def create_averages(table_return, variables):
    """ Averages are done by cumulatively summing the stats by games, and then dividing by no of games!
        I will directly over-write the columns to make the table manageable."""
    table = table_return.copy()
    table[[i + "_x" for i in variables]] = table[[i + "_x" for i in variables]].astype('float').div(table['G_x'].astype('float'), axis='index')

    table[[i + "_y" for i in variables]] = table[[i + "_y" for i in variables]].astype('float').div(table['G_x'].astype('float'), axis='index')
    return table


# #### Creating Shooting Percentages!
def create_shooting_percentage_vars(main_table):
    for i in ["y", "x"]:
        main_table["FGP_" + i] = main_table["FGM_" + i] / main_table["FGA_" + i]
        main_table["FG3P_" + i] = main_table["FG3M_" + i] / main_table["FG3A_" + i]
        main_table["FTP_" + i] = main_table["FTM_" + i] / main_table["FTA_" + i]
    return main_table


# #### NBA domain-knowledge stats
# - I am also adding NBA domain-knowledge stats (for a lack of a better name) to see how much they can help with my prediction scores.

# 1. **Efficiency ratings: EFG% = (FGM + (0.5 * 3PM)) / FGA **
def create_efg_var(main_table):
    main_table["EFG_x"] = (main_table["FGM_x"] + (.5 * main_table["FG3M_x"])) / main_table["FGA_x"]
    main_table["FGP_x"] = main_table["FGM_x"] / main_table["FGA_x"]

    main_table["EFG_y"] = (main_table["FGM_y"] + (.5 * main_table["FG3M_y"])) / main_table["FGA_y"]
    main_table["FGP_y"] = main_table["FGM_y"] / main_table["FGA_y"]

    return main_table


# 1. **Free-throw attempts to Field-goal attempts ratio**
def create_fta_to_fga_ratio(main_table):
    main_table["FTA_FGA_x"] = main_table["FTA_x"] / main_table["FGA_x"]
    main_table["FTA_FGA_y"] = main_table["FTA_y"] / main_table["FGA_y"]
    return main_table


# 2. **Offensive rebounding percentage (OREB%) = Offensive rebounds / (Offensive rebounds + Opponent defensive rebounds)**
# 3. **Defensive rebounding percentage (DREB%) = Defensive rebounds / (Defensive rebounds + Opponent offensive rebounds)**


def create_rebs_efficiency_vars(main_table):
    main_table["oreb_p_x"] = main_table["OREB_x"] / (main_table["OREB_x"] + main_table["DREB_y"])
    main_table["dreb_p_x"] = main_table["DREB_x"] / (main_table["DREB_x"] + main_table["OREB_y"])

    main_table["oreb_p_y"] = main_table["OREB_y"] / (main_table["OREB_y"] + main_table["DREB_x"])
    main_table["dreb_p_y"] = main_table["DREB_y"] / (main_table["DREB_y"] + main_table["OREB_x"])
    return main_table


# Filter for variables that I want:** Shifting the stats so predictions can be based on past cumulated data
def shift_game_stats_down_by_one(main_table):
    """ Shifting the games stats down by one (for each team each season)
        so that predictions can be based on past cumulated data."""
    base_shift = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'W', 'L', "EFG", "FGP", "oreb_p", "dreb_p", "FG3P", "FTP"]

    vars_to_shift = [i + "_x" for i in base_shift] + [i + "_y" for i in base_shift] + ["FTA_FGA_x", "FTA_FGA_y"]
    final_shift = main_table.copy()
    final_shift[vars_to_shift] = final_shift[vars_to_shift].shift(1)
    return final_shift[(final_shift["G_x"] > 1)]


# Base table is on TEAM_A:
# - Base table has team_a (x) and team_a_opp (y) stats. Duplicating from the same table, the final step is adding **team_b** and **team_b opponents** to the base table!
# - **(1) final_shift is the base (2) final_shift_opp is the appending table.**
# - From final_shift, remove ta_opp stuff (because those will come from final_shift_opp)
# - From final_shift_opp, remove **"WL_tb"** and **"Home"**, because base is on final_shift!
def create_team_ab_and_opp_table(final_shift):
    # Create dup table
    final_shift_opp = final_shift.copy()

    # Rename team_x and team_y to team_a and team_a_opponents
    final_shift.columns = [i.replace("_x", "_ta").replace("_y", "_ta_opp") for i in final_shift.columns]

    # Rename team_x and team_y to team_b and team_b_opponents
    final_shift_opp.columns = [i.replace("_x", "_tb").replace("_y", "_tb_opp") for i in final_shift_opp.columns]

    # Renaming final_shift_opp so that I can merge the 2 tables the same IDs...
    final_shift_opp.rename(columns={"TEAM_ID_tb_opp": "TEAM_ID_ta", "TEAM_NAME_tb_opp": "TEAM_NAME_ta"}, inplace=True)

    ffinals = pd.merge(final_shift.drop(["TEAM_ID_ta_opp", "TEAM_NAME_ta_opp"], axis=1), final_shift_opp.drop(["WL_tb", "Home", "SEASON", "LIVE_PERIOD"], axis=1), on=["GAME_ID", "TEAM_ID_ta", "TEAM_NAME_ta", "GAME_DATE"])
    return ffinals


# #### Keep only Home
def filter_home_teams(main_table):
    finals_home = main_table[main_table["Home"] == "Home"]
    finals_home.columns = [i.lower() for i in finals_home.columns]
    finals_home.rename(columns={"w_ta": "w_rate_ta", "w_tb": "w_rate_tb"}, inplace=True)
    return finals_home
