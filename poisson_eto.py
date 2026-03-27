import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam


fizzdf = pd.read_excel("FIZZ2526.xlsx", index_col=0)
print(fizzdf.mean(numeric_only=True))

poisson_pred = np.column_stack([[poisson.pmf(i, fizzdf.mean(numeric_only=True)[j]) for i in range(8)] for j in range(2)])
print(poisson_pred)

# plot histogram of actual goals
# plot histogram of actual goals
#The plot below shows the proportion of goals scored compared to the number of goals estimated by the corresponding Poisson distributions.
# construct Poisson  for each mean goals value

plt.hist(fizzdf[['HomeGoal', 'AwayGoal']].values, range(9), 
         alpha=0.7, label=['Home', 'Away'], color=["#FFA07A", "#20B2AA"])

# add lines for the Poisson distributions
pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0]*100,
                  linestyle='-', marker='o',label="Home", color = '#CD5C5C')
pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1]*100,
                  linestyle='-', marker='o',label="Away", color = '#006400')

leg=plt.legend(loc='upper right', fontsize=13, ncol=2)
leg.set_title("Poisson           Actual        ", prop = {'size':'14', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(8)])
plt.xlabel("Goals per Match",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Number of Goals per Match (EPL 2016/17 Season)",size=14,fontweight='bold')
#plt.ylim([-0.004, 0.4])
plt.tight_layout()
plt.show()

# probability of draw between home and away team
print("probability of draw between home and away team:")
skellam.pmf(0.0,  fizzdf.mean(numeric_only=True)[0],  fizzdf.mean(numeric_only=True)[1])

# probability of home team winning by one goal
print("probability of home team winning by one goal:")
skellam.pmf(1,  fizzdf.mean(numeric_only=True)[0],  fizzdf.mean(numeric_only=True)[1])

# The difference between Home Goals and Away Goals
# Skellam distribution models the difference between two independent Poisson variables.
skellam_pred = [skellam.pmf(i,  fizzdf.mean(numeric_only=True)[0],  fizzdf.mean(numeric_only=True)[1]) for i in range(-6,8)]

plt.hist(fizzdf[['HomeGoal']].values - fizzdf[['AwayGoal']].values, range(-6,8), 
         alpha=0.7, label='Actual',density=True)
plt.plot([i+0.5 for i in range(-6,8)], skellam_pred,
                  linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')
plt.legend(loc='upper right', fontsize=13)
plt.xticks([i+0.5 for i in range(-6,8)],[i for i in range(-6,8)])
plt.xlabel("Home Goals - Away Goals",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Difference in Goals Scored (Home Team vs Away Team)",size=14,fontweight='bold')
plt.ylim([-0.004, 0.26])
plt.tight_layout()
plt.show()

# the distribution of goals scored by Ferencváros and ZTE FC

# 1. Initialize Figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

# Teams and Colors (Easily swappable for your Hungarian NB I teams!)
t1, t2 = 'FERENCVÁROSI TC', 'ZTE FC'
c1, c2 = "#034694", "#EB172B"
c1_l, c2_l = "#0a7bff", "#ff7c89"

def get_poisson_and_actual(df, team, col_name, is_home=True):
    side = 'home_team' if is_home else 'away_team'
    # Use single brackets to ensure 1D Series (Fixes ValueError)
    team_data = df[df[side] == team][col_name]
    
    # Calculate Actual Proportions and reindex to 0-7 goals (Fixes alignment)
    actual = team_data.value_counts(normalize=True).sort_index()
    actual = actual.reindex(range(8), fill_value=0)
    
    # Calculate Poisson Lambda (Mean)
    mean_goals = team_data.mean()
    pois_preds = [poisson.pmf(i, mean_goals) for i in range(8)]
    
    return actual, pois_preds

# --- Execute Calculations ---
c_h_act, c_h_pois = get_poisson_and_actual(fizzdf, t1, 'HomeGoal', True)
s_h_act, s_h_pois = get_poisson_and_actual(fizzdf, t2, 'HomeGoal', True)
c_a_act, c_a_pois = get_poisson_and_actual(fizzdf, t1, 'AwayGoal', False)
s_a_act, s_a_pois = get_poisson_and_actual(fizzdf, t2, 'AwayGoal', False)

# --- AX1: HOME PLOT ---
ax1.bar(c_h_act.index - 0.2, c_h_act.values, width=0.4, color=c1, label=t1)
ax1.bar(s_h_act.index + 0.2, s_h_act.values, width=0.4, color=c2, label=t2)
ax1.plot(range(8), c_h_pois, linestyle='-', marker='o', label=f"{t1} Poisson", color=c1_l)
ax1.plot(range(8), s_h_pois, linestyle='-', marker='o', label=f"{t2} Poisson", color=c2_l)

# --- AX2: AWAY PLOT ---
ax2.bar(c_a_act.index - 0.2, c_a_act.values, width=0.4, color=c1, label=t1)
ax2.bar(s_a_act.index + 0.2, s_a_act.values, width=0.4, color=c2, label=t2)
ax2.plot(range(8), c_a_pois, linestyle='-', marker='o', color=c1_l)
ax2.plot(range(8), s_a_pois, linestyle='-', marker='o', color=c2_l)

# --- Formatting & Style ---
for ax, side in zip([ax1, ax2], ["Home", "Away"]):
    ax.set_xlim([-0.5, 7.5])
    ax.set_ylim([-0.01, 0.65])
    ax.set_xticks(range(8)) # Fixes FixedLocator error
    # Right-side status labels
    ax.text(7.65, 0.3, f"      {side}      ", rotation=-90, verticalalignment='center',
            bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5}, weight='bold')

ax1.set_xticklabels([]) # Clean top subplot
ax1.set_title("Number of Goals per Match (Poisson vs Actual)", size=14, fontweight='bold')
ax2.set_xlabel("Goals per Match", size=13)
fig.text(0.04, 0.5, 'Proportion of Matches', va='center', rotation='vertical', size=13, weight='bold')

# Legend Styling
leg = ax1.legend(loc='upper right', fontsize=10, ncol=2)
leg.set_title("Poisson              Actual", prop={'size':'11', 'weight':'bold'})

plt.tight_layout()
plt.show()

# Building the Model

# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf

goal_model_data = pd.concat([fizzdf[['home_team','away_team','HomeGoal']].assign(home=1).rename(
            columns={'home_team':'team', 'away_team':'opponent','HomeGoal':'goals'}),
           fizzdf[['away_team','home_team','AwayGoal']].assign(home=0).rename(
            columns={'away_team':'team', 'home_team':'opponent','AwayGoal':'goals'})])
# fit the model
poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
print(poisson_model.summary())

# Make Prediction from the model
prediction = poisson_model.predict(pd.DataFrame(data={'team': 'FERENCVÁROSI TC', 'opponent': 'ZTE FC',
                                       'home':1},index=[1]))
print(f"Predicted goals for Ferencváros: {prediction.values[0]:.2f}")

prediction = poisson_model.predict(pd.DataFrame(data={'team': 'ZTE FC', 'opponent': 'FERENCVÁROSI TC',
                                       'home':0},index=[1]))
print(f"Predicted goals for ZTE FC: {prediction.values[0]:.2f}")

# Simulating the Full Match Result (simulate_match)
''' This function takes the expected goals for both teams and turns them into a Probability Matrix.'''

def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

# Test the simulate_match function
simulate_match(poisson_model, 'FERENCVÁROSI TC', 'ZTE FC', max_goals=3)

# Basic matrix manipulation functions to perform these calculations.
chel_sun = simulate_match(poisson_model, "FERENCVÁROSI TC", "ZTE FC", max_goals=10)

# Extract the outcomes from the matrix
# np.tril (lower triangle) = Home Wins
# np.diag (diagonal line) = Draws
# np.triu (upper triangle) = Away Winsz
win_prob = np.sum(np.tril(chel_sun, -1))
draw_prob = np.sum(np.diag(chel_sun))
loss_prob = np.sum(np.triu(chel_sun, 1))

print(f"Home Win: {win_prob:.2%}")
print(f"Draw: {draw_prob:.2%}")
print(f"Away Win: {loss_prob:.2%}")

