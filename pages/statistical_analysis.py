import sys
sys.path.append('.')
from scipy.stats import poisson, skellam
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dixon_coles import rho_correction
import streamlit as st

st.header("📊 Statistical Analysis")

# Load data
try:
    fizzdf = pd.read_excel("FIZZ2526.xlsx", index_col=0)
    st.write("Data loaded successfully. Shape:", fizzdf.shape)
    st.dataframe(fizzdf.head())
except FileNotFoundError:
    st.error("FIZZ2526.xlsx not found. Please ensure the file exists in the parent directory.")
    st.stop()

st.subheader("Poisson Distribution for Goals")

# Compute means
home_mean = fizzdf['HomeGoal'].mean()
away_mean = fizzdf['AwayGoal'].mean()

st.write(f"Home goals mean: {home_mean:.2f}")
st.write(f"Away goals mean: {away_mean:.2f}")

# Poisson predictions
poisson_pred = np.column_stack([[poisson.pmf(i, [home_mean, away_mean][j]) for i in range(8)] for j in range(2)])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(fizzdf[['HomeGoal', 'AwayGoal']].values, range(9), 
        alpha=0.7, label=['Home', 'Away'], color=["#FFA07A", "#20B2AA"])

# add lines for the Poisson distributions
pois1, = ax.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0]*len(fizzdf)*0.5,  # Approximate scaling
                  linestyle='-', marker='o',label="Home Poisson", color = '#CD5C5C')
pois2, = ax.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1]*len(fizzdf)*0.5,
                  linestyle='-', marker='o',label="Away Poisson", color = '#006400')

ax.legend(loc='upper right')
ax.set_xticks([i-0.5 for i in range(1,9)])
ax.set_xticklabels([i for i in range(8)])
ax.set_xlabel("Goals per Match")
ax.set_ylabel("Number of Matches")
ax.set_title("Number of Goals per Match")
st.pyplot(fig)

# Skellam for match outcomes
st.subheader("Match Outcome Probabilities (Skellam Distribution)")

skellam_pred = [skellam.pmf(i, home_mean, away_mean) for i in range(-6,8)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(-6,8), skellam_pred)
ax.set_xlabel("Goal Difference (Home - Away)")
ax.set_ylabel("Probability")
ax.set_title("Skellam Distribution of Goal Differences")
st.pyplot(fig)

prob_draw = skellam.pmf(0, home_mean, away_mean)
prob_home_win_1 = skellam.pmf(1, home_mean, away_mean)
prob_away_win_1 = skellam.pmf(-1, home_mean, away_mean)

st.write(f"Probability of draw: {prob_draw:.3f}")
st.write(f"Probability of home win by 1 goal: {prob_home_win_1:.3f}")
st.write(f"Probability of away win by 1 goal: {prob_away_win_1:.3f}")

st.subheader("Dixon-Coles Model")

def dc_log_like(params, dataset):
    teams = np.sort(dataset['home_team'].unique())
    n_teams = len(teams)
    
    attack_coefs = dict(zip(teams, params[:n_teams]))
    defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
    rho, home_adv = params[-2:]
    
    log_like = 0
    for row in dataset.itertuples():
        lambda_x = np.exp(attack_coefs[row.home_team] + defend_coefs[row.away_team] + home_adv)
        mu_y = np.exp(attack_coefs[row.away_team] + defend_coefs[row.home_team])
        
        row_log_like = (np.log(rho_correction(row.HomeGoal, row.AwayGoal, lambda_x, mu_y, rho)) + 
                        poisson.logpmf(row.HomeGoal, lambda_x) + 
                        poisson.logpmf(row.AwayGoal, mu_y))
        log_like += row_log_like
    
    return -log_like  # Negative for minimization

if st.button("Run Dixon-Coles Optimization"):
    with st.spinner("Running optimization..."):
        # Prepare data for Dixon-Coles
        # Assuming columns: home_team, away_team, HomeGoal, AwayGoal
        if 'home_team' in fizzdf.columns and 'away_team' in fizzdf.columns:
            teams = np.sort(fizzdf['home_team'].unique())
            n_teams = len(teams)
            
            # Initial values
            init_vals = np.concatenate((np.ones(n_teams), -np.ones(n_teams), [0, 0.2]))
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:n_teams]) - n_teams}]
            
            # Bounds
            bounds = [(-5, 5)] * (2 * n_teams + 2)
            
            # Optimize
            opt = minimize(dc_log_like, init_vals, args=(fizzdf,), constraints=constraints, bounds=bounds, options={'disp': False, 'maxiter': 200})
            
            if opt.success:
                params = opt.x
                attack_coefs = dict(zip(teams, params[:n_teams]))
                defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
                rho, home_adv = params[-2:]
                
                st.write(f"Home advantage: {home_adv:.3f}")
                st.write(f"Rho correction: {rho:.3f}")
                
                # Plot team strengths
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                teams_list = list(attack_coefs.keys())
                attack_vals = list(attack_coefs.values())
                defense_vals = list(defend_coefs.values())
                
                ax1.barh(teams_list, attack_vals, color='green', alpha=0.7)
                ax1.set_title('Attack Strengths')
                ax1.set_xlabel('Strength')
                
                ax2.barh(teams_list, defense_vals, color='red', alpha=0.7)
                ax2.set_title('Defense Strengths')
                ax2.set_xlabel('Strength')
                
                st.pyplot(fig)
                
                # Team strengths table
                strengths_df = pd.DataFrame({
                    'Team': teams_list,
                    'Attack': attack_vals,
                    'Defense': defense_vals
                })
                st.dataframe(strengths_df)
            else:
                st.error("Optimization failed.")
        else:
            st.warning("Dixon-Coles requires 'home_team', 'away_team', 'HomeGoal', 'AwayGoal' columns. Columns found: " + str(list(fizzdf.columns)))