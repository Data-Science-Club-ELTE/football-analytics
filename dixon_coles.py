import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

# 1. Load your Hungarian League data
# Replace with the path to your actual excel file
fizzdf = pd.read_excel("FIZZ2526.xlsx")

# 2. Dixon-Coles Rho Correction Function
# This adjusts the probabilities for 0-0, 1-0, 0-1, and 1-1 scores
def rho_correction(x, y, lambda_x, mu_y, rho):
    if x == 0 and y == 0:
        return 1 - (lambda_x * mu_y * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_x * rho)
    elif x == 1 and y == 0:
        return 1 + (mu_y * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

# 3. Log-Likelihood Function (The "Engine" of the model)
def solve_parameters(dataset, options={'disp': True, 'maxiter': 100}):
    teams = np.sort(dataset['home_team'].unique())
    n_teams = len(teams)
    
    # Initial values: Attack strengths (1.0), Defense strengths (-1.0), rho (0), home_adv (0.2)
    init_vals = np.concatenate((np.ones(n_teams), -np.ones(n_teams), [0, 0.2]))
    
    # Identifying constraint: Average attack strength must be 1
    # This keeps the numbers from getting infinitely large
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x[:n_teams]) - n_teams}]

    def dc_log_like(params):
        attack_coefs = dict(zip(teams, params[:n_teams]))
        defend_coefs = dict(zip(teams, params[n_teams:(2*n_teams)]))
        rho, home_adv = params[-2:]
        
        log_like = 0
        for row in dataset.itertuples():
            # Home team lambda (expected goals)
            lambda_x = np.exp(attack_coefs[row.home_team] + defend_coefs[row.away_team] + home_adv)
            # Away team lambda (expected goals)
            mu_y = np.exp(attack_coefs[row.away_team] + defend_coefs[row.home_team])
            
            # Apply Dixon-Coles adjustment to the log-likelihood
            row_log_like = (np.log(rho_correction(row.HomeGoal, row.AwayGoal, lambda_x, mu_y, rho)) + 
                            poisson.logpmf(row.HomeGoal, lambda_x) + 
                            poisson.logpmf(row.AwayGoal, mu_y))
            log_like += row_log_like
            
        return -log_like # We minimize the negative log-likelihood

    opt_output = minimize(dc_log_like, init_vals, options=options, constraints=constraints)
    
    return dict(zip(["attack_" + t for t in teams] + ["defence_" + t for t in teams] + ['rho', 'home_adv'], opt_output.x))

# 4. Predict Match Outcomes
def predict_outcome(params, home_team, away_team, max_goals=8):
    # Calculate lambdas
    lambda_x = np.exp(params['attack_' + home_team] + params['defence_' + away_team] + params['home_adv'])
    mu_y = np.exp(params['attack_' + away_team] + params['defence_' + home_team])
    
    # Probability matrices
    home_probs = poisson.pmf(range(max_goals + 1), lambda_x)
    away_probs = poisson.pmf(range(max_goals + 1), mu_y)
    m = np.outer(home_probs, away_probs)
    
    # Apply Dixon-Coles Rho adjustment to the 0 and 1 goal cells
    for x in range(2):
        for y in range(2):
            m[x, y] *= rho_correction(x, y, lambda_x, mu_y, params['rho'])
            
    # Calculate Win/Draw/Loss
    prob_home = np.sum(np.tril(m, -1))
    prob_draw = np.sum(np.diag(m))
    prob_away = np.sum(np.triu(m, 1))
    
    return prob_home, prob_draw, prob_away

if __name__ == "__main__":
    # --- Execution ---
    print("Optimizing Dixon-Coles parameters for NB I...")
    params = solve_parameters(fizzdf)

    # params

    # Example: Ferencváros vs ETO FC
    h, d, a = predict_outcome(params, 'FERENCVÁROSI TC', 'ETO FC')
    print(f"\nPrediction for FERENCVÁROSI TC vs ETO FC:")
    print(f"Home Win: {h:.2%}, Draw: {d:.2%}, Away Win: {a:.2%}")
    print(f"Dixon-Coles Rho (Draw Adjustment): {params['rho']:.4f}")