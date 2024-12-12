import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# URL for the dataset
data_url = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv"

# Load the dataset
game_data = pd.read_csv(data_url)

# Calculate and display retention rates
avg_retention = game_data.groupby('version')[['retention_1', 'retention_7']].mean()
print("\nAverage Retention Rates by Version:")
print(avg_retention)

def perform_bayesian_analysis(retention_metric):
    retention_version_30 = game_data[game_data['version'] == 'gate_30'][retention_metric]
    retention_version_40 = game_data[game_data['version'] == 'gate_40'][retention_metric]

    with pm.Model() as retention_model:
        prob_30 = pm.Uniform(f'prob_30_{retention_metric}', lower=0, upper=1)
        prob_40 = pm.Uniform(f'prob_40_{retention_metric}', lower=0, upper=1)

        obs_30 = pm.Bernoulli(f'obs_30_{retention_metric}', p=prob_30, observed=retention_version_30)
        obs_40 = pm.Bernoulli(f'obs_40_{retention_metric}', p=prob_40, observed=retention_version_40)

        retention_difference = pm.Deterministic('retention_difference', prob_30 - prob_40)

        retention_trace = pm.sample(20000, chains=2)

    return retention_trace

def visualize_and_summarize_results(trace, retention_metric):
    # Visualize posterior distributions for probabilities
    plt.figure(figsize=(12.5, 4))

    prob_30_samples = np.concatenate(trace.posterior[f'prob_30_{retention_metric}'].data[:, 1000:])
    prob_40_samples = np.concatenate(trace.posterior[f'prob_40_{retention_metric}'].data[:, 1000:])

    plt.hist(prob_30_samples, bins=40, label='Posterior for Prob_30', density=True, alpha=0.7)
    plt.hist(prob_40_samples, bins=40, label='Posterior for Prob_40', density=True, alpha=0.7)
    plt.title(f'{retention_metric.capitalize()} Retention Probabilities')
    plt.legend()
    plt.show()

    # Visualize posterior distribution for difference
    retention_diff_samples = np.concatenate(trace.posterior['retention_difference'].data[:, 1000:])

    plt.figure(figsize=(12.5, 4))
    plt.hist(retention_diff_samples, bins=40, label='Posterior for Retention Difference', density=True)
    plt.title(f'Difference in {retention_metric.capitalize()} Retention')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.legend()
    plt.show()

    # Display summary statistics
    print(f"\n{retention_metric.capitalize()}  Analysis Summary:")
    print(az.summary(trace, hdi_prob=0.95))

    # Calculate and display superiority probability
    prob_superiority = np.mean(retention_diff_samples > 0)
    return prob_superiority

# Analyze 1-day retention
print("Analyzing 1-Day Retention")
trace_1_day = perform_bayesian_analysis('retention_1')
prob_1_day = visualize_and_summarize_results(trace_1_day, 'retention_1')

# Analyze 7-day retention
print("Analyzing 7-Day Retention...")
trace_7_day = perform_bayesian_analysis('retention_7')
prob_7_day = visualize_and_summarize_results(trace_7_day, 'retention_7')
