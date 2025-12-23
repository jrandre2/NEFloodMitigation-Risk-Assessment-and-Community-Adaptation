#!/usr/bin/env python3
"""
Owner-level SFHA intensity model for Table S6.
Addresses Reviewer 2's concern about multi-property owners being mechanically
more likely to own SFHA homes by modeling at the owner level with offset.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, NegativeBinomial
import warnings
warnings.filterwarnings('ignore')

# Load parcel classification data (has owner names)
print("Loading parcel classification data...")
parcel_df = pd.read_csv(
    '/Users/jesseandrews/projects/2025_Q4/Flood/Flood_Projects/results/production_classification_results/parcel_classification_complete_20250802_192450.csv',
    usecols=['Parcel_ID', 'owner_name', 'predicted_owner_type', 'prediction_confidence']
)

# Load SFR regression data (has SFHA status)
print("Loading SFR regression data...")
sfr_df = pd.read_csv(
    '/Users/jesseandrews/projects/2025_Q4/Flood/Flood_Projects/results/integration_run/sfr_regression_data.csv',
    usecols=['parcel_id', 'in_sfha', 'owner_Corporation', 'owner_LLC', 'owner_Other', 'owner_Trust']
)

# Merge on parcel ID
print("Merging datasets...")
# Try different ID column names
parcel_df['parcel_id'] = parcel_df['Parcel_ID'].astype(str)
sfr_df['parcel_id'] = sfr_df['parcel_id'].astype(str)

merged = pd.merge(sfr_df, parcel_df[['parcel_id', 'owner_name', 'predicted_owner_type']],
                  on='parcel_id', how='inner')

print(f"Merged: {len(merged)} SFR parcels with owner names")

# Create entity type from predicted_owner_type
merged['entity_type'] = merged['predicted_owner_type'].fillna('Individual')

# Aggregate to owner level
print("\nAggregating to owner level...")
owner_agg = merged.groupby(['owner_name', 'entity_type']).agg(
    total_parcels=('parcel_id', 'count'),
    sfha_parcels=('in_sfha', 'sum')
).reset_index()

print(f"Number of unique owners: {len(owner_agg)}")

# Summary statistics by entity type
print("\n=== Owner-Level Summary Statistics ===")
summary = owner_agg.groupby('entity_type').agg(
    n_owners=('owner_name', 'count'),
    total_parcels=('total_parcels', 'sum'),
    mean_parcels=('total_parcels', 'mean'),
    total_sfha=('sfha_parcels', 'sum'),
    mean_sfha=('sfha_parcels', 'mean'),
    pct_any_sfha=('sfha_parcels', lambda x: (x > 0).mean() * 100)
).round(3)

print(summary.to_string())

# Prepare for Poisson model
print("\n=== Running Owner-Level Poisson Model ===")

# Create dummy variables for entity type (Individual as reference)
owner_agg['is_LLC'] = (owner_agg['entity_type'] == 'LLC').astype(int)
owner_agg['is_Corporation'] = (owner_agg['entity_type'] == 'Corporation').astype(int)
owner_agg['is_Trust'] = (owner_agg['entity_type'] == 'Trust').astype(int)
owner_agg['is_Other'] = (owner_agg['entity_type'] == 'Other').astype(int)

# Create offset (log of total parcels)
owner_agg['log_total'] = np.log(owner_agg['total_parcels'])

# Design matrix
X = owner_agg[['is_LLC', 'is_Corporation', 'is_Trust', 'is_Other']]
X = sm.add_constant(X)

# Outcome
y = owner_agg['sfha_parcels']

# Fit Poisson model with offset
print("Fitting Poisson model with offset log(total_parcels)...")
poisson_model = sm.GLM(y, X, family=Poisson(), offset=owner_agg['log_total'])
poisson_results = poisson_model.fit()

print("\n=== Poisson Model Results ===")
print(poisson_results.summary())

# Extract IRRs and CIs
print("\n=== Incidence Rate Ratios (IRR) ===")
params = poisson_results.params
conf = poisson_results.conf_int()
pvalues = poisson_results.pvalues

irr_df = pd.DataFrame({
    'Coefficient': params,
    'IRR': np.exp(params),
    'CI_lower': np.exp(conf[0]),
    'CI_upper': np.exp(conf[1]),
    'p_value': pvalues
})

print(irr_df.round(4).to_string())

# Format for Table S6
print("\n\n=== TABLE S6 OUTPUT ===")
print("Owner-level SFHA intensity model (Poisson with offset)")
print("Reference category: Individual")
print("-" * 80)

entity_order = ['Individual', 'LLC', 'Corporation', 'Trust', 'Gov/Nonprofit']
entity_map = {
    'const': 'Individual (reference)',
    'is_LLC': 'LLC',
    'is_Corporation': 'Corporation',
    'is_Trust': 'Trust',
    'is_Other': 'Gov/Nonprofit'
}

# Get owner counts by entity type
owner_counts = owner_agg.groupby('entity_type')['owner_name'].count()

print(f"\n| Entity type | N owners | Mean SFHA parcels | IRR [95% CI] | p-value |")
print(f"|-------------|----------|-------------------|--------------|---------|")

# Individual (reference)
ind_owners = summary.loc['Individual', 'n_owners'] if 'Individual' in summary.index else 0
ind_mean = summary.loc['Individual', 'mean_sfha'] if 'Individual' in summary.index else 0
print(f"| Individual | {int(ind_owners):,} | {ind_mean:.3f} | 1.000 (ref) | — |")

# Other entity types
for var, entity in [('is_LLC', 'LLC'), ('is_Corporation', 'Corporation'),
                     ('is_Trust', 'Trust'), ('is_Other', 'Gov/Nonprofit')]:
    if var in irr_df.index:
        irr = irr_df.loc[var, 'IRR']
        ci_l = irr_df.loc[var, 'CI_lower']
        ci_u = irr_df.loc[var, 'CI_upper']
        pval = irr_df.loc[var, 'p_value']

        # Get entity type name for lookup in summary
        entity_lookup = 'Other' if entity == 'Gov/Nonprofit' else entity
        n_owners = summary.loc[entity_lookup, 'n_owners'] if entity_lookup in summary.index else 0
        mean_sfha = summary.loc[entity_lookup, 'mean_sfha'] if entity_lookup in summary.index else 0

        pval_str = '<0.001' if pval < 0.001 else f'{pval:.3f}'
        print(f"| {entity} | {int(n_owners):,} | {mean_sfha:.3f} | {irr:.3f} [{ci_l:.3f}–{ci_u:.3f}] | {pval_str} |")

print("\n*Note: Poisson GLM with log link; offset = log(total SFR parcels owned). N = {:,} owners.*".format(len(owner_agg)))

# Check for overdispersion
print("\n=== Overdispersion Check ===")
pearson_chi2 = poisson_results.pearson_chi2
df_resid = poisson_results.df_resid
dispersion = pearson_chi2 / df_resid
print(f"Pearson chi-squared: {pearson_chi2:.2f}")
print(f"Residual DF: {df_resid}")
print(f"Dispersion parameter: {dispersion:.3f}")

if dispersion > 1.5:
    print("\nSubstantial overdispersion detected. Fitting Negative Binomial model...")
    try:
        nb_model = sm.GLM(y, X, family=NegativeBinomial(alpha=1.0), offset=owner_agg['log_total'])
        nb_results = nb_model.fit()
        print("\n=== Negative Binomial Results ===")
        nb_params = nb_results.params
        nb_conf = nb_results.conf_int()
        nb_pvalues = nb_results.pvalues

        nb_irr_df = pd.DataFrame({
            'IRR': np.exp(nb_params),
            'CI_lower': np.exp(nb_conf[0]),
            'CI_upper': np.exp(nb_conf[1]),
            'p_value': nb_pvalues
        })
        print(nb_irr_df.round(4).to_string())
    except Exception as e:
        print(f"NB model failed: {e}")
else:
    print("Dispersion acceptable for Poisson model.")

# Save results
results_df = pd.DataFrame({
    'entity_type': ['Individual', 'LLC', 'Corporation', 'Trust', 'Gov/Nonprofit'],
    'n_owners': [
        summary.loc['Individual', 'n_owners'] if 'Individual' in summary.index else 0,
        summary.loc['LLC', 'n_owners'] if 'LLC' in summary.index else 0,
        summary.loc['Corporation', 'n_owners'] if 'Corporation' in summary.index else 0,
        summary.loc['Trust', 'n_owners'] if 'Trust' in summary.index else 0,
        summary.loc['Other', 'n_owners'] if 'Other' in summary.index else 0
    ],
    'mean_sfha_parcels': [
        summary.loc['Individual', 'mean_sfha'] if 'Individual' in summary.index else 0,
        summary.loc['LLC', 'mean_sfha'] if 'LLC' in summary.index else 0,
        summary.loc['Corporation', 'mean_sfha'] if 'Corporation' in summary.index else 0,
        summary.loc['Trust', 'mean_sfha'] if 'Trust' in summary.index else 0,
        summary.loc['Other', 'mean_sfha'] if 'Other' in summary.index else 0
    ],
    'IRR': [1.0] + [np.exp(params[v]) if v in params.index else np.nan
                    for v in ['is_LLC', 'is_Corporation', 'is_Trust', 'is_Other']],
    'CI_lower': [np.nan] + [np.exp(conf[0][v]) if v in conf[0].index else np.nan
                             for v in ['is_LLC', 'is_Corporation', 'is_Trust', 'is_Other']],
    'CI_upper': [np.nan] + [np.exp(conf[1][v]) if v in conf[1].index else np.nan
                             for v in ['is_LLC', 'is_Corporation', 'is_Trust', 'is_Other']],
    'p_value': [np.nan] + [pvalues[v] if v in pvalues.index else np.nan
                           for v in ['is_LLC', 'is_Corporation', 'is_Trust', 'is_Other']]
})

results_df.to_csv('/Users/jesseandrews/projects/2025_Q4/Flood/Flood_Projects/results/integration_run/owner_level_sfha_model_results.csv', index=False)
print("\nResults saved to: results/integration_run/owner_level_sfha_model_results.csv")
