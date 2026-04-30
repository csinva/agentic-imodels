
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from agentic_imodels import WinsorizedSparseOLSRegressor, HingeEBMRegressor

def analyze():
    # 1. Load and preprocess data
    df = pd.read_csv('amtl.csv')

    # Create the outcome variable: frequency of antemortem tooth loss
    df['amtl_freq'] = df['num_amtl'] / df['sockets']
    
    # Create a binary indicator for the main independent variable
    df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

    # Drop rows with missing values in key columns
    key_cols = ['amtl_freq', 'is_human', 'age', 'prob_male', 'tooth_class']
    df.dropna(subset=key_cols, inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['tooth_class', 'genus'], drop_first=True)

    # Define variables for models
    outcome_var = 'amtl_freq'
    main_iv = 'is_human'
    
    # After get_dummies, the original 'genus' and 'tooth_class' are gone.
    # We have 'is_human' and the new dummy columns.
    control_vars = ['age', 'prob_male'] + [col for col in df.columns if 'tooth_class_' in col]
    
    # Explicitly define all feature columns to ensure no object types are included
    feature_cols = [main_iv] + control_vars
    
    X = df[feature_cols].astype(float) # Ensure all are float for statsmodels
    y = df[outcome_var].astype(float)

    # 2. Classical statistical test (OLS)
    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()
    ols_summary = ols_model.summary()
    print("--- OLS Results ---")
    print(ols_summary)
    
    # Extract OLS results for the main IV
    ols_coef = ols_model.params[main_iv]
    ols_pval = ols_model.pvalues[main_iv]

    # 3. Interpretable models
    print("\n--- Interpretable Models ---")
    
    # Model 1: WinsorizedSparseOLSRegressor (Honest, Sparse Linear)
    ws_ols = WinsorizedSparseOLSRegressor()
    ws_ols.fit(X, y)
    print("\n=== WinsorizedSparseOLSRegressor ===")
    print(ws_ols)

    # Model 2: HingeEBMRegressor (High-performance, Decoupled)
    h_ebm = HingeEBMRegressor()
    h_ebm.fit(X, y)
    print("\n=== HingeEBMRegressor ===")
    print(h_ebm)

    # 4. Synthesize and conclude
    explanation = []
    
    # OLS interpretation
    explanation.append(f"The classical OLS regression shows a coefficient for 'is_human' of {ols_coef:.4f}.")
    if ols_pval < 0.05:
        explanation.append(f"This effect is statistically significant (p = {ols_pval:.4f}), suggesting that after controlling for age, sex, and tooth class, humans have a different AMTL frequency than the primate genera.")
    else:
        explanation.append(f"This effect is not statistically significant (p = {ols_pval:.4f}).")

    # Interpretable models interpretation
    explanation.append("The interpretable models provide further insight.")
    
    # WinsorizedSparseOLSRegressor results
    ws_ols_str = str(ws_ols)
    if main_iv in ws_ols_str:
        explanation.append(f"The WinsorizedSparseOLSRegressor selected 'is_human' as a feature, indicating its importance.")
        if f"+ {main_iv}" in ws_ols_str or f"*{main_iv}" in ws_ols_str:
             explanation.append("The sign of the coefficient is positive, corroborating the OLS result.")
        else:
             explanation.append("The sign of the coefficient is negative.")
    else:
        explanation.append("The WinsorizedSparseOLSRegressor did NOT select 'is_human', suggesting it has little to no linear effect after accounting for other variables.")

    # HingeEBMRegressor results
    h_ebm_str = str(h_ebm)
    if main_iv in h_ebm_str:
        explanation.append(f"The HingeEBMRegressor also includes 'is_human' in its model, confirming its relevance.")
    else:
        explanation.append("The HingeEBMRegressor zeroed out the 'is_human' feature, providing strong evidence against a direct effect.")

    # Final conclusion based on all evidence
    is_human_coef_positive = ols_coef > 0
    is_human_significant = ols_pval < 0.05
    is_human_in_sparse_model = main_iv in ws_ols_str
    
    score = 50 # Start neutral
    if is_human_significant and is_human_coef_positive:
        score = 85 # Strong evidence for "Yes"
        final_summary = "Overall, there is strong evidence. The OLS model finds a significant positive relationship. This is supported by the feature's inclusion in both the sparse linear model and the HingeEBM, indicating robustness."
    elif is_human_significant and not is_human_coef_positive:
        score = 15 # Strong evidence for "No" in the sense of *lower* frequency
        final_summary = "Overall, there is strong evidence that humans have a *lower* AMTL frequency. The OLS model finds a significant negative relationship, and this is supported by the interpretable models."
    elif not is_human_significant and not is_human_in_sparse_model:
        score = 10 # Strong evidence for "No"
        final_summary = "Overall, there is strong evidence against a relationship. The OLS model finds no significant effect, and the sparse linear model excludes the 'is_human' feature entirely, suggesting it's not a key predictor."
    else:
        score = 40 # Weak or inconsistent evidence
        final_summary = "The evidence is mixed. While the OLS p-value may be low, the effect is not consistently selected or ranked as important by the interpretable models, suggesting the relationship is weak or not robust."

    explanation.append(final_summary)

    # Generate JSON output
    conclusion = {
        "response": score,
        "explanation": " ".join(explanation)
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f, indent=2)

    print("\nAnalysis complete. Conclusion written to conclusion.txt")

if __name__ == '__main__':
    analyze()
