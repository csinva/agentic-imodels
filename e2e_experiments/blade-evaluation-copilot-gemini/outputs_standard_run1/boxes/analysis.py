
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imodels import RuleFitClassifier
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
from scipy.stats import chi2_contingency

def analyze_data():
    # Load data
    data = pd.read_csv('boxes.csv')
    with open('info.json', 'r') as f:
        info = json.load(f)

    # Preprocessing
    # Convert y to a binary variable: 1 if majority option was chosen, 0 otherwise
    data['chose_majority'] = (data['y'] == 2).astype(int)

    # One-hot encode culture
    data = pd.get_dummies(data, columns=['culture'], prefix='culture', dtype=int)

    # Define features and target
    features = [col for col in data.columns if col not in ['y', 'chose_majority']]
    X = data[features]
    y = data['chose_majority']

    # --- Modeling ---

    # 1. Logistic Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    
    # Get coefficients
    coef_df = pd.DataFrame({'feature': features, 'coefficient': log_reg.coef_[0]})
    age_coef = coef_df[coef_df['feature'] == 'age']['coefficient'].iloc[0]

    # 2. Statsmodels for p-values
    X_sm = sm.add_constant(X)
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit()
    age_pvalue = result.pvalues['age']

    # --- Interpretation ---
    
    # The research question is: "How do children's reliance on majority preference develop over growth in age across different cultural contexts?"
    
    # We are looking for a positive relationship between age and choosing the majority option.
    # The logistic regression coefficient for age is positive, suggesting that as age increases, the log-odds of choosing the majority option increase.
    # The p-value for the age coefficient from the statsmodels output will tell us if this relationship is statistically significant.

    is_significant = age_pvalue < 0.05
    is_positive_relationship = age_coef > 0

    if is_significant and is_positive_relationship:
        response = 95
        explanation = f"There is a strong, statistically significant positive relationship between age and the likelihood of choosing the majority option (coefficient: {age_coef:.3f}, p-value: {age_pvalue:.3f}). This indicates that as children get older, they are more likely to follow the majority preference. The analysis accounts for cultural differences by including culture as a covariate."
    elif is_positive_relationship:
        response = 60
        explanation = f"There is a positive relationship between age and choosing the majority option (coefficient: {age_coef:.3f}), but it is not statistically significant (p-value: {age_pvalue:.3f}). This provides weak evidence that older children might be more likely to follow the majority, but the effect is not strong enough to be conclusive."
    else:
        response = 10
        explanation = f"There is no significant relationship (or a negative one) between age and choosing the majority option (coefficient: {age_coef:.3f}, p-value: {age_pvalue:.3f}). The data does not support the idea that reliance on majority preference increases with age in this context."


    # --- Create conclusion file ---
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze_data()
