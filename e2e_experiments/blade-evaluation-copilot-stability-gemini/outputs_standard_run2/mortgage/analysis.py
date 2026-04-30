
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imodels import RuleFitClassifier
import statsmodels.api as sm
from scipy.stats import chi2_contingency

def analyze_mortgage_data():
    # Load data
    df = pd.read_csv('mortgage.csv')

    # Research Question: How does gender affect whether banks approve an individual’s mortgage application?
    
    # 1. Data Exploration
    # Drop unnecessary column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Target variable is 'deny'
    y = df['deny']
    X = df.drop(columns=['deny', 'accept']) # 'accept' is redundant with 'deny'

    # Handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())

    # 2. Statistical Test: Chi-squared test for independence
    contingency_table = pd.crosstab(df['female'], df['deny'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # 3. Interpretable Models
    # Split data for modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # b) Logistic Regression
    log_reg = LogisticRegression(random_state=42, solver='liblinear')
    log_reg.fit(X_train, y_train)
    
    # Get coefficients
    coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': log_reg.coef_[0]})
    female_coef = coef_df[coef_df['feature'] == 'female']['coefficient'].iloc[0]

    # 4. Interpretation and Conclusion
    # The p-value from the chi-squared test indicates the significance of the association.
    # The logistic regression coefficient for 'female' shows the direction and strength of the relationship.
    
    # Based on the p-value, a small p-value (e.g., < 0.05) suggests a significant association.
    # A positive coefficient for 'female' in the logistic regression would mean females are more likely to be denied.
    
    # Let's decide the response based on the p-value.
    # If p < 0.05, there is a statistically significant relationship.
    # We can then look at the coefficient to determine the direction.
    
    is_significant = p < 0.05
    
    if is_significant:
        # If significant, let's determine the strength.
        # A larger absolute coefficient means a stronger effect.
        # Let's set a threshold for the coefficient to determine the score.
        if abs(female_coef) > 0.1:
            response = 90 # Strong effect
        else:
            response = 70 # Weaker but significant effect
        explanation = f"There is a statistically significant relationship between gender and mortgage denial (p-value: {p:.4f}). The logistic regression coefficient for 'female' is {female_coef:.4f}, suggesting a non-zero effect."
    else:
        response = 10 # Not a significant relationship
        explanation = f"There is no statistically significant relationship between gender and mortgage denial (p-value: {p:.4f}). The logistic regression coefficient for 'female' is {female_coef:.4f}, which is not statistically significant."


    # Create conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze_mortgage_data()
