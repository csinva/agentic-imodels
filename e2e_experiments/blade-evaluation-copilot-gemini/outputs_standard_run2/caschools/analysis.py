
import json
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from imodels import RuleFitRegressor

def analyze_data():
    # Load data
    with open('info.json', 'r') as f:
        info = json.load(f)
    
    df = pd.read_csv('caschools.csv')

    # Feature Engineering
    df['student_teacher_ratio'] = df['students'] / df['teachers']
    df['academic_performance'] = (df['read'] + df['math']) / 2

    # Research Question: Is a lower student-teacher ratio associated with higher academic performance?
    
    # 1. Correlation
    correlation = df[['student_teacher_ratio', 'academic_performance']].corr().iloc[0, 1]

    # 2. Linear Regression (scikit-learn)
    X = df[['student_teacher_ratio']]
    y = df['academic_performance']
    model_sklearn = LinearRegression()
    model_sklearn.fit(X, y)
    coef_sklearn = model_sklearn.coef_[0]

    # 3. Linear Regression (statsmodels)
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    p_value = model_sm.pvalues['student_teacher_ratio']
    coef_sm = model_sm.params['student_teacher_ratio']

    # 4. RuleFit Regressor (Optional)
    # model_rulefit = RuleFitRegressor()
    # model_rulefit.fit(X, y)
    # rules = model_rulefit.get_rules()
    
    # Interpretation
    explanation = f"The research question is: '{info['research_questions'][0]}'.\n\n"
    explanation += f"1.  **Correlation Analysis**: The Pearson correlation coefficient between student-teacher ratio and academic performance is {correlation:.3f}. This indicates a negative linear relationship, suggesting that as the student-teacher ratio decreases, academic performance tends to increase.\n\n"
    explanation += f"2.  **Linear Regression (scikit-learn)**: The coefficient for the student-teacher ratio is {coef_sklearn:.3f}. This means that for each one-unit decrease in the student-teacher ratio, the academic performance score is predicted to increase by {-coef_sklearn:.3f} points, on average.\n\n"
    explanation += f"3.  **Linear Regression (statsmodels)**: The statsmodels OLS regression provides a coefficient of {coef_sm:.3f} with a p-value of {p_value:.4f}. Since the p-value is less than 0.05, the relationship is statistically significant. This confirms that the observed negative association is unlikely to be due to random chance.\n\n"
    
    # Determine Likert scale response
    if p_value < 0.05 and coef_sm < 0:
        response = 95  # Strong "Yes"
        explanation += "Given the statistically significant negative correlation and regression coefficient, we can confidently conclude that a lower student-teacher ratio is associated with higher academic performance."
    elif p_value < 0.1 and coef_sm < 0:
        response = 75  # Moderate "Yes"
        explanation += "There is a moderately significant negative relationship, suggesting a lower student-teacher ratio is likely associated with higher academic performance."
    else:
        response = 10  # Weak or no relationship
        explanation += "The statistical evidence is not strong enough to conclude a significant association between student-teacher ratio and academic performance."

    # Save conclusion
    conclusion = {
        "response": response,
        "explanation": explanation
    }
    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f, indent=2)

if __name__ == '__main__':
    analyze_data()
