
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imodels import RuleFitClassifier
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
from scipy.stats import ttest_ind

def analyze():
    # Load data
    data = pd.read_csv('mortgage.csv')

    # Research Question: How does gender affect whether banks approve an individual’s mortgage application?
    
    # 1. Exploratory Data Analysis
    # Check for missing values
    # data.isnull().sum() # No missing values
    data = data.fillna(data.mean())

    # Summary statistics by gender
    # data.groupby('female')['deny'].describe()

    # 2. Statistical Tests
    # T-test for difference in denial rates between genders
    male_denials = data[data['female'] == 0]['deny']
    female_denials = data[data['female'] == 1]['deny']
    ttest_result = ttest_ind(male_denials, female_denials)

    # 3. Interpretable Models
    # Prepare data for modeling
    X = data.drop(['deny', 'accept', 'Unnamed: 0'], axis=1)
    y = data['deny']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    log_reg = LogisticRegression(random_state=42, solver='liblinear')
    log_reg.fit(X_train, y_train)
    log_reg_coef = pd.DataFrame({'feature': X.columns, 'coefficient': log_reg.coef_[0]})

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    
    # RuleFit
    rulefit = RuleFitClassifier(max_rules=10, random_state=42)
    rulefit.fit(X_train, y_train)
    rules = rulefit.rules_

    # Statsmodels OLS
    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()
    p_value_gender = ols_model.pvalues['female']


    # 4. Interpretation and Conclusion
    explanation = "To determine if gender affects mortgage approval, I analyzed the data using several methods. "
    
    # T-test result
    explanation += f"A t-test comparing the mortgage denial rates for males and females shows a p-value of {ttest_result.pvalue:.3f}. "
    if ttest_result.pvalue < 0.05:
        explanation += "This suggests a statistically significant difference in denial rates between genders. "
    else:
        explanation += "This suggests no statistically significant difference in denial rates between genders. "

    # Logistic Regression result
    gender_coef = log_reg_coef[log_reg_coef['feature'] == 'female']['coefficient'].iloc[0]
    explanation += f"A logistic regression model shows a coefficient of {gender_coef:.3f} for the 'female' variable. "
    if gender_coef > 0:
        explanation += "This indicates that being female is associated with a higher likelihood of denial, holding other factors constant. "
    else:
        explanation += "This indicates that being female is associated with a lower likelihood of denial, holding other factors constant. "

    # Statsmodels OLS result
    explanation += f"Furthermore, an OLS regression model gives a p-value of {p_value_gender:.3f} for the 'female' coefficient. "
    if p_value_gender < 0.05:
        explanation += "This confirms the statistical significance of gender as a predictor of mortgage denial. "
    else:
        explanation += "This fails to confirm the statistical significance of gender as a predictor of mortgage denial. "

    # Final conclusion based on p-value
    if p_value_gender < 0.05:
        response = 80  # Strong "Yes"
        explanation += "Based on the significant p-value from the OLS model and supporting evidence from the t-test and logistic regression, there is a clear relationship between gender and mortgage approval."
    else:
        response = 20  # Strong "No"
        explanation += "Based on the non-significant p-value from the OLS model and the t-test, there is not enough evidence to conclude that gender has a statistically significant effect on mortgage approval."

    # Write conclusion to file
    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    analyze()
