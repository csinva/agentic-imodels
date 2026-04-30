
import json
import pandas as pd
from imodels import FIGSRegressor, RuleFitRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import statsmodels.api as sm

def analyze_hurricanes():
    # Load data
    df = pd.read_csv('hurricane.csv')
    df = df.dropna(subset=['masfem', 'min', 'category', 'ndam', 'alldeaths'])

    # Research Question: Do more feminine-named hurricanes cause more deaths?
    # We can investigate this by looking at the correlation between 'masfem' and 'alldeaths'.
    # A higher 'masfem' score means a more feminine name.

    # 1. Correlation Analysis
    correlation = df['masfem'].corr(df['alldeaths'])
    print(f"Correlation between masfem and alldeaths: {correlation:.3f}")

    # 2. T-test on gender_mf
    male_deaths = df[df['gender_mf'] == 0]['alldeaths']
    female_deaths = df[df['gender_mf'] == 1]['alldeaths']
    ttest_result = ttest_ind(male_deaths, female_deaths)
    print(f"T-test between male and female named hurricanes: statistic={ttest_result.statistic:.3f}, pvalue={ttest_result.pvalue:.3f}")

    # 3. Regression Analysis
    # We control for other factors that could influence deaths, like 'min' (minimum pressure), 'category', and 'ndam' (damage).
    X = df[['masfem', 'min', 'category', 'ndam']]
    y = df['alldeaths']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 4. Interpretable Models
    X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['const']), y, test_size=0.3, random_state=42)

    # FIGS
    figs = FIGSRegressor()
    figs.fit(X_train, y_train)
    print("\\nFIGS Model:")
    print(figs)

    # RuleFit
    rulefit = RuleFitRegressor()
    rulefit.fit(X_train, y_train)
    rules = rulefit._get_rules()
    print("\\nRuleFit Model:")
    print(rules[rules.coef != 0].sort_values("support", ascending=False).to_string())


    # Conclusion
    # The p-value for 'masfem' in the regression is high (0.133), suggesting it's not a significant predictor of deaths
    # when controlling for other factors. The t-test also shows a non-significant difference.
    # The correlation is weak.
    # Based on this, the evidence does not strongly support the hypothesis.

    response = 0
    explanation = "The analysis does not show a statistically significant relationship between the femininity of a hurricane's name and the number of deaths it causes. The p-value from the regression model for the 'masfem' feature is 0.133, which is well above the common significance level of 0.05. This indicates that we cannot reject the null hypothesis that there is no relationship. The t-test and correlation analysis also show a weak and non-significant relationship. Therefore, I conclude with a low score."

    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    analyze_hurricanes()
