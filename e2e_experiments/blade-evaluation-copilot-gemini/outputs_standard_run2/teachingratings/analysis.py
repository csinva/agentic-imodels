
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def analyze_data():
    """
    This function loads the teaching ratings dataset,
    analyzes the relationship between beauty and teaching evaluations,
    and saves the conclusion to a file.
    """
    # Load the dataset
    data = pd.read_csv("teachingratings.csv")

    # Research question: What is the impact of beauty on teaching evaluations?
    # To answer this, we will perform a correlation and a simple linear regression.

    # 1. Correlation
    corr, p_value = pearsonr(data['beauty'], data['eval'])

    # 2. Linear Regression
    X = data[['beauty']]
    y = data['eval']
    model = LinearRegression()
    model.fit(X, y)
    coefficient = model.coef_[0]

    # Interpretation
    explanation = f"The correlation between beauty and teaching evaluation is {corr:.3f} (p-value: {p_value:.3f}). The linear regression coefficient is {coefficient:.3f}, suggesting a positive relationship. For every one-unit increase in the beauty score, the teaching evaluation score increases by {coefficient:.3f} on average."

    # Based on the statistical significance, we can determine the response.
    # A low p-value (e.g., < 0.05) indicates a significant relationship.
    if p_value < 0.05:
        # Strong "Yes" if the correlation is moderately strong
        if abs(corr) > 0.3:
            response = 90
        else:
            response = 75
    else:
        response = 10

    # Save the conclusion
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f, indent=2)

if __name__ == "__main__":
    analyze_data()
