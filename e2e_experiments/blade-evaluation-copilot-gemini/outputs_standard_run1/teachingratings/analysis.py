
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def analyze_data():
    """
    Analyzes the teaching ratings dataset to determine the impact of beauty on teaching evaluations.
    """
    # Load the dataset
    df = pd.read_csv('teachingratings.csv')

    # Extract the relevant columns
    beauty = df['beauty']
    evaluation = df['eval']

    # Perform a linear regression
    model = LinearRegression()
    model.fit(beauty.values.reshape(-1, 1), evaluation)

    # Get the coefficient and p-value
    coefficient = model.coef_[0]
    _, p_value = pearsonr(beauty, evaluation)

    # Determine the response based on the p-value
    if p_value < 0.05 and coefficient > 0:
        response = 85  # Strong "Yes"
        explanation = f"A statistically significant positive correlation (p={p_value:.3f}) was found between beauty and teaching evaluations. The linear regression coefficient of {coefficient:.3f} indicates that for every one-unit increase in beauty rating, the teaching evaluation score increases by {coefficient:.3f} points, suggesting that more attractive teachers tend to receive higher evaluations."
    else:
        response = 15  # Strong "No"
        explanation = f"No statistically significant correlation (p={p_value:.3f}) was found between beauty and teaching evaluations. The linear regression coefficient was {coefficient:.3f}, which is not statistically significant at the p < 0.05 level."

    # Create the conclusion dictionary
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    # Write the conclusion to a file
    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f, indent=4)

if __name__ == '__main__':
    analyze_data()
