
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def analyze_fish_data():
    # Load the dataset
    try:
        data = pd.read_csv('fish.csv')
    except FileNotFoundError:
        # Adjust path for local testing if necessary
        data = pd.read_csv('e2e_experiments/blade-evaluation-copilot-stability-gemini/outputs_interpretml_run2/fish/fish.csv')


    # The research question is about the rate of fish caught per hour.
    # So, we should create a new feature for this rate.
    # Avoid division by zero for cases where hours is 0
    data['fish_per_hour'] = data['fish_caught'] / data['hours'].replace(0, 0.001)

    # The question asks for factors influencing the number of fish caught.
    # Let's use a linear model to investigate this.
    # We will predict 'fish_caught' using other variables.

    # Define features (X) and target (y)
    features = ['livebait', 'camper', 'persons', 'child', 'hours']
    X = data[features]
    y = data['fish_caught']

    # Add a constant for the intercept term
    X = sm.add_constant(X)

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()

    # The average rate of fish caught per hour
    average_rate = data['fish_per_hour'].mean()

    # Check the p-values from the model summary to see which factors are significant.
    p_values = model.pvalues
    significant_factors = p_values[p_values < 0.05].index.tolist()

    # Formulate the explanation
    explanation = f"The average rate of fish caught per hour is approximately {average_rate:.2f}. "
    explanation += "To determine the factors influencing the number of fish caught, a linear regression model was fitted. "
    
    if 'const' in significant_factors:
        significant_factors.remove('const')

    if significant_factors:
        explanation += f"The statistically significant factors (p < 0.05) are: {', '.join(significant_factors)}. "
        explanation += "This suggests that these factors have a meaningful impact on the number of fish caught."
        response = 80 # Strong "Yes" as we found significant factors
    else:
        explanation += "No statistically significant factors were found to influence the number of fish caught. "
        explanation += "This suggests that within this dataset, the variations in fish caught cannot be reliably explained by the recorded factors."
        response = 20 # Leaning "No"

    # Create the conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze_fish_data()
