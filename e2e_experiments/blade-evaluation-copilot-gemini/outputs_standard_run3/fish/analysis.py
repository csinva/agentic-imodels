
import pandas as pd
import json
import statsmodels.api as sm

def analyze_fish_data():
    """
    Analyzes the fish dataset to answer the research question.
    """
    # Load the dataset
    try:
        df = pd.read_csv('fish.csv')
    except FileNotFoundError:
        print("Error: fish.csv not found. Make sure the data file is in the same directory.")
        return

    # Feature Engineering: Calculate fish caught per hour
    # To avoid division by zero, we'll replace 0 hours with a small number
    df['hours'] = df['hours'].replace(0, 0.001)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Address the first part of the research question: "How many fish on average do visitors take per hour, when fishing?"
    average_fish_per_hour = df['fish_per_hour'].mean()

    # Address the second part: "What factors influence the number of fish caught by visitors to a national park?"
    # We will use a regression model to determine the influence of different factors.
    # For this, we will predict 'fish_caught' based on other variables.
    
    # Define independent variables (X) and the dependent variable (y)
    X = df[['livebait', 'camper', 'persons', 'child', 'hours']]
    y = df['fish_caught']
    
    # Add a constant to the independent variables matrix for the intercept
    X = sm.add_constant(X)
    
    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()
    
    # Interpret the results
    # We will look at the p-values to determine which factors are statistically significant.
    # A p-value less than 0.05 is generally considered statistically significant.
    p_values = model.pvalues
    significant_factors = p_values[p_values < 0.05].index.tolist()

    # Formulate the explanation
    explanation = f"The average number of fish caught per hour is {average_fish_per_hour:.2f}. "
    explanation += "To understand the factors influencing the number of fish caught, a multiple regression analysis was performed. "
    
    if 'const' in significant_factors:
        significant_factors.remove('const')

    if significant_factors:
        explanation += f"The statistically significant factors (p < 0.05) are: {', '.join(significant_factors)}. "
        explanation += "This means that changes in these factors are associated with changes in the number of fish caught. "
    else:
        explanation += "No factors were found to be statistically significant in influencing the number of fish caught. "

    # Based on the analysis, provide a response on the Likert scale.
    # If there are significant factors, we can say "Yes" there is a relationship.
    response = 100 if significant_factors else 0

    # Create the conclusion dictionary
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    # Write the conclusion to a file
    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze_fish_data()
