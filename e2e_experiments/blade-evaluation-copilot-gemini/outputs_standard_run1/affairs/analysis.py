
import pandas as pd
import json
from scipy.stats import ttest_ind, chi2_contingency

def analyze_data():
    # Load the dataset
    try:
        data = pd.read_csv('affairs.csv')
    except FileNotFoundError:
        print("Error: affairs.csv not found. Make sure the data file is in the same directory.")
        return

    # Drop the 'rownames' column as it's just an identifier
    if 'rownames' in data.columns:
        data = data.drop('rownames', axis=1)

    # Create a binary 'has_affair' variable
    data['has_affair'] = (data['affairs'] > 0).astype(int)

    # Separate data into two groups: with and without children
    with_children = data[data['children'] == 'yes']['affairs']
    without_children = data[data['children'] == 'no']['affairs']

    # Perform an independent t-test
    ttest_result = ttest_ind(with_children, without_children, equal_var=False)

    # Create a contingency table for the chi-square test
    contingency_table = pd.crosstab(data['children'], data['has_affair'])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # The research question is: "Does having children decrease (if at all) the engagement in extramarital affairs?"
    
    explanation = f"T-test result: statistic={ttest_result.statistic:.3f}, p-value={ttest_result.pvalue:.3f}. "
    explanation += f"Chi-square test result: chi2={chi2:.3f}, p-value={p:.3f}. "

    # The t-test p-value is the most direct test of the hypothesis.
    # If the p-value is low, and the mean for the 'with_children' group is lower, then the answer is "yes".
    if ttest_result.pvalue < 0.05 and with_children.mean() < without_children.mean():
        response = 80  # Strong "Yes"
        explanation += "The t-test shows a statistically significant decrease in affairs for individuals with children."
    else:
        response = 20  # Leaning "No"
        explanation += "The t-test does not show a statistically significant decrease in affairs for individuals with children."


    # Write the conclusion to a file
    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    analyze_data()
