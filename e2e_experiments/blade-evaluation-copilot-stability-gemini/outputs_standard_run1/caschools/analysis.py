
import pandas as pd
import json
import statsmodels.api as sm

def analyze_data():
    # Load the dataset
    data = pd.read_csv('caschools.csv')

    # Feature Engineering
    data['student_teacher_ratio'] = data['students'] / data['teachers']
    data['avg_score'] = (data['read'] + data['math']) / 2

    # Define variables for the regression model
    # We want to see if student_teacher_ratio predicts avg_score,
    # controlling for other factors that might influence scores.
    X = data[['student_teacher_ratio', 'income', 'english', 'calworks', 'lunch']]
    y = data['avg_score']

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y, X).fit()

    # Get the coefficient and p-value for student_teacher_ratio
    str_coefficient = model.params['student_teacher_ratio']
    str_pvalue = model.pvalues['student_teacher_ratio']

    # Determine the response and explanation
    explanation = f"The analysis was conducted using an Ordinary Least Squares (OLS) regression to determine the association between student-teacher ratio and average academic performance (an average of math and reading scores). The model controlled for potential confounding variables: district average income, percentage of English learners, percentage of families on CalWorks, and percentage of students qualifying for free lunch. The coefficient for the student-teacher ratio was {str_coefficient:.4f} with a p-value of {str_pvalue:.4f}. "

    # A negative coefficient means as the ratio goes down, scores go up.
    # A p-value < 0.05 indicates statistical significance.
    if str_coefficient < 0 and str_pvalue < 0.05:
        response = 95 # Strong "Yes"
        explanation += "A statistically significant negative association was found. This indicates that a lower student-teacher ratio is associated with higher academic performance, even after accounting for socioeconomic factors. The strong significance and expected direction of the relationship warrant a high confidence score."
    elif str_coefficient < 0 and str_pvalue >= 0.05:
        response = 40 # Weak "No"
        explanation += "A negative association was observed, suggesting that a lower student-teacher ratio might be linked to higher scores. However, this result was not statistically significant (p >= 0.05), meaning we cannot confidently rule out that this association is due to random chance. Therefore, the evidence is not strong enough to confirm the hypothesis."
    else:
        response = 5 # Strong "No"
        explanation += f"The analysis did not find a statistically significant negative association. The coefficient was {str_coefficient:.4f}, which is not in the expected direction, and/or the p-value was not below 0.05. This suggests that, based on this dataset and model, a lower student-teacher ratio is not associated with higher academic performance."


    # Create the conclusion file
    conclusion = {
        "response": response,
        "explanation": explanation
    }

    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    analyze_data()
