
import pandas as pd
import json
import statsmodels.api as sm

# Load the dataset
try:
    df = pd.read_csv('caschools.csv')
except FileNotFoundError:
    print("Error: caschools.csv not found. Make sure the data file is in the same directory.")
    exit()

# --- Feature Engineering ---
# Create the student-teacher ratio
df['stu_teach_ratio'] = df['students'] / df['teachers']

# Create an academic performance score as the average of read and math scores
df['academic_performance'] = (df['read'] + df['math']) / 2

# --- Analysis ---
# The research question is: "Is a lower student-teacher ratio associated with higher academic performance?"
# This translates to a negative correlation between stu_teach_ratio and academic_performance.

# Define the variables for the regression model
Y = df['academic_performance']
# We expect a negative coefficient for stu_teach_ratio
X = df[['stu_teach_ratio', 'income', 'english', 'lunch']] 
X = sm.add_constant(X) # Add a constant (intercept) to the model

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(Y, X).fit()

# --- Interpretation ---
# Get the coefficient and p-value for the student-teacher ratio
stu_teach_ratio_coef = model.params['stu_teach_ratio']
stu_teach_ratio_pvalue = model.pvalues['stu_teach_ratio']

# Determine the response based on the results
# A lower student-teacher ratio is better, so we look for a negative coefficient.
# The relationship must be statistically significant (p-value < 0.05).
is_significant_and_negative = stu_teach_ratio_pvalue < 0.05 and stu_teach_ratio_coef < 0

if is_significant_and_negative:
    # The coefficient is negative and significant, so a lower ratio is associated with higher scores.
    # The magnitude of the coefficient indicates the strength.
    # Let's scale the response based on the coefficient's magnitude. A typical range for this kind of data might be -0.1 to -2.
    # We can create a simple linear scale. Let's say -2.0 is a very strong effect (100) and 0 is no effect (50).
    # A simple mapping could be response = 50 - coefficient * 25.
    # e.g., coef = -0.5 -> 50 - (-0.5 * 25) = 62.5
    # e.g., coef = -1.0 -> 50 - (-1.0 * 25) = 75
    # e.g., coef = -2.0 -> 50 - (-2.0 * 25) = 100
    response = min(100, 50 - stu_teach_ratio_coef * 20) # Cap at 100
    explanation = (
        f"Yes, there is a statistically significant negative association between the student-teacher ratio and academic performance (p-value: {stu_teach_ratio_pvalue:.4f}). "
        f"The regression coefficient for the student-teacher ratio is {stu_teach_ratio_coef:.4f}, indicating that for each one-unit increase in the student-teacher ratio, "
        "the academic performance score is expected to decrease by {-stu_teach_ratio_coef:.4f} points, holding other factors (income, English learners, free lunch eligibility) constant. "
        "This supports the hypothesis that a lower student-teacher ratio is associated with higher academic performance."
    )
else:
    response = 10 # A low score indicating no significant relationship in the expected direction.
    explanation = (
        f"No, the analysis does not support a significant negative association between the student-teacher ratio and academic performance. "
        f"The p-value for the student-teacher ratio coefficient is {stu_teach_ratio_pvalue:.4f}, which is not statistically significant at the 0.05 level. "
        f"The coefficient itself is {stu_teach_ratio_coef:.4f}. Therefore, we cannot conclude that a lower student-teacher ratio is associated with higher academic performance in this dataset when controlling for other socioeconomic factors."
    )


# --- Generate Conclusion File ---
conclusion = {
    "response": int(response),
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f, indent=2)

print("Analysis complete. conclusion.txt has been generated.")
print(model.summary())
