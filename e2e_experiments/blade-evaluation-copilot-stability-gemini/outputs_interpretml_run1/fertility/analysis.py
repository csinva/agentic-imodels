
import pandas as pd
import json
from scipy import stats
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('fertility.csv')

# Convert date columns to datetime objects
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    data[col] = pd.to_datetime(data[col], errors='coerce')

# Feature Engineering: Create a composite religiosity score
data['religiosity'] = data[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Feature Engineering: Estimate day in cycle
data['days_since_last_period'] = (data['DateTesting'] - data['StartDateofLastPeriod']).dt.days
data['estimated_day_in_cycle'] = data['days_since_last_period'] % data['ReportedCycleLength']

# Define fertile window (e.g., days 8-19 for a 28-day cycle)
# This is a simplification; a more accurate model would be more complex.
data['in_fertile_window'] = data['estimated_day_in_cycle'].apply(
    lambda x: 1 if 8 <= x <= 19 else 0
)

# Drop rows with missing values that are critical for the analysis
data.dropna(subset=['religiosity', 'in_fertile_window'], inplace=True)

# Split data for modeling
X = data[['in_fertile_window']]
y = (data['religiosity'] > data['religiosity'].median()).astype(int) # Binarize for classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modeling ---
# Using Explainable Boosting Machine (EBM)
ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# --- Statistical Analysis ---
fertile_group = data[data['in_fertile_window'] == 1]['religiosity']
non_fertile_group = data[data['in_fertile_window'] == 0]['religiosity']

# Perform t-test to compare the means of the two groups
t_stat, p_value = stats.ttest_ind(fertile_group, non_fertile_group, nan_policy='omit')

# --- Interpretation ---
explanation = (
    f"A t-test was conducted to compare religiosity scores between women in their fertile window "
    f"and those outside of it. The t-statistic is {t_stat:.2f} and the p-value is {p_value:.3f}. "
)

# Determine the response based on the p-value
if p_value < 0.05:
    explanation += "This suggests a statistically significant difference in religiosity. "
    # Check the direction of the effect
    if fertile_group.mean() > non_fertile_group.mean():
        explanation += "Women in their fertile window reported higher religiosity."
        response = 80  # Strong "Yes"
    else:
        explanation += "Women in their fertile window reported lower religiosity."
        response = 20 # Leaning "No"
else:
    explanation += "This suggests no statistically significant difference in religiosity between the groups."
    response = 10  # Strong "No"

# Save the conclusion
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
