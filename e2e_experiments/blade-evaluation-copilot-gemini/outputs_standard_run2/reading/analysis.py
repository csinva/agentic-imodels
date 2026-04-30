
import pandas as pd
from scipy import stats
import json

# Load the dataset
df = pd.read_csv('reading.csv')

# The research question is: "Does 'Reader View' --- a modified web page layout --- improve reading speed for individuals with dyslexia?"
# We need to compare the reading speed of individuals with dyslexia when using Reader View versus not using it.

# Filter the data for individuals with dyslexia
dyslexia_df = df[df['dyslexia_bin'] == 1]

# Separate the data into two groups: with and without Reader View
reader_view_on = dyslexia_df[dyslexia_df['reader_view'] == 1]['speed']
reader_view_off = dyslexia_df[dyslexia_df['reader_view'] == 0]['speed']

# Perform an independent t-test to compare the means of the two groups
# The null hypothesis is that the means of the two groups are equal.
# The alternative hypothesis is that the mean reading speed with Reader View is greater than without.
ttest_result = stats.ttest_ind(reader_view_on, reader_view_off, equal_var=False, alternative='greater')

# The t-test returns a p-value. A small p-value (typically < 0.05) indicates that we can reject the null hypothesis.
# If the p-value is small, it suggests that there is a statistically significant difference between the two groups.
# In our case, a small p-value would mean that Reader View does improve reading speed for individuals with dyslexia.

# Let's determine the response based on the p-value.
# If p-value is less than 0.05, we have strong evidence to say "Yes".
# If p-value is greater than 0.05, we don't have enough evidence, so we say "No".

# We can map the p-value to a 0-100 scale.
# A common way to do this is to use 1 - p-value, but this is not a direct mapping to confidence.
# A more direct interpretation is to set a threshold.
# If p < 0.01, very strong "Yes" (e.g., 95-100)
# If 0.01 <= p < 0.05, strong "Yes" (e.g., 80-95)
# If 0.05 <= p < 0.1, weak "Yes" (e.g., 60-80)
# If p >= 0.1, "No" (e.g., 0-50)

p_value = ttest_result.pvalue
if p_value < 0.01:
    response = 95
    explanation = f"There is very strong evidence (p={p_value:.4f}) that Reader View improves reading speed for individuals with dyslexia. The average reading speed with Reader View was significantly higher than without it."
elif p_value < 0.05:
    response = 85
    explanation = f"There is strong evidence (p={p_value:.4f}) that Reader View improves reading speed for individuals with dyslexia. The average reading speed with Reader View was higher than without it."
else:
    response = 10
    explanation = f"There is not enough statistical evidence (p={p_value:.4f}) to conclude that Reader View improves reading speed for individuals with dyslexia. The difference in reading speeds was not statistically significant."


# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
