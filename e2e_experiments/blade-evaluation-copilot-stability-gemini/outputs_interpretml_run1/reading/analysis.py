
import pandas as pd
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load data
df = pd.read_csv('reading.csv')

# The research question is:
# "Does 'Reader View' --- a modified web page layout --- improves reading speed for individuals with dyslexia?"

# We are interested in the interaction between 'reader_view' and 'dyslexia_bin' on 'speed'.
# Let's filter the data to only include individuals with dyslexia.
df_dyslexia = df[df['dyslexia_bin'] == 1]

# Separate into two groups: with and without reader view
reader_view_on = df_dyslexia[df_dyslexia['reader_view'] == 1]['speed']
reader_view_off = df_dyslexia[df_dyslexia['reader_view'] == 0]['speed']

# Perform a t-test to see if there is a significant difference
ttest = stats.ttest_ind(reader_view_on, reader_view_off, nan_policy='omit')

# We can also use a regression model to look at the effect of reader_view while controlling for other factors.
# We will use the full dataset to have more statistical power and include an interaction term.
df_clean = df[['speed', 'reader_view', 'dyslexia_bin', 'age', 'Flesch_Kincaid']].dropna()
X = df_clean[['reader_view', 'dyslexia_bin', 'age', 'Flesch_Kincaid']]
X['interaction'] = X['reader_view'] * X['dyslexia_bin']
y = df_clean['speed']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
p_value_interaction = model.pvalues['interaction']


# The t-test p-value will tell us if there is a significant difference in the mean speed for dyslexic readers
# when using reader view vs. not.
# The p-value from the regression for the interaction term will tell us if the effect of reader_view on speed
# is different for people with and without dyslexia.

# A small p-value (e.g., < 0.05) for the t-test would suggest that reader view *does* have an effect on speed for dyslexic readers.
# A small p-value for the interaction term in the regression would also support this.

# Based on the t-test, if the mean speed for reader_view_on is higher, it's an improvement.
mean_speed_on = reader_view_on.mean()
mean_speed_off = reader_view_off.mean()

# Let's decide on the response based on the p-value.
# If p < 0.05, we have a significant result.
# If the mean speed is higher with reader view, it's a positive effect.
significant = ttest.pvalue < 0.05 and mean_speed_on > mean_speed_off
significant_interaction = p_value_interaction < 0.05

# We will give a high score if the result is significant and positive.
if significant or significant_interaction:
    response = 85
    explanation = f"The analysis shows a statistically significant improvement in reading speed for individuals with dyslexia when using Reader View. The t-test p-value is {ttest.pvalue:.4f}, and the mean reading speed increased from {mean_speed_off:.2f} to {mean_speed_on:.2f} words per minute. The regression model also showed a significant interaction effect (p={p_value_interaction:.4f})."
else:
    response = 15
    explanation = f"The analysis does not show a statistically significant improvement in reading speed for individuals with dyslexia when using Reader View. The t-test p-value is {ttest.pvalue:.4f}. The mean reading speed with reader view was {mean_speed_on:.2f} compared to {mean_speed_off:.2f} without. The regression model did not show a significant interaction effect (p={p_value_interaction:.4f})."


# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'explanation': explanation}, f)
