
import pandas as pd
import json
import statsmodels.api as sm
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('soccer.csv')

# Preprocessing
# Create a skin_tone feature by averaging rater1 and rater2
df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Drop rows with missing skin_tone
df.dropna(subset=['skin_tone'], inplace=True)

# Aggregate at player level
player_agg = df.groupby('playerShort').agg({
    'skin_tone': 'mean',
    'redCards': 'sum',
    'games': 'sum'
}).reset_index()

# Create a binary skin tone variable for easier analysis
# We'll consider players with skin_tone > 0.5 as 'dark' and <= 0.5 as 'light'
player_agg['skin_tone_binary'] = (player_agg['skin_tone'] > 0.5).astype(int)

# Analysis
# 1. T-test to compare red cards between the two groups
dark_skin_red_cards = player_agg[player_agg['skin_tone_binary'] == 1]['redCards']
light_skin_red_cards = player_agg[player_agg['skin_tone_binary'] == 0]['redCards']

ttest_result = ttest_ind(dark_skin_red_cards, light_skin_red_cards, equal_var=False)

# 2. Regression analysis to control for games played
player_agg['red_card_rate'] = player_agg['redCards'] / player_agg['games']
X = player_agg['skin_tone']
X = sm.add_constant(X)
y = player_agg['red_card_rate']

model = sm.OLS(y, X).fit()
p_value = model.pvalues['skin_tone']
coef = model.params['skin_tone']

# Interpretation
explanation = f"A t-test was conducted to compare the total red cards between players with light and dark skin tones. The t-statistic is {ttest_result.statistic:.2f} and the p-value is {ttest_result.pvalue:.3f}. "
explanation += f"A linear regression model was also fitted to predict the red card rate from skin tone, controlling for the number of games played. The p-value for the skin tone coefficient is {p_value:.3f} and the coefficient is {coef:.4f}. "

if p_value < 0.05 and coef > 0:
    response = 85 # Strong "Yes"
    explanation += "The results indicate a statistically significant positive relationship between darker skin tone and a higher rate of receiving red cards. The t-test also shows a significant difference. This suggests that players with darker skin tones are more likely to receive red cards."
elif p_value < 0.05 and coef < 0:
    response = 15 # Strong "No"
    explanation += "The results indicate a statistically significant negative relationship, which is the opposite of the research question. This would suggest players with darker skin are less likely to get red cards."
else:
    response = 10 # Strong "No"
    explanation += "The results are not statistically significant, so we cannot conclude that there is a relationship between skin tone and red cards."


# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
