
import pandas as pd
import json
from scipy.stats import ttest_ind
import statsmodels.api as sm

# Load the data
df = pd.read_csv('soccer.csv')

# The research question is: "Are soccer players with a dark skin tone more likely than those with a light skin tone to receive red cards from referees?"

# Preprocessing
# Create a binary skin tone variable. We'll define "light skin" as the average of rater1 and rater2 being less than 0.5, and "dark skin" as 0.5 or greater.
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2
df.dropna(subset=['skin_tone'], inplace=True)
df['skin_tone_binary'] = df['skin_tone'].apply(lambda x: 'dark' if x >= 0.5 else 'light')

# Separate the two groups
dark_skin_players = df[df['skin_tone_binary'] == 'dark']
light_skin_players = df[df['skin_tone_binary'] == 'light']

# Calculate red card rates
dark_skin_red_card_rate = dark_skin_players['redCards'].sum() / dark_skin_players['games'].sum()
light_skin_red_card_rate = light_skin_players['redCards'].sum() / light_skin_players['games'].sum()

# Perform an independent t-test on the number of red cards
# We are comparing the means of two independent groups
t_stat, p_value = ttest_ind(dark_skin_players['redCards'], light_skin_players['redCards'], equal_var=False)

# To get a better understanding, let's run a regression model.
# We want to predict red cards based on skin tone, controlling for other factors.
df['dark_skin'] = (df['skin_tone_binary'] == 'dark').astype(int)
X = df[['dark_skin', 'games', 'victories', 'defeats', 'goals', 'yellowCards', 'yellowReds']]
X = sm.add_constant(X)
y = df['redCards']

model = sm.OLS(y, X).fit()
p_value_regression = model.pvalues['dark_skin']


# Interpretation
explanation = f"The red card rate for dark-skinned players is {dark_skin_red_card_rate:.4f}, while for light-skinned players it is {light_skin_red_card_rate:.4f}. "
explanation += f"An independent t-test resulted in a p-value of {p_value:.4f}. "
explanation += f"A regression model controlling for games played, victories, defeats, goals, and other cards gives a p-value of {p_value_regression:.4f} for the dark_skin coefficient. "

if p_value < 0.05 and p_value_regression < 0.05:
    response = 90  # Strong "Yes"
    explanation += "Both the t-test and the regression analysis show a statistically significant relationship between skin tone and red cards, suggesting that dark-skinned players are more likely to receive red cards."
elif p_value < 0.05 or p_value_regression < 0.05:
    response = 70 # "Yes"
    explanation += "One of the statistical tests (t-test or regression) shows a significant relationship, suggesting a tendency for dark-skinned players to receive more red cards."
else:
    response = 10  # Strong "No"
    explanation += "Neither the t-test nor the regression analysis shows a statistically significant relationship between skin tone and red cards. Therefore, we cannot conclude that dark-skinned players are more likely to receive red cards."


# Write the conclusion to a file
with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'explanation': explanation}, f)
