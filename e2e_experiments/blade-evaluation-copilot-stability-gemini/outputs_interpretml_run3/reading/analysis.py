
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import ttest_ind

# Load the dataset
df = pd.read_csv('reading.csv')

# Separate data for dyslexic and non-dyslexic readers
dyslexic_df = df[df['dyslexia_bin'] == 1]
non_dyslexic_df = df[df['dyslexia_bin'] == 0]

# Separate by reader_view
dyslexic_reader_view_on = dyslexic_df[dyslexic_df['reader_view'] == 1]
dyslexic_reader_view_off = dyslexic_df[dyslexic_df['reader_view'] == 0]

# Perform t-test
# It is important to drop NA values for statistical tests
t_stat, p_value = ttest_ind(
    dyslexic_reader_view_on['speed'].dropna(),
    dyslexic_reader_view_off['speed'].dropna()
)

# EBC model
features = ['reader_view', 'num_words', 'Flesch_Kincaid', 'age', 'correct_rate']
X = dyslexic_df[features]
y = (dyslexic_df['speed'] > dyslexic_df['speed'].median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ebc = ExplainableBoostingClassifier(random_state=42)
ebc.fit(X_train, y_train)

# Interpret the results
explanation = "Based on the analysis, Reader View has a statistically significant impact on reading speed for individuals with dyslexia."
if p_value > 0.05:
    explanation = "Based on the analysis, there is no statistically significant evidence that Reader View improves reading speed for individuals with dyslexia."
    response = 20
else:
    # Check the direction of the effect
    if dyslexic_reader_view_on['speed'].mean() > dyslexic_reader_view_off['speed'].mean():
        explanation = "Reader View significantly improves reading speed for individuals with dyslexia."
        response = 90
    else:
        explanation = "Reader View significantly decreases reading speed for individuals with dyslexia."
        response = 10

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
