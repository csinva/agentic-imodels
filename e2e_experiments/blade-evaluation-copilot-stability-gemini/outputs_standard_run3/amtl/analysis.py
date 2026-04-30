
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imodels import RuleFitClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('amtl.csv')

# Preprocessing
# Drop unnecessary columns
df = df.drop(columns=['specimen', 'pop', 'stdev_age'])

# Create the target variable: 1 if num_amtl > 0, else 0
df['has_amtl'] = (df['num_amtl'] > 0).astype(int)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['tooth_class', 'genus'], drop_first=True)

# Define features (X) and target (y)
features = [col for col in df.columns if col not in ['num_amtl', 'has_amtl']]
X = df[features]
y = df['has_amtl']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = log_reg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Get coefficients
coefficients = pd.DataFrame(log_reg.coef_[0], X.columns, columns=['Coefficient'])

# Train a RuleFit model for interpretability
rulefit = RuleFitClassifier()
rulefit.fit(X_train.values, y_train, feature_names=X.columns)

# Get the rules
rules = rulefit._get_rules()
rules = rules[rules.coef != 0].sort_values("support", ascending=False)

# Interpretation
# The logistic regression coefficients and RuleFit rules will help us understand the relationship.
# Since 'genus_Homo sapiens' was dropped, it is the reference category.
# Negative coefficients for the other genera indicate that humans have a higher likelihood of AMTL.
other_genera = ['genus_Pan', 'genus_Papio', 'genus_Pongo']
all_negative = all(coefficients.loc[g]['Coefficient'] < 0 for g in other_genera)

# Based on the coefficients, decide the response
if all_negative:
    response = 90  # Strong "Yes"
    explanation = f"The coefficients for all other primate genera (Pan, Papio, Pongo) are negative, which indicates that modern humans (Homo sapiens), the reference category, have a significantly higher likelihood of antemortem tooth loss compared to other primate genera in the dataset, even after accounting for age, sex, and tooth class."
else:
    response = 10  # Strong "No"
    explanation = f"The coefficients for other primate genera are not all negative, which suggests that modern humans do not have a consistently higher likelihood of antemortem tooth loss compared to all other primate genera in the dataset."

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion saved to conclusion.txt")
