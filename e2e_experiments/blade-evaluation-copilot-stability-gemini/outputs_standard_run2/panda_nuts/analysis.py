
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from imodels import RuleFitRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('panda_nuts.csv')

# Preprocess the data
data['efficiency'] = data['nuts_opened'] / data['seconds']
data.dropna(subset=['efficiency'], inplace=True)
data = data[data['seconds'] > 0]

# Encode categorical variables
le_sex = LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])

le_help = LabelEncoder()
data['help'] = le_help.fit_transform(data['help'])

# Define features and target
features = ['age', 'sex', 'help']
target = 'efficiency'

X = data[features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RuleFitRegressor model
model = RuleFitRegressor()
model.fit(X_train, y_train)

# Get rules
rules = model._get_rules()
rules = rules[rules.coef != 0].sort_values("support", ascending=False)

# Determine the response based on feature importances
# If age is the most important feature, it strongly influences efficiency.
if 'age' in rules.iloc[0]['rule']:
    response = 80
    explanation = "Age is the most significant factor influencing nut-cracking efficiency. Older, more experienced chimpanzees are more efficient."
elif 'age' in rules['rule'].to_string():
    response = 60
    explanation = "Age has a notable influence on nut-cracking efficiency, though other factors also play a role."
else:
    response = 20
    explanation = "Age, sex, and receiving help do not appear to be primary drivers of nut-cracking efficiency based on the model."

# Save the conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
