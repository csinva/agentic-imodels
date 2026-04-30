
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('amtl.csv')

# Create the response variable: proportion of teeth lost
data['amtl_prop'] = data['num_amtl'] / data['sockets']

# Create a binary variable for Homo sapiens
data['is_human'] = (data['genus'] == 'Homo sapiens').astype(int)

# Define features and target
features = ['is_human', 'age', 'prob_male', 'tooth_class']
target = 'amtl_prop'

# Separate numeric and categorical features
numeric_features = ['age', 'prob_male']
categorical_features = ['tooth_class']

# Create preprocessing pipelines for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Define the model
# We use a binomial regression model (logistic regression with weights)
# to account for the fact that the response is a proportion.
X = data[features]
y = data[target]

# The number of trials is the number of sockets
weights = data['sockets']

# Preprocess the features
X_processed = preprocessor.fit_transform(X)

# Get the feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_feature_names) + ['is_human']

# Convert the processed features back to a DataFrame for statsmodels
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
X_processed_df = sm.add_constant(X_processed_df)


# Fit the binomial regression model
glm_binom = sm.GLM(data['num_amtl'], X_processed_df, family=sm.families.Binomial(), var_weights=weights)
res = glm_binom.fit()

# Get the coefficient for 'is_human'
human_coeff = res.params['is_human']
p_value = res.pvalues['is_human']

# Interpret the results
explanation = f"The coefficient for 'is_human' is {human_coeff:.4f} with a p-value of {p_value:.4f}. "
if p_value < 0.05 and human_coeff > 0:
    explanation += "This indicates that modern humans have a statistically significant higher frequency of antemortem tooth loss compared to non-human primates, after controlling for age, sex, and tooth class."
    response = 95
elif p_value < 0.05 and human_coeff < 0:
    explanation += "This indicates that modern humans have a statistically significant lower frequency of antemortem tooth loss compared to non-human primates, after controlling for age, sex, and tooth class."
    response = 5
else:
    explanation += "There is no statistically significant difference in the frequency of antemortem tooth loss between modern humans and non-human primates, after controlling for age, sex, and tooth class."
    response = 10

# Create the conclusion file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Analysis complete. Conclusion written to conclusion.txt")
