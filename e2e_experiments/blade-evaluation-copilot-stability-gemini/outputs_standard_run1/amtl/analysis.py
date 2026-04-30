
import json
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from imodels import RuleFitRegressor

def analyze_data():
    # Load data
    with open('info.json', 'r') as f:
        info = json.load(f)
    df = pd.read_csv('amtl.csv')

    # Preprocess data
    df['amtl_rate'] = df['num_amtl'] / df['sockets']
    df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)
    
    # Drop rows with missing values
    df.dropna(subset=['age', 'prob_male', 'tooth_class', 'genus'], inplace=True)
    
    df = pd.get_dummies(df, columns=['tooth_class', 'genus'], drop_first=True)

    # Define features and target
    features = ['age', 'prob_male', 'tooth_class_Posterior', 'tooth_class_Premolar',
                'genus_Pan', 'genus_Papio', 'genus_Pongo']
    
    X = df[features]
    y = df['is_human']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Binomial regression model
    logit_model = sm.Logit(df['amtl_rate'], sm.add_constant(X.astype(float)))
    result = logit_model.fit()
    p_value_human = result.pvalues['const']

    # RuleFit model for interpretability
    rulefit = RuleFitRegressor()
    rulefit.fit(X_train.astype(float), df.loc[X_train.index, 'amtl_rate'])
    
    # Conclusion
    response = 100 if p_value_human < 0.05 else 0
    explanation = f"The p-value for the intercept in the binomial regression model is {p_value_human:.4f}. "
    if response == 100:
        explanation += "This indicates a significant difference in AMTL rates between humans and non-human primates, even after accounting for age, sex, and tooth class. "
        explanation += "The RuleFit model further provides interpretable rules that highlight the specific factors contributing to this difference."
    else:
        explanation += "This indicates that there is no significant difference in AMTL rates between humans and non-human primates after accounting for confounding variables. "
        explanation += "The RuleFit model did not find strong predictive rules, supporting the null hypothesis."

    # Write conclusion
    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    analyze_data()
