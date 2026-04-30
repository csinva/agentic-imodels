
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor
from scipy.stats import ttest_ind

def analysis():
    # Load data
    data = pd.read_csv('reading.csv')
    with open('info.json', 'r') as f:
        info = json.load(f)

    # Prepare data for modeling
    # Calculate reading speed in words per minute
    data['reading_speed_wpm'] = (data['num_words'] / data['adjusted_running_time']) * 60000

    # Filter out extreme outliers in reading speed
    data = data[data['reading_speed_wpm'] < 2000]

    # Separate data for dyslexic and non-dyslexic readers
    dyslexic_data = data[data['dyslexia_bin'] == 1]
    
    # Perform t-test to compare reading speeds
    reader_view_speed = dyslexic_data[dyslexic_data['reader_view'] == 1]['reading_speed_wpm']
    no_reader_view_speed = dyslexic_data[dyslexic_data['reader_view'] == 0]['reading_speed_wpm']
    
    # Check for sufficient data
    if len(reader_view_speed) < 2 or len(no_reader_view_speed) < 2:
        response = 50
        explanation = "Not enough data to perform a conclusive analysis."
    else:
        ttest_result = ttest_ind(reader_view_speed, no_reader_view_speed, equal_var=False, nan_policy='omit')
        
        # Interpret results
        p_value = ttest_result.pvalue
        mean_diff = reader_view_speed.mean() - no_reader_view_speed.mean()

        if p_value < 0.05 and mean_diff > 0:
            response = 90
            explanation = f"Strong evidence that Reader View improves reading speed for individuals with dyslexia. The average reading speed with Reader View was {mean_diff:.2f} WPM higher, a statistically significant difference (p={p_value:.3f})."
        elif p_value < 0.05 and mean_diff <= 0:
            response = 10
            explanation = f"Strong evidence that Reader View does not improve (and may hinder) reading speed for individuals with dyslexia. The average reading speed with Reader View was {-mean_diff:.2f} WPM lower, a statistically significant difference (p={p_value:.3f})."
        else:
            response = 30
            explanation = f"No significant evidence that Reader View improves reading speed for individuals with dyslexia. The difference in reading speed was not statistically significant (p={p_value:.3f})."

    # Save conclusion
    with open('conclusion.txt', 'w') as f:
        json.dump({'response': response, 'explanation': explanation}, f)

if __name__ == '__main__':
    analysis()
