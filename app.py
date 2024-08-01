from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

# Default values
default_values = {
    'DerogCnt': 2,
    'CollectCnt': 2,
    'BanruptcyInd': 1,
    'InqCnt06': 4,
    'InqTimeLast': 5,
    'InqFinanceCnt24': 6,
    'TLTimeFirst': 170,
    'TLTimeLast': 14,
    'TLCnt03': 1,
    'TLCnt12': 1,
    'TLCnt24': 4,
    'TLCnt': 8,
    'TLSum': 78000,
    'TLMaxSum': 31205,
    'TLSatCnt': 14,
    'TLDel60Cnt': 12,
    'TLBadCnt24': 1,
    'TL75UtilCnt': 3,
    'TL50UtilCnt': 4,
    'TLBalHCPct': 3,
    'TLSatPct': 0.5,
    'TLDel3060Cnt24': 1,
    'TLDel90Cnt24': 0,
    'TLDel60CntAll': 0,
    'TLOpenPct': 3,
    'TLBadDerogCnt': 2,
    'TLDel60Cnt24': 1,
    'TLOpen24Pct': 0
}

# Define your function to be run
def prediction(new_data):
    # Replace this with your actual script logic
    # result = f"Processed input: {new_data.to_dict(orient='records')[0]}"
        
    # Load the saved normalization coefficients and logistic regression model
    sc = joblib.load('models/Normalisation_CreditScoring')
    model = joblib.load('models/f1_Classifier_CreditScoring')

    # new_data.head()

    input_features = new_data.values
    # Standardize the input features using the loaded scaler
    input_features_scaled = sc.transform(input_features)

    # Get the predicted probabilities
    probabilities = model.predict_proba(input_features_scaled)


    print(probabilities)

    # Set the threshold to 80.4% for profit maximization
    threshold = 0.804

    # calclate good loan probability
    probability_percentage = probabilities[:, 0] * 100
    probability_value = probability_percentage.item()  # Extract the value from the array


    # Make the prediction based on the threshold
    prediction = f"Loan Approved ({probability_value:.2f}% Good Loan)" if (probabilities[:, 0] > threshold) else f'No loan ({probability_value:.2f}% Good Loan)'
    print(probability_value)

  
    # Print the prediction
    result = prediction



    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {key: float(request.form[key]) for key in default_values.keys()}
        new_data = pd.DataFrame([user_input])
        # result = 9000
        result = prediction(new_data)
        return render_template('index.html', default_values=user_input, result=result)
    return render_template('index.html', default_values=default_values, result=None)

if __name__ == '__main__':
    app.run(debug=True)
