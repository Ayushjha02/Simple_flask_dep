from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('feature_model.joblib')

# Load the saved LabelEncoder
le = joblib.load('label_encoder.joblib')

@app.route('/')
def home():
    return render_template('index.htm')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    x0 = request.form['X_0']
    x1 = float(request.form['X_1'])
    x2 = float(request.form['X_2'])

    # Encode X_0 using the saved LabelEncoder
    x0_encoded = le.transform([x0])[0]

    # Use the model for prediction
    prediction = model.predict([[x0_encoded, x1, x2]])

    return render_template('result.htm', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
