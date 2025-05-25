
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and columns
with open('model/accident_model.pkl', 'rb') as f:
    model, model_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(df)[0]
        return render_template('index.html', prediction=f'Predicted Severity: {prediction}')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
