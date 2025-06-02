from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Carga el modelo
model = pickle.load(open('savedmodel.sav', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    clases = model.classes_
    
    # Crear lista con clases y probabilidades para pasar al template
    probs = list(zip(clases, probabilities))
    
    return render_template('index.html', prediction=prediction, probs=probs)

