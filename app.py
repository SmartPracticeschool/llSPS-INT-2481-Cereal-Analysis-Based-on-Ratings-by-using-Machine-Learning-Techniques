import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('strength.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    X_test = [[int(X) for X in request.form.values()]]
    
    prediction = model.predict(X_test)
    print(prediction)
    output=prediction[0][0]
    return render_template('index.html', 
  prediction_text=
  'Rating = '.format(output))
if __name__ == "__main__":
    app.run(debug=True)
