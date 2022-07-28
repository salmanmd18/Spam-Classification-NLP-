
import pickle
from tkinter.tix import Tree
from flask import Flask,request,app,jsonify,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('clamodel.pkl','rb'))
countvectorizer = pickle.load(open('count.pkl','rb'))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data  = [list(data.values())]
    final_data = countvectorizer.transform(new_data[0]).toarray()
    output = int(model.predict(final_data))
    if output == 1:
        text = "Spam message"
    else:
        text = "ham message"
    return jsonify(text,output)

if __name__ == "__main__":
    app.run(debug=True)
