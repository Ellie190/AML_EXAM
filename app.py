import json
from flask import Flask, request

import numpy as np
import pandas as pd
import pickle
import pymc3 as pm
from sklearn.linear_model import LogisticRegression

# Loading Simple Logistic Regression Model
lr_model = pickle.load(open('./LR_model.pkl', 'rb'))

# Loading Probabilistic Model
with open('./potability_model.pkl', 'rb') as buff:
    PMLmodel_data = pickle.load(buff)

# Obtaining Model and Model trace from loaded Probabilistic Model
PML_loadModel1, PML1_trace = PMLmodel_data['model'], PMLmodel_data['trace']

app = Flask(__name__)


@app.route('/predict', methods=['Post'])
def predict():
    X = json.loads(request.data)
    feat = list(map(np.float, X['xv']))
    feat = np.array(feat).reshape(1, -1)
    model = pickle.load(open('./LR_model.sav', 'rb'))
    pred = model.pred(feat)
    return str(pred[0])


if __name__ == '__main__':
    app.run(debug=True)

# {% if pred==1%}
# <p>Potable</p>
# {% else %}
# <p>Not Potable</p>
# {% endif%}
