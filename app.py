from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)

# to allow all origins for Access-Control-Allow_Origin header in CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# importing all required ML libraries
import pandas as pd
import numpy as np
import pickle
import sklearn

# this variables will store the DecisionTreeClassifier model trained on the titanic dataset
model = None
scaler = None



def preprocess(df):
    dataframe = df





    dataframe = pd.concat([pd.DataFrame(scaler.transform(dataframe[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']]), columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']), dataframe[['Sex','Title']]], axis=1)

    dataframe.Sex.replace({'male': 0, 'female':1}, inplace=True)

    dataframe.Title.replace(
        {
            'Mr': 0,
            'Mrs': 1,
            'Miss': 2,
            'Master': 3,

        }, inplace=True
    )


    dataframe.drop(['Parch', 'SibSp', 'Pclass'], axis=1, inplace=True)

    return dataframe


with open('/home/darrendube/mysite/model.mdl','rb') as f:
    model = pickle.load(f)

with open('/home/darrendube/mysite/scaler.mdl','rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET'])
def hello_world():
    age = request.args.get('age', type=int)
    sex = request.args.get('sex', type=str)
    fare = request.args.get('fare', type=int)
    parents = request.args.get('parents', type=int)
    siblings = request.args.get('siblings', type=int)
    spouse = request.args.get('spouse', type=int)
    children = request.args.get('children', type=int)
    title = request.args.get('title', type=str)

    sibsp = siblings + spouse
    parch = parents+children
    family_size = sibsp+parch+ 1

    to_predict = pd.DataFrame([[1, age, fare, family_size, sex, title, sibsp, parch],], columns=['Pclass','Age', 'Fare', 'FamilySize', 'Sex', 'Title', 'SibSp', 'Parch'])
    to_predict = preprocess(to_predict)
    return str(model.predict(to_predict))



