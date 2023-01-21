import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

import pickle

data = pd.read_csv('train.csv')

data['Title'] = data.Name.str.extract('([A-Za-z]+)\.')

data.Title.replace(['Don','Rev','Dr','Mme','Ms','Major','Lady','Sir','Mlle','Col','Capt','Countess','Jonkheer'], ['Mr','Mr','Mr','Mrs','Miss','Mr','Mrs','Mr','Miss','Mr','Mr','Mrs','Mr'], inplace=True)

data['FamilySize'] = data.SibSp + data.Parch + 1

data.Age.fillna(data.Age.median(), inplace=True)

scaler = StandardScaler().fit(data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']])

newdata = pd.concat([pd.DataFrame(scaler.transform(data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']]), columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']), data[['PassengerId','Survived','Name','Sex','Ticket','Embarked','Title']]], axis=1)

newdata.Sex.replace({'male': 0, 'female':1}, inplace=True)


newdata.Title.replace(
    {
        'Mr': 0,
        'Mrs': 1,
        'Miss': 2,
        'Master': 3,

    }, inplace=True
)

newdata.set_index('PassengerId', inplace=True)

newdata.drop(['Name', 'Ticket', 'Pclass','Parch', 'SibSp', 'Embarked'], axis=1, inplace=True)

x = newdata.drop(['Survived'], axis=1)

y = newdata['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=2)

model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, criterion='entropy')

model.fit(x_train, y_train)

with open('model.mdl', 'wb') as f:
    pickle.dump(model,f)

with open('scaler.mdl','wb') as f:
    pickle.dump(scaler,f)


