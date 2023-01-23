# Titanic Survival Prediction

An algorithm that determines whether you're likely to have survived the Titanic or not, based on parameters such as age, fare, numebr of dependents on board, and sex.

Try it interactively [here](https://darrendube.com/projects/titanic-dataset).

## Structure

- `app.py` is the main file. It is a Flask server that accepts a GET request containing the necessary parameters, and returns the appropriate response.

- `train.py` is the python program that trains the DecisionTreeClassifier on the data in `train.csv`. The DecisionTreeClassifier model object is then pickled into the `model.mdl` file. A StandardScaler is also trained on the data in `train.csv` and is then pickled into `scaler.mdl` to be used to scale the parameters input in `app.py`.
