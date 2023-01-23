# Titanic Survival Prediction

An algorithm that determines whether you're likely to have survived the Titanic or not to an accuracy of 77%, based on parameters such as age, fare, number of dependents on board, and sex.

Try it interactively [here](https://darrendube.com/projects/titanic-dataset).

## Structure

- `app.py` is the main file. It is a Flask server that accepts a GET request containing the necessary parameters, and returns the appropriate response.

- `train.py` is the python program that trains the DecisionTreeClassifier on the data in `train.csv`. The DecisionTreeClassifier model object is then pickled into the `model.mdl` file. A StandardScaler is also trained on the data in `train.csv` and is then pickled into `scaler.mdl` to be used to scale the parameters input in `app.py`.

## Use

To use this algorithm, there are two options:

- ### Interactive
  I've made a webpage that allows you to input the required parameters in a form, with the result output on submission of the form. Try it [here](https://darrendube.com/projects/titanic-dataset).
  
- ### GET request
  The flask server is hosted at https://darrendube.pythonanywhere.com/. 
  
  The required parameters in the GET request are:
  
  - **age** - `int`
  - **sex** - `string` ('female' or 'male')
  - **fare** - `int` (either 13, 20, or 83 - approximately $1300, $2000, or $8300 in today's US$)
  - **parents** - `int` (number of parents on board)
  - **siblings** - `int` (number of siblings on board)
  - **spouse** - `int` (0 = spouse is not on board. 1 = spouse is on board)
  - **children** - `int` (number of children on board)
  - **title** - `string ('Mr', 'Mrs', 'Miss', or 'Master')

  An example of a request would be https://darrendube.pythonanywhere.com/?age=36&sex=female&fare=83&parents=0&siblings=0&spouse=0&children=0&title=Mrs. 
  
  The server either returns:
  - `[0]` - likely **would not** have survived, or
  - `[1]` - likely **would** have survived
