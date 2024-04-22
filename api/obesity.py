from flask import Blueprint, jsonify, Flask, request
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from flask_restful import Api, Resource
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import sqlite3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

app = Flask(__name__)
obesity_api = Blueprint('obesity_api', __name__, url_prefix='/api/obesity')
api = Api(obesity_api)

class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()
            obesity_data = pd.DataFrame(data, index=[0])
            obesity_data.head(5)
            
            # Preprocesssing
            #stroke_data['gender'] = stroke_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
            #passenger_data['alone'] = passenger_data['alone'].apply(lambda x: 1 if x == True else 0)

            #stroke_data.drop(['id', 'ever_married', 'work_type'], axis=1, inplace=True)

            ## dropping all NA values in dataset
            obesity_data.dropna(inplace=True)
            ## convert all sex values to 0/1 (ML models can only process quantitative data)
            obesity_data['gender'] = obesity_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
            #stroke_data['heart_disease'] = stroke_data['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)
            obesity_data['physical_activity'] = obesity_data['physical_activity'].apply(lambda x: 1 if x == '4' else 0)
            obesity_data['obesity_category'] = obesity_data['obesity_category'].apply(lambda x: 1 if x == 'obese' else 0)

            #onehot = enc.transform(passenger_data[['embarked']]).toarray()
            #cols = ['embarked_' + val for val in enc.categories_[0]]
            #passenger_data[cols] = pd.DataFrame(onehot)
            #passenger_data.drop(['embarked'], axis=1, inplace=True)
            
            # Predict the survival probability for the new passenger using a gaussian naive bayes model
            obesity_prob = gnb.predict_proba(obesity_data)[:, 1]
            #stroke_prob = 1 - survival_prob

            ## returns a percentage of the chance of stroke for each individual
            return {'Chance of Stroke': float(obesity_prob * 100)}, 200
        except Exception as e:
            return {'error': str(e)}, 400

## api endpoint
api.add_resource(Predict, '/predict')

# Load the stroke dataset
url='https://drive.google.com/file/d/1JeYYtlWiW_emr0uC4whuNgiwBJc8ky9K/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
obesity_data = pd.read_csv(url)

# Preprocess the data
## dropping the columns of the dataframe for the id, marriage, and work_type
#obesity_data.drop(['id', 'ever_married', 'work_type'], axis=1, inplace=True)

## dropping all NA values in dataset
obesity_data.dropna(inplace=True)

## convert all sex values to 0/1 (ML models can only process quantitative data)
obesity_data['gender'] = obesity_data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
#stroke_data['heart_disease'] = stroke_data['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)
obesity_data['physical_activity'] = obesity_data['physical_activity'].apply(lambda x: 1 if x == '4' else 0)
obesity_data['obesity_category'] = obesity_data['obesity_category'].apply(lambda x: 1 if x == 'obese' else 0)

# Encode categorical variables

## onehotencode was not required for this data as there were only binary values for most variables
## enc = OneHotEncoder(handle_unknown='ignore')
## enc.fit(stroke_data[['embarked']])
## onehot = enc.transform(titanic_data[['embarked']]).toarray()
## cols = ['embarked_' + val for val in enc.categories_[0]]
## titanic_data[cols] = pd.DataFrame(onehot)
## titanic_data.drop(['embarked'], axis=1, inplace=True)
##titanic_data.dropna(inplace=True)

# Split the data into training and testing sets
X = obesity_data.drop('obesity', axis=1)
y = obesity_data['obesity']

# Train a decision tree classifier
#dt = DecisionTreeClassifier()
#dt.fit(X_train, y_train)

# Test the model
#y_pred = dt.predict(X_test)

## slightly lower accuracies
# Split the data into features and target
X = obesity_data.drop('obesity', axis=1)
y = obesity_data['obesity']

# Train the logistic regression model
#logreg = LogisticRegression()
#regr = svm.SVR()
#regr.fit(X, y)

## gaussian naive bayes was tested instead of the original model and it ended up having a slightly lower accuracy
gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)

#accuracy = accuracy_score(y_test, y_pred)
#print('Accuracy:', accuracy)

#logreg.fit(X, y)

