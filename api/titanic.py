## Python Titanic Sample API endpoint
from flask import Blueprint, request, jsonify
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from flask_restful import Api, Resource # used for REST API building

# Import the TitanicModel class from the model file
# from model.titanic import TitanicModel

titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic')


class TitanicModel:
    """A class used to represent the Titanic Model for passenger survival prediction.
    """
    # a singleton instance of TitanicModel, created to train the model only once, while using it for prediction multiple times
    ## underbar in Python means that it is not for general use - you need to use another accessor to get to it (it will be assigned something if you use the system appropriately)
    _instance = None


    ## creating + cleaning + training of the instance
    
    # constructor, used to initialize the TitanicModel
    def __init__(self):
        # the titanic ML model
        self.model = None
        self.dt = None
        # define ML features and target
        self.features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'alone']
        self.target = 'survived'
        # load the titanic dataset
        self.titanic_data = sns.load_dataset('titanic')
        # one-hot encoder used to encode 'embarked' column
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    # clean the titanic dataset, prepare it for training
    def _clean(self):
        # Drop unnecessary columns
        self.titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)

        # Convert boolean columns to integers
        self.titanic_data['sex'] = self.titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.titanic_data['alone'] = self.titanic_data['alone'].apply(lambda x: 1 if x == True else 0)

        # Drop rows with missing 'embarked' values before one-hot encoding
        self.titanic_data.dropna(subset=['embarked'], inplace=True)
        
        # One-hot encode 'embarked' column
        onehot = self.encoder.fit_transform(self.titanic_data[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.titanic_data = pd.concat([self.titanic_data, onehot_df], axis=1)
        self.titanic_data.drop(['embarked'], axis=1, inplace=True)

        # Add the one-hot encoded 'embarked' features to the features list
        self.features.extend(cols)
        
        # Drop rows with missing values
        self.titanic_data.dropna(inplace=True)

    # train the titanic model, using logistic regression as key model, and decision tree to show feature importance
    def _train(self):
        # split the data into features and target
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]
        
        # perform train-test split
        self.model = LogisticRegression(max_iter=1000)
        
        # train the model
        self.model.fit(X, y)
        
        # train a decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
        
    @classmethod
    def get_instance(cls):
        """ Gets, and conditionaly cleans and builds, the singleton instance of the TitanicModel.
        The model is used for analysis on titanic data and predictions on the survival of theoritical passengers.
        
        Returns:
            TitanicModel: the singleton _instance of the TitanicModel, which contains data and methods for prediction.
        """        
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        # return the instance, to be used for prediction
        return cls._instance

    def predict(self, passenger):
        """ Predict the survival probability of a passenger.

        Args:
            passenger (dict): A dictionary representing a passenger. The dictionary should contain the following keys:
                'pclass': The passenger's class (1, 2, or 3)
                'sex': The passenger's sex ('male' or 'female')
                'age': The passenger's age
                'sibsp': The number of siblings/spouses the passenger has aboard
                'parch': The number of parents/children the passenger has aboard
                'fare': The fare the passenger paid
                'embarked': The port at which the passenger embarked ('C', 'Q', or 'S')
                'alone': Whether the passenger is alone (True or False)

        Returns:
           dictionary : contains die and survive probabilities 
        """
        # clean the passenger data
        passenger_df = pd.DataFrame(passenger, index=[0])
        passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
        passenger_df['alone'] = passenger_df['alone'].apply(lambda x: 1 if x == True else 0)
        onehot = self.encoder.transform(passenger_df[['embarked']]).toarray()
        cols = ['embarked_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        passenger_df = pd.concat([passenger_df, onehot_df], axis=1)
        passenger_df.drop(['embarked'], axis=1, inplace=True)
        
        # predict the survival probability and extract the probabilities from numpy array
        die, survive = np.squeeze(self.model.predict_proba(passenger_df))
        # return the survival probabilities as a dictionary
        return {'die': die, 'survive': survive}
    
    def feature_weights(self):
        """Get the feature weights
        The weights represent the relative importance of each feature in the prediction model.

        Returns:
            dictionary: contains each feature as a key and its weight of importance as a value
        """
        # extract the feature importances from the decision tree model
        importances = self.dt.feature_importances_
        # return the feature importances as a dictionary, using dictionary comprehension
        return {feature: importance for feature, importance in zip(self.features, importances)} 

api = Api(titanic_api)
class TitanicAPI:
    class _Predict(Resource):
        
        def post(self):
            """ Semantics: In HTTP, POST requests are used to send data to the server for processing.
            Sending passenger data to the server to get a prediction fits the semantics of a POST request.
            
            POST requests send data in the body of the request...
            1. which can handle much larger amounts of data and data types, than URL parameters
            2. using an HTTPS request, the data is encrypted, making it more secure
            3. a JSON formated body is easy to read and write between JavaScript and Python, great for Postman testing
            """     
            # Get the passenger data from the request
            passenger = request.get_json()

            # Get the singleton instance of the TitanicModel
            titanicModel = TitanicModel.get_instance()
            # Predict the survival probability of the passenger
            response = titanicModel.predict(passenger)

            # Return the response as JSON
            return jsonify(response)

    api.add_resource(_Predict, '/predict')
