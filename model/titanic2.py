# Organized Version using the initialize function
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# definew the class for data loading, preprocessing,training, evaluation, and prediction
class TitanicPredictor:
    def __init__(self):
        # initialize class variables
        # stores data
        self.data = None 
        self.encoder = None 
        self.model_dt = None 
        self.model_logreg = None 
        self.X_test = None
        self.y_test = None 
        
    def load_data(self):
        # loading titanic dataset
        self.data = sns.load_dataset('titanic')

    def preprocess_data(self):
        # checking data has been loaded
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # dropping columns
        self.data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        
        # removing rows with missing values
        self.data.dropna(inplace=True)
        
        # converting categorical features to binary values
        self.data['sex'] = self.data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        self.data['alone'] = self.data['alone'].apply(lambda x: 1 if x == True else 0)
        
        # creating a OneHotEncoder instance for categorical features
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(self.data[['embarked']])
        
        # applying encoder to 'embarked'and adding encoded columns
        onehot = self.encoder.transform(self.data[['embarked']]).toarray()
        cols = ['embarked_' + val for val in self.encoder.categories_[0]]
        self.data[cols] = pd.DataFrame(onehot)
        
        # dropping 'embarked' column and rows with missing values
        self.data.drop(['embarked'], axis=1, inplace=True)
        self.data.dropna(inplace=True)

    def train_models(self):
        # separating features x and y
        X = self.data.drop('survived', axis=1)
        y = self.data['survived']
        
        # splitting data into training and testing
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # training decision tree classifier
        self.model_dt = DecisionTreeClassifier()
        self.model_dt.fit(X_train, y_train)
        
        # training a logistic regression model
        self.model_logreg = LogisticRegression()
        self.model_logreg.fit(X_train, y_train)

    def evaluate_models(self):
        # checking if models were trained
        if self.model_dt is None or self.model_logreg is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # evaluating decision tree classifier
        y_pred_dt = self.model_dt.predict(self.X_test)
        accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
        print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy_dt)) 
        
        # evaluating logistic regression model
        y_pred_logreg = self.model_logreg.predict(self.X_test)
        accuracy_logreg = accuracy_score(self.y_test, y_pred_logreg)
        print('LogisticRegression Accuracy: {:.2%}'.format(accuracy_logreg)) 
  
    def predict_survival_probability(self, new_passenger):
        # checking if models have been trained
        if self.model_logreg is None:
            raise ValueError("Models not trained. Call train_models() first.")
      
        # converting categorical features to binary values
        new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
        new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)
        
        # applying encoder to the new passenger's 'embarked' feature and add the encoded columns
        onehot = self.encoder.transform(new_passenger[['embarked']]).toarray()
        cols = ['embarked_' + val for val in self.encoder.categories_[0]]
        new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
        
        # dropping'embarked' and 'name' columns from the new data
        new_passenger.drop(['embarked', 'name'], axis=1, inplace=True)
        
        # predicting the survival probability with log regression model
        dead_proba, alive_proba = np.squeeze(self.model_logreg.predict_proba(new_passenger))
        print('Death probability: {:.2%}'.format(dead_proba)) 
        print('Survival probability: {:.2%}'.format(alive_proba))
        
        # returns death and survival probabilities
        return dead_proba, alive_proba


# Usage example
# Create an instance of TitanicPredictor
titanic_predictor = TitanicPredictor()

# Define a function to initialize and run the predictor
def initTitanic():
    # Load data, preprocess, train models, and evaluate models
    titanic_predictor.load_data()
    titanic_predictor.preprocess_data()
    titanic_predictor.train_models()
    titanic_predictor.evaluate_models()
    
    # Define a new passenger as a pandas DataFrame
    passenger = pd.DataFrame({
        'name': ['John Mortensen'],
        'pclass': [2],
        'sex': ['male'],
        'age': [64],
        'sibsp': [1],
        'parch': [1],
        'fare': [16.00],
        'embarked': ['S'],
        'alone': [False]
    })
    
    # Predict survival probability for the new passenger
    return titanic_predictor.predict_survival_probability(passenger)
