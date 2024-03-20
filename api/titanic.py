from flask import Blueprint, request, jsonify
from flask_restful import Resource, Api
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from model.titanics import TitanicPredictor

titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')
api = Api(titanic_api)

# Initialize the TitanicPredictor instance
titanic_predictor = TitanicPredictor()

class TitanicAPI(Resource):
    def post(self):
        try:
            # Get passenger data from the API request
            data = request.get_json()  # get the data as JSON
            modified_data = pd.DataFrame(data)  # create DataFrame from JSON
            response = titanic_predictor.predict_survival_probability(modified_data)
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)})


# Add resource to the API
api.add_resource(TitanicAPI, '/create')