import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object          #to load our pickle files


class PredictPipeline:
    def __init__(self):     #in this init function, I don't want to initializse anything. This is an empty constructoor by default
        pass

    def predict(self, features):     #create the actual prediction function
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'   #initialize the paths to the model and preprocessor pickle files
            model = load_object(file_path=model_path)     #go to utils and create the load_object function
            preprocessor = load_object(file_path=preprocessor_path)      #load the model path and the preprocessor path
            data_scaled = preprocessor.transform(features)         #scale the data using the preprocessor
            preds = model.predict(data_scaled)         #make prediction using the model
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:           # mapping variable class:this class is responsible for mapping all inputs given to the html to the backend and produces a dataframe
    def __init__( self,     # mapping function:this fxn maps all inputs given to the html to the backend
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):    #and this fxn produces a dataframe
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)