import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        Proposal_Type: str,
        test_preparation_course: str,
        Website_Lead: int,
        Social_Media_Lead: int,
        Referral_Lead: int,
        total_lead_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.Proposal_Type = Proposal_Type

        self.test_preparation_course = test_preparation_course

        self.Website_Lead = Website_Lead

        self.Social_Media_Lead = Social_Media_Lead

        self.Referral_Lead = Referral_Lead
    
        self.total_lead_score = total_lead_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "Proposal_Type": [self.Proposal_Type],
                "test_preparation_course": [self.test_preparation_course],
                "Website_Lead": [self.Website_Lead],
                "Social_Media_Lead": [self.Social_Media_Lead],
                "Referral_Lead": [self.Referral_Lead],
                "total_lead_score": [self.total_lead_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)