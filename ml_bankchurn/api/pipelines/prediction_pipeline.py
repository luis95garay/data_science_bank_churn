from pathlib import Path
import pickle

import pandas as pd


class PredictPipeline:
    def __init__(self):
        self.result_dict = {0: "No Churn", 1: "Churn"}
    
    def predict(self, df_features):
        file_path = Path(__file__).parent.parent.parent
        model_path = file_path / "data_science_bank_churn" / "data" / "06_models" / "best_model.pkl"
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        
        preprocessor_path = file_path / "data_science_bank_churn" / "data" / "05_model_input" / "preprocessor.pkl"
        with open(preprocessor_path, 'rb') as file:
            loaded_preprocessor = pickle.load(file)
        
        transformed_features = loaded_preprocessor.transform(df_features)
        result = loaded_model.predict(transformed_features)
        return self.result_dict[result[0]]


class CustomData:
    def __init__(self,
        customer_age: int,
        gender: str,
        dependent_count: int,
        education_level: str,
        marital_status: str,
        income_category: str,
        card_category: str,
        months_on_book: int,
        total_relationship_count: int,
        months_inactive_12_mon: int,
        contacts_count_12_mon: int,
        credit_limit: float,
        total_revolving_Bal: int,
        avg_open_to_buy: float,
        total_amt_chng_q4_q1: float,
        total_trans_amt: int,
        total_trans_ct: int,
        total_ct_chng_q4_q1: float,
        avg_utilization_ratio: float
        ):

        self.customer_age = customer_age
        self.gender = gender
        self.dependent_count = dependent_count
        self.education_level = education_level
        self.marital_status = marital_status
        self.income_category = income_category
        self.card_category = card_category
        self.months_on_book = months_on_book
        self.total_relationship_count = total_relationship_count
        self.months_inactive_12_mon = months_inactive_12_mon
        self.contacts_count_12_mon = contacts_count_12_mon
        self.credit_limit = credit_limit
        self.total_revolving_Bal = total_revolving_Bal
        self.avg_open_to_buy = avg_open_to_buy
        self.total_amt_chng_q4_q1 = total_amt_chng_q4_q1
        self.total_trans_amt = total_trans_amt
        self.total_trans_ct = total_trans_ct
        self.total_ct_chng_q4_q1 = total_ct_chng_q4_q1
        self.avg_utilization_ratio = avg_utilization_ratio
        self.custom_data_input_dict = {}

    def get_data_as_data_frame(self):
        self.custom_data_input_dict = {
            'Customer_Age': [self.customer_age],
            'Gender': [self.gender],
            'Dependent_count': [self.dependent_count],
            'Education_Level': [self.education_level],
            'Marital_Status': [self.marital_status],
            'Income_Category': [self.income_category],
            'Card_Category': [self.card_category],
            'Months_on_book': [self.months_on_book],
            'Total_Relationship_Count': [self.total_relationship_count],
            'Months_Inactive_12_mon': [self.total_relationship_count],
            'Contacts_Count_12_mon': [self.contacts_count_12_mon],
            'Credit_Limit': [self.credit_limit],
            'Total_Revolving_Bal': [self.total_revolving_Bal],
            'Avg_Open_To_Buy': [self.avg_open_to_buy],
            'Total_Amt_Chng_Q4_Q1': [self.total_amt_chng_q4_q1],
            'Total_Trans_Amt': [self.total_trans_amt],
            'Total_Trans_Ct': [self.total_trans_ct],
            'Total_Ct_Chng_Q4_Q1': [self.total_ct_chng_q4_q1],
            'Avg_Utilization_Ratio': [self.avg_utilization_ratio]
        }

        return pd.DataFrame(self.custom_data_input_dict)
