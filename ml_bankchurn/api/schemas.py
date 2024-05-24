from pydantic import BaseModel


class EmployeeData(BaseModel):
    customer_age: int = 54
    gender: str = 'M'
    dependent_count: int = 3
    education_level: str = 'High School'
    marital_status: str = 'Married'
    income_category: str = '$60K - $80K'
    card_category: str = 'Blue'
    months_on_book: int = 39
    total_relationship_count: int = 5
    months_inactive_12_mon: int = 1
    contacts_count_12_mon: int = 3
    credit_limit: float = 12691.0
    total_revolving_Bal: int = 777
    avg_open_to_buy: float = 11914.0
    total_amt_chng_q4_q1: float = 1.335
    total_trans_amt: int = 1144
    total_trans_ct: int = 42
    total_ct_chng_q4_q1: float = 1.625
    avg_utilization_ratio: float = 0.061
