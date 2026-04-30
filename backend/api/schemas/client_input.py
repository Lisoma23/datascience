from pydantic import BaseModel

class ClientInput(BaseModel):
    gender: str
    age: int
    country: str
    city: str
    customer_segment: str
    tenure_months : int
    signup_channel : str
    contract_type : str
    monthly_logins : int
    weekly_active_days : int
    avg_session_time : float
    features_used : int
    usage_growth_rate : float
    last_login_days_ago : int
    monthly_fee : int
    total_revenue : int
    payment_method : str
    payment_failures : int
    discount_applied : str
    price_increase_last_3m : str
    support_tickets : int
    avg_resolution_time : float
    complaint_type : str | None
    csat_score : float
    escalations : int
    email_open_rate : float
    marketing_click_rate : float
    nps_score : int
    survey_response: str
    referral_count : int
