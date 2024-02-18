from pathlib import Path
from fastapi import Request
from fastapi.routing import APIRouter
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates

from src.api.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.api.schemas import EmployeeData


router = APIRouter(tags=['prediction'])

# templates_folder = Path(__file__).parent.parent.parent / "templates"
# templates = Jinja2Templates(directory=str(templates_folder))


# @router.get("/", response_class=HTMLResponse)
@router.get("/")
async def index(request: Request):
    # return templates.TemplateResponse("index.html", {"request": request})
    return {'result': 'exito'}


@router.post("/predict")
async def predict(request: Request, employeedata: EmployeeData):
    data = CustomData(
        employeedata.customer_age,
        employeedata.gender,
        employeedata.dependent_count,
        employeedata.education_level,
        employeedata.marital_status,
        employeedata.income_category,
        employeedata.card_category,
        employeedata.months_on_book,
        employeedata.total_relationship_count,
        employeedata.months_inactive_12_mon,
        employeedata.contacts_count_12_mon,
        employeedata.credit_limit,
        employeedata.total_revolving_Bal,
        employeedata.avg_open_to_buy,
        employeedata.total_amt_chng_q4_q1,
        employeedata.total_trans_amt,
        employeedata.total_trans_ct,
        employeedata.total_ct_chng_q4_q1,
        employeedata.avg_utilization_ratio
    )
    pred_df = data.get_data_as_data_frame()
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("after Prediction")

    # return templates.TemplateResponse("index.html", {"request": request, "results": results['result']})
    return {'data': results}
