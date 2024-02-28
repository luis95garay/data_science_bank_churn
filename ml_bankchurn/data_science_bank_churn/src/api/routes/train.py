import subprocess
from fastapi.routing import APIRouter


router = APIRouter(tags=['train'])


@router.post("/train")
async def train():
    subprocess.run(["kedro", "run"], cwd='data_science_bank_churn')
    return {'data': "Modelo entrenado exitosamente"}
