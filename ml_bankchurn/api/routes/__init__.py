from fastapi import FastAPI

from . import prediction, train


def route_registry(app: FastAPI):
    app.include_router(prediction.router)
    app.include_router(train.router)
