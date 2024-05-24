from fastapi import FastAPI
# from api.exceptions import registry_exceptions
from api.routes import route_registry


def get_api() -> FastAPI:
    app = FastAPI(title="Bank Churn")
    route_registry(app)
    # registry_exceptions(app)
    return app
