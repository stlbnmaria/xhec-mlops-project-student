from typing import Any, Dict

from fastapi import FastAPI

from src.web_service.app_config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from src.web_service.lib.inference import infer_age
from src.web_service.lib.models import ModelInput, ModelOutput

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)


@app.get("/")
def home() -> Dict[str, Any]:
    """The home end-point to check if the app is running.

    Returns
    -------
    dict
        The health-check message.
    """
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=ModelOutput, status_code=201)
def predict(payload: ModelInput) -> ModelOutput:
    """The endpoint to get model prediction.

    Parameters
    ----------
    payload : ModelInput
        The model input.

    Returns
    -------
    predicted_age : ModelOutput
        The predicted age.
    """
    predicted_age = infer_age(payload)
    return predicted_age
