# Pydantic models for the web service
from pydantic import BaseModel


class ModelInput(BaseModel):
    """The class representing model input."""

    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float
    sex: str


class ModelOutput(BaseModel):
    """The class representing model output."""

    abalone_age: float
