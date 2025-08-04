from typing import Optional
from pydantic import BaseModel, Field


# These schemas are used to ensure the request and response formats for the API endpoints are well-defined.
# They help in validating the data structure and types for incoming requests and outgoing responses.


# Request Schema for /predict
class PredictionRequest(BaseModel):
    name: str = Field(..., description="Name of the stock")
    date: str = Field(..., description="Date of the prediction in YYYY-MM-DD format")
    current_price: float = Field(..., description="Current price of the stock")


class PredictionResponse(BaseModel):
    # utilize the sample aboive to creratge the general structure
    four_hours: dict = Field(
        ...,
        description="Predicted price and change for the next 4 hours",
    )
    twenty_four_hours: dict = Field(
        ...,
        description="Predicted price and change for the next 24 hours",
    )
    two_days: dict = Field(
        ...,
        description="Predicted price and change for the next 2 days",
    )
    seven_days: dict = Field(
        ...,
        description="Predicted price and change for the next 7 days",
    )
