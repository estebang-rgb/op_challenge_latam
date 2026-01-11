from typing import Any, Dict, List

import fastapi
import pandas as pd
from pydantic import BaseModel

from challenge.model import DelayModel

app = fastapi.FastAPI()

model = DelayModel()
model_trained = False


class FlightRequest(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class PredictRequest(BaseModel):
    flights: List[FlightRequest]


def train_model():
    """Train the model if not already trained."""
    global model_trained
    if not model_trained:
        data = pd.read_csv("data/data.csv")

        features, target = model.preprocess(data, target_column="delay")
        model.fit(features, target)
        model_trained = True


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> Dict[str, Any]:
    try:
        # Train model if not trained
        train_model()

        # Validate and convert request to DataFrame
        flights_data = []
        for flight in request.flights:
            # Manual validation
            if not isinstance(flight.MES, int) or not (1 <= flight.MES <= 12):
                raise fastapi.HTTPException(
                    status_code=400, detail="MES must be an integer between 1 and 12"
                )
            if flight.TIPOVUELO not in ["I", "N"]:
                raise fastapi.HTTPException(
                    status_code=400,
                    detail='TIPOVUELO must be "I" (International) or "N" (National)',
                )

            flights_data.append(
                {
                    "OPERA": flight.OPERA,
                    "TIPOVUELO": flight.TIPOVUELO,
                    "MES": flight.MES,
                }
            )

        df = pd.DataFrame(flights_data)

        # Preprocess for prediction
        features = model.preprocess(df)

        # Make predictions
        predictions = model.predict(features)

        return {"predict": predictions}

    except fastapi.HTTPException:
        raise
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )
