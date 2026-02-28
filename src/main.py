from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, predict_probability


app = FastAPI(title="Wine Classification API")


# Wine dataset has 13 features
class WineData(BaseModel):

    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float


class PredictionResponse(BaseModel):

    predicted_class: int


class ProbabilityResponse(BaseModel):

    probabilities: list


@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "Wine API healthy"}


# Endpoint 1: Prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict_wine(data: WineData):

    try:

        features = [[
            data.alcohol,
            data.malic_acid,
            data.ash,
            data.alcalinity_of_ash,
            data.magnesium,
            data.total_phenols,
            data.flavanoids,
            data.nonflavanoid_phenols,
            data.proanthocyanins,
            data.color_intensity,
            data.hue,
            data.od280_od315,
            data.proline
        ]]

        prediction = predict_data(features)

        return PredictionResponse(
            predicted_class=int(prediction[0])
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Endpoint 2: NEW ENDPOINT (required by lab)
@app.post("/predict_proba", response_model=ProbabilityResponse)
async def predict_probability_endpoint(data: WineData):

    try:

        features = [[
            data.alcohol,
            data.malic_acid,
            data.ash,
            data.alcalinity_of_ash,
            data.magnesium,
            data.total_phenols,
            data.flavanoids,
            data.nonflavanoid_phenols,
            data.proanthocyanins,
            data.color_intensity,
            data.hue,
            data.od280_od315,
            data.proline
        ]]

        probabilities = predict_probability(features)

        return ProbabilityResponse(
            probabilities=probabilities[0].tolist()
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
