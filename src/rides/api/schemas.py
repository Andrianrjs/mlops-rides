from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["Safe", "Safe Overall", "Bad Weather", "Tandem", "Reckless"]


class Ride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readings: list[list[float]] = Field(
        ...,
        min_length=10,
        description="Série IMU du trajet : une ligne par pas de temps, une colonne par capteur",
        examples=[[[0.01, -0.02, 9.81, 0.0, 0.0, 0.0]] * 10],
    )


class Prediction(BaseModel):
    label: Label
    probabilities: dict[str, float]
    model_version: str
