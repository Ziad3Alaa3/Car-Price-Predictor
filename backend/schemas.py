from pydantic import BaseModel, Field

class CarIn(BaseModel):
    brand: str = Field(..., examples=["Toyota"])
    year: int = Field(..., ge=1950, le=2040, examples=[2015])
    km_driven: int = Field(..., ge=0, examples=[60000])
    fuel: str = Field(..., examples=["Petrol"])
    transmission: str = Field(..., examples=["Manual"])
    owner: str = Field(..., examples=["First Owner"])

class Car_Price(BaseModel):
    price: float
    currency: str = "EGP"
    model_version: str