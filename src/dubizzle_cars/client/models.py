"""Frontend response contracts, independent of backend domain classes."""

from pydantic import BaseModel, ConfigDict, Field


class ResponseModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class Metadata(ResponseModel):
    sale_price_aed: str | None
    mileage_km: int | None
    warranty_mention: str | None
    regional_specs: str | None


class Car(ResponseModel):
    listing_id: str
    year: int | None
    make: str | None
    model: str | None
    trim: str | None
    title: str | None
    photo_url: str | None
    metadata: Metadata


class Chat(ResponseModel):
    reply: str = Field(min_length=1)
    cars: list[Car]
    session_id: str | None = None


class Session(ResponseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    returning_user: bool


class Profile(ResponseModel):
    user_id: str
    display_name: str | None
    preferences: dict[str, str | int | bool | None]


class Health(ResponseModel):
    inventory_count: int = Field(ge=0)
    semantic_retrieval_available: bool
