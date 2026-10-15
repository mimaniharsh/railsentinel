"""Request bodies for the REST API."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ConfirmationRequest(BaseModel):
    pnr: str | None = Field(None, description="Predict for an existing RailSentinel PNR")
    train_number: str | None = None
    origin: str | None = None
    destination: str | None = None
    travel_class: str | None = None
    journey_date: date | None = None
    booking_status: str = Field("WL", pattern="^(WL|RAC)$")
    queue_position: int = Field(20, ge=1, le=2000)
    quota: str = Field("GN", pattern="^(GN|TQ)$")


class UserProfile(BaseModel):
    hist_class_mode: str = "SL"
    hist_night_share: float = Field(0.6, ge=0, le=1)
    hist_fare_per_km: float = Field(0.5, gt=0, le=20)
    segment: str = "unknown"


class RecommendRequest(BaseModel):
    origin: str
    destination: str
    journey_date: date
    budget: float | None = Field(None, gt=0)
    classes: list[str] | None = None
    profile: UserProfile = UserProfile()


class DelayRequest(BaseModel):
    train_number: str
    station: str
    run_date: date | None = None


class FareRequest(BaseModel):
    train_number: str
    origin: str
    destination: str
    travel_class: str
    journey_date: date
    quoted_fare: float | None = Field(None, description="Override the fare-feed quote (to test anomaly checks)")


class PlanRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    session_id: str | None = None
    profile: UserProfile | None = None


class ApproveBooking(BaseModel):
    passenger_name: str = Field(..., min_length=2, max_length=60)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    interaction_id: str
    helpful: bool


class FraudSession(BaseModel):
    requests_per_min: float = Field(..., ge=0)
    interarrival_cv: float = Field(..., ge=0)
    form_fill_seconds: float = Field(..., ge=0)
    headless_signals: int = Field(0, ge=0, le=1)
    is_tatkal_window: int = Field(0, ge=0, le=1)
    ms_after_tatkal_open: float = -1
    booking_attempts: int = Field(1, ge=0)
    passengers_per_booking: int = Field(1, ge=1, le=6)
    accounts_on_device: int = Field(1, ge=1)
    accounts_on_payment_instrument: int = Field(1, ge=1)
    account_age_days: float = Field(365, ge=0)
    failed_payments: int = Field(0, ge=0)
    distinct_passenger_names_30d: int = Field(2, ge=0)
    ip_bookings_last_hour: int = Field(1, ge=0)
    account_tenure: str | None = None


class PassportVerify(BaseModel):
    passport: dict


class ApprovalDecision(BaseModel):
    approve: bool


class RollbackRequest(BaseModel):
    to_version: int | None = None


class PromoteRequest(BaseModel):
    version: int


class ScenarioRequest(BaseModel):
    scenario: str
    max_rounds: int = Field(8, ge=1, le=12)
    reset_first: bool = Field(True, description="Autopilot only: return to the launch state before the scenario, "
                                                "so models adapted during an earlier scenario do not carry over")


class TrafficRequest(BaseModel):
    scale: float = Field(1.0, gt=0, le=3)
