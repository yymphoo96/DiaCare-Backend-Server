from pydantic import BaseModel, EmailStr
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, UniqueConstraint


class ActivityData(BaseModel):
    user_id: int
    activity_type: str
    value: float
    unit: str
    date: str

class UserSignup(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class AuthResponse(BaseModel):
    token: str
    user: UserResponse

class UserUpdate(BaseModel):
    name: str

class ActivityResponse(BaseModel):
    success: bool
    message: str

class MissingDatesResponse(BaseModel):
    missing_dates: List[str]
    total_missing: int
    date_range: dict
