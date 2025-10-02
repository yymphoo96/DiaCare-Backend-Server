from pydantic import BaseModel, EmailStr


class ActivityData(BaseModel):
    user_id: int
    activity_type: str
    value: int
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