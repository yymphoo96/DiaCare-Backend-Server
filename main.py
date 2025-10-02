from fastapi import FastAPI, Depends, HTTPException
from typing import List, Optional
import asyncio
from datetime import datetime
from models import *
from database import *
from auth import *

# Create a FastAPI instance. This is the main object for your API.
app = FastAPI()


@app.get("/phoo/first")
async def read_root():
    return {"Hello": "World"}

@app.post("/api/steps/update")
async def update_steps(
    data: ActivityData, 
    conn: asyncpg.Connection = Depends(get_database)
):
    try:
        # Convert to Python date object
        date_obj = datetime.strptime(data.date, "%Y-%m-%d").date()
        
        query = """
        INSERT INTO health_activities (user_id, activity_type, value, date, source)
        VALUES ($1, $2, $3, $4, 'HealthKit')
        ON CONFLICT (user_id, activity_type, date)
        DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
        """
        await conn.execute(query, data.user_id, data.activity_type, data.value, date_obj)
        return {"message": "Step count updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/steps/info/{user_id}", response_model=List[ActivityData])
async def get_steps(
    user_id: int,
    conn: asyncpg.Connection = Depends(get_database)
):
    try:
        query = """
        SELECT user_id, activity_type, value, date 
        FROM health_activities 
        WHERE user_id = $1
        ORDER BY date DESC;
        """
        rows = await conn.fetch(query, user_id)
        
        if not rows:
            raise HTTPException(status_code=404, detail="No activities found for this user")
        
        return [
            ActivityData(
                user_id=row['user_id'],
                activity_type=row['activity_type'],
                value=row['value'],
                date=str(row['date'])
            )
            for row in rows
        ]
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
@app.post("/auth/signup", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def signup(
    user_data: UserSignup,
    conn: asyncpg.Connection = Depends(get_database)
):
    """Register a new user"""
    
    # Check if email already exists
    existing_user = await conn.fetchrow(
        "SELECT id FROM users WHERE email = $1",
        user_data.email
    )
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password and create user
    hashed_password = hash_password(user_data.password)
    user = await conn.fetchrow(
        """
        INSERT INTO users (name, email, password) 
        VALUES ($1, $2, $3) 
        RETURNING id, name, email
        """,
        user_data.name, user_data.email, hashed_password
    )
    
    # Generate JWT token
    token = create_access_token(user["id"], user["email"])
    
    return AuthResponse(
        token=token,
        user=UserResponse(**dict(user))
    )

@app.post("/auth/login", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def login(
    user_data: UserLogin,
    conn: asyncpg.Connection = Depends(get_database)
):
    """Login user and return JWT token"""
    
    # Find user by email
    user = await conn.fetchrow(
        "SELECT id, name, email, password FROM users WHERE email = $1",
        user_data.email
    )
    
    # Verify user exists and password is correct
    if not user or not verify_password(user_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Generate JWT token
    token = create_access_token(user["id"], user["email"])
    
    return AuthResponse(
        token=token,
        user=UserResponse(id=user["id"], name=user["name"], email=user["email"])
    )

@app.get("/user/profile", response_model=UserResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile (requires authentication)"""
    return UserResponse(**current_user)

@app.put("/user/profile", response_model=UserResponse)
async def update_profile(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_database)
):
    """Update user profile (requires authentication)"""
    
    updated_user = await conn.fetchrow(
        """
        UPDATE users 
        SET name = $1 
        WHERE id = $2 
        RETURNING id, name, email
        """,
        user_update.name, current_user["id"]
    )
    
    return UserResponse(**dict(updated_user))

# Run with: uvicorn main:app --reload