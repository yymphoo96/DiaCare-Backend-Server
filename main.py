from fastapi import FastAPI, Depends, HTTPException
from typing import List, Optional
import asyncio
from datetime import datetime,date
from models import *
from database import *
from auth import *
from prediction import *
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
import os
import joblib
import numpy as np

# Create a FastAPI instance. This is the main object for your API.
app = FastAPI()

class Base(DeclarativeBase):
    pass
class Activity(Base):
    __tablename__ = "health_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    activity_type = Column(String(50), nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)
    date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('user_id', 'activity_type', 'date', name='unique_user_activity_date'),
    )

@app.get("/api/activities/missing-dates", response_model=MissingDatesResponse)
async def get_missing_dates(
    user_id: str,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get list of dates that don't have activity data in the database.
    
    Parameters:
    - user_id: User identifier
    - days: Number of days to check (default: 30)
    
    Returns:
    - missing_dates: List of date strings in YYYY-MM-DD format
    - total_missing: Count of missing dates
    - date_range: Start and end dates of the check
    """
    try:
        # Generate list of all dates in the range
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)
        
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            all_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        # Query database for existing dates with any activity
        existing_dates_query = db.query(Activity.date).filter(
            Activity.user_id == user_id,
            Activity.date >= start_date,
            Activity.date <= end_date
        ).distinct().all()
        
        # Convert query results to set of date strings
        existing_dates = {str(d[0]) for d in existing_dates_query}
        
        # Find missing dates
        missing_dates = [d for d in all_dates if d not in existing_dates]
        
        return MissingDatesResponse(
            missing_dates=missing_dates,
            total_missing=len(missing_dates),
            date_range={
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "total_days": days
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching missing dates: {str(e)}")

# ============================================================================
# ENDPOINT 2: Update/Create Activity Data
# ============================================================================

@app.post("/api/activities/update", response_model=ActivityResponse)
async def update_activity(
    data: ActivityData,
    db: Session = Depends(get_db)
):
    """
    Create or update activity data for a specific date.
    Uses UPSERT logic (update if exists, insert if not).
    
    Parameters:
    - user_id: User identifier
    - activity_type: Type of activity (e.g., "step_count", "active_energy")
    - value: Numeric value of the activity
    - unit: Unit of measurement (e.g., "steps", "kcal")
    - date: Date in YYYY-MM-DD format
    
    Returns:
    - success: Boolean indicating if operation was successful
    - message: Descriptive message
    """
    try:
        # Parse date string to date object
        activity_date = datetime.strptime(data.date, "%Y-%m-%d").date()
        
        # Check if record already exists
        existing_activity = db.query(Activity).filter(
            Activity.user_id == data.user_id,
            Activity.activity_type == data.activity_type,
            Activity.date == activity_date
        ).first()
        
        if existing_activity:
            # Update existing record
            existing_activity.value = data.value
            existing_activity.unit = data.unit
            existing_activity.updated_at = datetime.utcnow()
            message = f"Updated {data.activity_type} for {data.date}"
        else:
            # Create new record
            new_activity = Activity(
                user_id=data.user_id,
                activity_type=data.activity_type,
                value=data.value,
                unit=data.unit,
                date=activity_date,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(new_activity)
            message = f"Created {data.activity_type} for {data.date}"
        
        db.commit()
        
        return ActivityResponse(
            success=True,
            message=message
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating activity: {str(e)}")

# ============================================================================
# ENDPOINT 3: Get Activities for Date Range (Bonus)
# ============================================================================

@app.get("/api/activities/range")
async def get_activities_range(
    user_id: str,
    start_date: str,
    end_date: str,
    activity_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all activities for a user within a date range.
    Useful for displaying chart data.
    """
    try:
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Build query
        query = db.query(Activity).filter(
            Activity.user_id == user_id,
            Activity.date >= start,
            Activity.date <= end
        )
        
        # Filter by activity type if provided
        if activity_type:
            query = query.filter(Activity.activity_type == activity_type)
        
        # Execute query
        activities = query.order_by(Activity.date.asc()).all()
        
        # Format response
        result = []
        for activity in activities:
            result.append({
                "date": str(activity.date),
                "activity_type": activity.activity_type,
                "value": activity.value,
                "unit": activity.unit
            })
        
        return {
            "success": True,
            "count": len(result),
            "activities": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching activities: {str(e)}")

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "message": "Wellness API is running",
        "version": "1.0.0"
    }


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

@app.post("/api/predict-diabetes", response_model=PredictionResponse)
async def predict_diabetes(data: BRFSSPredictionInput):
    try:
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features
        features_df = prepare_features_for_model(data)
        print(data)
        #Scaling
        values = list(data.dict().values())

        # Shape (1, n_features)
        user_input = np.array([values], dtype=float)

        # Scale using SAME scaler
        # print(model_manager.scaler.var_)
        print(user_input)
        user_input[:, [3,13,14,15,18,19,20]] = model_manager.scaler.transform(user_input[:, [3,13,14,15,18,19,20]])
        print(user_input)
        
        # Make prediction
        prediction_proba = model_manager.model.predict_proba(user_input)[0]
        probability = float(prediction_proba[1])
        prediction_class = int(probability >= 0.5)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        risk_score = int(probability * 100)
        risk_factors = assess_risk_factors(features_df)
        # print(features_df)
        # print(user_input_scaled)
        print(prediction_proba)
        recommendations = generate_recommendations(risk_factors, features_df)
        
        response = PredictionResponse(
            prediction="High Risk" if prediction_class == 1 else "Low Risk",
            probability=round(probability, 3),
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
        
        print(f"✅ Prediction: {response.prediction} ({risk_score}%)")
        print("Done, Thanks for prediction..............")
        return response
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Info Endpoint
# ============================================================================

@app.get("/api/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    
    if model_manager.model is None:
        return {
            "status": "error",
            "message": "Model not loaded"
        }
    
    try:
        return {
            "status": "ready",
            "model_type": "CatBoost Classifier",
            "model_loaded": True,
            "feature_count": len(model_manager.feature_names) if model_manager.feature_names else "Unknown",
            "model_path": MODEL_PATH
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
# Run with: uvicorn main:app --reload