# ============================================================================
# FastAPI Backend - Health Activities Endpoints
# ============================================================================

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date, datetime, timedelta
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# ============================================================================
# Database Setup
# ============================================================================

# Database URL - Update with your database credentials
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/wellness_db")
# For SQLite (development): DATABASE_URL = "sqlite:///./wellness.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============================================================================
# Database Models
# ============================================================================

class Activity(Base):
    __tablename__ = "activities"
    
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



# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(title="Wellness API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your iOS app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Database Dependency
# ============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# Authentication (Optional - Add your auth logic)
# ============================================================================

async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Extract user from Bearer token
    Implement your own authentication logic here
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        # TODO: Validate token and get user_id
        # For now, we'll use user_id from query params
        return token
    return None

# ============================================================================
# ENDPOINT 1: Get Missing Dates
# ============================================================================

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

# ============================================================================
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ============================================================================