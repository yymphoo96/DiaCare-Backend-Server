from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import *
import joblib


MODEL_PATH = "model/my_tuned_catboost_model.cbm"  # CatBoost native format
SCALER_PATH ="model/scaler.pkl"
# OR
# MODEL_PATH = "models/diabetes_model.pkl"  # Pickle format

class ModelManager:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained CatBoost model"""
        try:
            if MODEL_PATH.endswith('.cbm'):
                # Load CatBoost native format
                self.model = CatBoostClassifier()
                self.model.load_model(MODEL_PATH)
                self.scaler =  joblib.load(SCALER_PATH)

                print("✅ Model loaded from .cbm file")
            else:
                # Load from pickle
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ Model loaded from pickle file")
            
            # Get feature names
            try:
                self.feature_names = self.model.feature_names_
                print(f"📊 Model expects {len(self.feature_names)} features")
            except:
                print("⚠️ Could not extract feature names from model")
            
            print("🎯 Model ready for predictions!")
            
        except FileNotFoundError:
            print(f"❌ Model file not found at: {MODEL_PATH}")
            print("Please train and save your model first!")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")

# Initialize model manager
model_manager = ModelManager()


class BRFSSPredictionInput(BaseModel):
    """Input data for diabetes prediction"""

    high_bp: bool = Field(..., description="High blood pressure, 0 = no high BP, 1 = high BP")
    high_chol: bool = Field(..., description="HighChol, 0 = no high cholesterol, 1 = high cholesterol")
    chol_check: bool = Field(description="CholCheck, 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years")
    bmi: float = Field(description="BMI")
    smoker: bool = Field(description="Smoker, Have you smoked at least 100 cigarettes in your entire life? \
                         [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes")
    stroke: bool = Field(description="Stroke, 0=No, 1=Yes")
    heart_disease: bool = Field(..., description="HeartDiseaseorAttack coronary heart disease (CHD) or myocardial infarction (MI) \
                                0 = no 1 = yes")
    physical_activity: bool = Field(description="Physical activity, \
                                               physical activity in past 30 days - not including job 0 = no 1 = yes")
    fruits: bool = Field(description="Fruits, Consume Fruit 1 or more times per day 0 = no, 1 = yes")
    veggies:  bool = Field(description="Veggies, Consume Vegetables 1 or more times per day 0 = no, 1 = yes")
    heavy_alcohol: bool = Field(description="HvyAlcoholConsump, (adult men >=14 drinks per week and adult women>=7 drinks per week)\
                                , 0 = no, 1 = yes")
    health_insurance: bool = Field(description="AnyHealthcare, Have any kind of health care coverage, including health insurance, \
                                   prepaid plans such as HMO, etc. 0 = no 1 = yes")
    no_doctor_cost: bool = Field(description="NoDocbcCost, Was there a time in the past 12 months when you needed to see a doctor but\
                                  could not because of cost? 0 = no 1 = yes")
    general_health: int = Field(..., ge=1, le=5, description="General health,Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor")
    mental_health: int = Field(..., ge=1, le=30, description="Mental health, days of poor mental health scale 1-30 days")
    physical_health: int = Field(..., ge=1, le=30, description="Physical health, physical illness or injury days in past 30 days scale 1-30")
    difficulty_walking: bool = Field(..., description="Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes")
    gender: bool = Field(..., description="0=female, 1=male")
    age: int = Field(..., ge=1, le=13, description="Age in years,1: Age 18 to 24\
                                                                2: Age 25 to 29\
                                                                3: Age 30 to 34\
                                                                4: Age 35 to 39\
                                                                5: Age 40 to 44\
                                                                6: Age 45 to 49\
                                                                7: Age 50 to 54\
                                                                8: Age 55 to 59\
                                                                9: Age 60 to 64\
                                                                10: Age 65 to 69\
                                                                11: Age 70 to 74\
                                                                12: Age 75 to 79\
                                                                13: Age 80 or older ")
    education: Optional[int] = Field(None, ge=1, le=6, description="Education level (1-6), \
                                                                1: Never attended school or only kindergarten\
                                                                2: Grades 1 through 8 (Elementary)\
                                                                3: Grades 9 through 11 (Some high school)\
                                                                4: Grade 12 or GED (High school graduate)\
                                                                5: College 1 year to 3 years (Some college or technical school)\
                                                                6: College 4 years or more (College graduate)")
    
    income: Optional[int] = Field(None, ge=1, le=8, description="Income level (1-8)")

    
    class Config:
        schema_extra = {
            "example": {
                "high_bp": 1,
                "high_chol": 1,
                "chol_check": 1,
                "bmi": 32.5,
                "smoker": 0,
                "stroke": 0,
                "heart_disease": 0,
                "physical_activity": 1,
                "fruits": 0,
                "veggies": 1,
                "heavy_alcohol": 1,
                "health_insurance": 1,
                "no_doctor_cost": 0,
                "general_health": 3,
                "mental_health": 12,
                "physical_health": 18,
                "difficulty_walking": 0,
                "gender": 1,
                "age": 11,
                "education": 4,
                "income": 5
            }
        }

class PredictionResponse(BaseModel):
    """Response with prediction results"""
    prediction: str  # "High Risk" or "Low Risk"
    probability: float  # Probability of diabetes (0-1)
    risk_level: str  # "Low", "Moderate", "High"
    risk_score: int  # 0-100
    risk_factors: List[str]
    recommendations: List[str]
    timestamp: str
# ============================================================================
# Convert iOS Health Profile to BRFSS Features
# ============================================================================

def convert_ios_to_brfss(
    # iOS inputs (what we collect in the app)
    age_years: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    high_bp: bool,
    high_chol: bool,
    chol_checked: bool,
    smoking: bool,
    heart_disease: bool,
    physical_activity_mins: float,
    eat_fruit_daily: bool,
    eat_veggies_daily: bool,
    heavy_alcohol: bool,
    general_health_rating: int,  # 1-5
    mental_health_rating: int,    # 1-5
    physical_health_rating: int,  # 1-5
    difficulty_walking: bool,
    education_level: int,         # 1-6
    income_level: int,            # 1-8
    has_insurance: bool = True,
    stroke_history: bool = False,
    cant_afford_doctor: bool = False
) -> BRFSSPredictionInput:
    """
    Convert iOS health profile data to BRFSS model features
    """
    
    # Calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # Convert age to category (1-13)
    age_category = min(13, max(1, age_years // 5))
    
    # Convert health ratings (1-5) to days (0-30)
    # Rating 1 (excellent) → 0 days, Rating 5 (poor) → 30 days
    mental_health_days = int((mental_health_rating - 1) * 7.5)
    physical_health_days = int((physical_health_rating - 1) * 7.5)
    
    # Gender: 0=female, 1=male
    gender_code = 1 if gender.lower() == 'male' else 0
    
    # Physical activity: 0=none, 1=any
    has_physical_activity = 1 if physical_activity_mins > 0 else 0
    
    return BRFSSPredictionInput(
        high_bp=int(high_bp),
        high_chol=int(high_chol),
        chol_check=int(chol_checked),
        bmi=bmi,
        smoker=int(smoking),
        stroke=int(stroke_history),
        heart_disease=int(heart_disease),
        physical_activity=has_physical_activity,
        fruits=int(eat_fruit_daily),
        veggies=int(eat_veggies_daily),
        heavy_alcohol=int(heavy_alcohol),
        health_insurance=int(has_insurance),
        no_doctor_cost=int(cant_afford_doctor),
        general_health=general_health_rating,
        mental_health=mental_health_days,
        physical_health=physical_health_days,
        difficulty_walking=int(difficulty_walking),
        gender=gender_code,
        age=age_category,
        education=education_level,
        income=income_level
    )

# ============================================================================
# Feature Engineering for Model
# ============================================================================

def prepare_features_for_model(data: BRFSSPredictionInput) -> pd.DataFrame:
    """
    Create feature DataFrame matching EXACT model feature order
    """
    
    # Use defaults if not provided
    features = {
        'HighBP': float(data.high_bp if data.high_bp is not None else 0),
        'HighChol': float(data.high_chol if data.high_chol is not None else 0),
        'CholCheck': float(data.chol_check if data.chol_check is not None else 1),
        'BMI': float(data.bmi if data.bmi is not None else 25.0),
        'Smoker': float(data.smoker if data.smoker is not None else 0),
        'Stroke': float(data.stroke if data.stroke is not None else 0),
        'HeartDiseaseorAttack': float(data.heart_disease if data.heart_disease is not None else 0),
        'PhysActivity': float(data.physical_activity if data.physical_activity is not None else 1),
        'Fruits': float(data.fruits if data.fruits is not None else 1),
        'Veggies': float(data.veggies if data.veggies is not None else 1),
        'HvyAlcoholConsump': float(data.heavy_alcohol if data.heavy_alcohol is not None else 0),
        'AnyHealthcare': float(data.health_insurance if data.health_insurance is not None else 1),
        'NoDocbcCost': float(data.no_doctor_cost if data.no_doctor_cost is not None else 0),
        'GenHlth': float(data.general_health if data.general_health is not None else 3),
        'MentHlth': float(data.mental_health if data.mental_health is not None else 0),
        'PhysHlth': float(data.physical_health if data.physical_health is not None else 0),
        'DiffWalk': float(data.difficulty_walking if data.difficulty_walking is not None else 0),
        'Sex': float(data.gender if data.gender is not None else 1),
        'Age': float(data.age if data.age is not None else 8),
        'Education': float(data.education if data.education is not None else 4),
        'Income': float(data.income if data.income is not None else 4)
    }
    
    return pd.DataFrame([features])

# ============================================================================
# Risk Assessment
# ============================================================================

def assess_risk_factors(features: pd.DataFrame) -> List[str]:
    risk_factors = []
    
    if features['BMI'].iloc[0] > 30:
        risk_factors.append("High BMI (Obesity - BMI > 30)")
    elif features['BMI'].iloc[0] > 25:
        risk_factors.append("Overweight (BMI 25-30)")
    
    if features['HighBP'].iloc[0] == 1:
        risk_factors.append("High blood pressure")
    
    if features['HighChol'].iloc[0] == 1:
        risk_factors.append("High cholesterol")
    
    if features['Smoker'].iloc[0] == 1:
        risk_factors.append("Smoking history")
    
    if features['PhysActivity'].iloc[0] == 0:
        risk_factors.append("No physical activity in past 30 days")
    
    if features['Fruits'].iloc[0] == 0:
        risk_factors.append("Low fruit consumption")
    
    if features['Veggies'].iloc[0] == 0:
        risk_factors.append("Low vegetable consumption")
    
    if features['GenHlth'].iloc[0] >= 4:
        risk_factors.append("Fair or poor general health")
    
    if features['Age'].iloc[0] >= 9:
        risk_factors.append("Age 60 or older")
    
    return risk_factors

def generate_recommendations(risk_factors: List[str], features: pd.DataFrame) -> List[str]:
    recommendations = []
    
    if features['BMI'].iloc[0] > 25:
        recommendations.append("🎯 Aim for healthy weight through balanced diet and exercise")
    
    if features['PhysActivity'].iloc[0] == 0:
        recommendations.append("🏃 Get at least 30 minutes of moderate exercise daily")
    
    if features['Fruits'].iloc[0] == 0 or features['Veggies'].iloc[0] == 0:
        recommendations.append("🥗 Increase fruit and vegetable intake to 5+ servings daily")
    
    if features['Smoker'].iloc[0] == 1:
        recommendations.append("🚭 Quit smoking - consult your doctor for support programs")
    
    if features['HighBP'].iloc[0] == 1:
        recommendations.append("💊 Monitor blood pressure regularly and follow doctor's advice")
    
    if features['HighChol'].iloc[0] == 1:
        recommendations.append("❤️ Manage cholesterol through diet and medication if prescribed")
    
    recommendations.append("📅 Schedule regular health checkups with your doctor")
    recommendations.append("📊 Monitor blood sugar levels if at high risk")
    
    return recommendations[:6]