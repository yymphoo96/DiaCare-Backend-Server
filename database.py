import asyncpg
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
import os

# Database connection
DATABASE_URL = "postgresql://yinyinmayphoo:password@localhost:5432/DiaCare"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://yinyinmayphoo:password@localhost:5432/DiaCare")
# For SQLite (development): DATABASE_URL = "sqlite:///./wellness.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def get_database():
    """Get database connection"""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()