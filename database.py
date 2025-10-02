import asyncpg

# Database connection
DATABASE_URL = "postgresql://yinyinmayphoo:password@localhost:5432/DiaCare"

async def get_database():
    """Get database connection"""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()