import asyncio
from app.database.session import AsyncSessionLocal
from sqlalchemy import text

async def get_users():
    async with AsyncSessionLocal() as session:
        result = await session.execute(text('SELECT id, name FROM users;'))
        for row in result:
            print(f'{row[1]}: {row[0]}')

asyncio.run(get_users())
