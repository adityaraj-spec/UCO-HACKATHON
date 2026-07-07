import asyncio
from app.database.session import AsyncSessionLocal
from sqlalchemy import text

async def get_vp():
    async with AsyncSessionLocal() as session:
        res = await session.execute(text('SELECT user_id, recording_count FROM voiceprints;'))
        for row in res:
            print(row)

asyncio.run(get_vp())
