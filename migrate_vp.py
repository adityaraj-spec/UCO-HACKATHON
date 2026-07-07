import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import torch
from app.database.session import AsyncSessionLocal
from app.models.user import User
from app.models.voiceprint import Voiceprint
from sqlalchemy.future import select

async def migrate_voiceprints():
    # Load the local pth file
    try:
        voiceprints = torch.load("models/voiceprints.pth", weights_only=True)
    except:
        voiceprints = torch.load("models/voiceprints.pth")
    
    async with AsyncSessionLocal() as session:
        # Get all users
        result = await session.execute(select(User))
        users = result.scalars().all()
        
        for user in users:
            # Match name (e.g. "Aaditya Raj" -> "aditya")
            name_lower = user.name.lower()
            match_key = None
            for key in voiceprints.keys():
                if key in name_lower or name_lower in key:
                    match_key = key
                    break
            
            if match_key:
                print(f"Found match: {user.name} -> {match_key}")
                emb = voiceprints[match_key].flatten().tolist()
                
                # Check if voiceprint exists
                vp_result = await session.execute(select(Voiceprint).where(Voiceprint.user_id == user.id))
                vp = vp_result.scalars().first()
                
                if not vp:
                    vp = Voiceprint(user_id=user.id, embedding=emb, recording_count=10)
                    session.add(vp)
                    print(f"Added voiceprint for {user.name}")
                else:
                    vp.embedding = emb
                    print(f"Updated voiceprint for {user.name}")
                    
        await session.commit()
        print("Migration complete!")

asyncio.run(migrate_voiceprints())
