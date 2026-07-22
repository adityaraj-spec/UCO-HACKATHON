"""
migrate_voiceprints_to_biohashes.py

Data migration script converting all existing raw vector voiceprints in PostgreSQL
into 256-bit BioHash tokens using BioHashService.
Ensures zero disruption for existing enrolled users.
"""

import asyncio
from sqlalchemy import select
from app.database.session import AsyncSessionLocal
from app.models.voiceprint import Voiceprint
from app.services.biohash_service import BioHashService


async def migrate_voiceprints():
    print("Starting voiceprint BioHash migration...")
    biohash_service = BioHashService()

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Voiceprint))
        voiceprints = result.scalars().all()

        updated_count = 0
        for vp in voiceprints:
            if not vp.biohash and vp.embedding:
                vp.biohash = biohash_service.compute_biohash(vp.embedding)
                updated_count += 1

        await session.commit()
        print(f"Migration complete: Successfully generated BioHashes for {updated_count} existing voiceprints.")


if __name__ == "__main__":
    asyncio.run(migrate_voiceprints())
