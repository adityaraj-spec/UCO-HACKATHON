"""
app/repositories/user_repository.py

Data-access layer for the `users` table.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User


class UserRepository:
    """Encapsulates all database operations for the User model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, email: str) -> User:
        """Insert a new user row and flush so its generated id is available."""
        user = User(name=name, email=email)
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        """Fetch a user by primary key, or None if not found."""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id_with_voiceprint(self, user_id: uuid.UUID) -> User | None:
        """Fetch a user eagerly loading its voiceprint relationship."""
        result = await self.session.execute(
            select(User)
            .options(selectinload(User.voiceprint))
            .where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user and user.voiceprint:
            from app.repositories.voiceprint_repository import VoiceprintRepository
            VoiceprintRepository(self.session)._attach_plaintext_voiceprint(user.voiceprint)
        return user

    async def get_by_email(self, email: str) -> User | None:
        """Fetch a user by unique email address, or None if not found."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def exists(self, user_id: uuid.UUID) -> bool:
        """Return True if a user with the given id exists."""
        result = await self.session.execute(
            select(User.id).where(User.id == user_id)
        )
        return result.scalar_one_or_none() is not None
