"""
Database configuration and session management.
"""
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData

from backend.app.core.config import get_settings

settings = get_settings()

# Create async engine
# Use asyncpg for async operations, fallback to sync for migrations
if settings.database_url:
    # Convert sync URL to async if needed
    if settings.database_url.startswith("postgresql://"):
        async_database_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
    elif settings.database_url.startswith("postgresql+psycopg2://"):
        async_database_url = settings.database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    else:
        async_database_url = settings.database_url
    
    engine = create_async_engine(
        async_database_url,
        echo=settings.debug,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
else:
    engine = None

# Session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
) if engine else None


class Base(DeclarativeBase):
    """Base class for all database models."""
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


async def init_db() -> None:
    """Initialize database - create all tables."""
    if engine is None:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    if engine is not None:
        await engine.dispose()


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if async_session_factory is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL environment variable.")
    
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session."""
    async with get_db() as session:
        yield session


# Sync engine for migrations (uses psycopg2)
def get_sync_engine():
    """Get synchronous engine for migrations."""
    if not settings.database_url:
        return None
    
    if settings.database_url.startswith("postgresql+asyncpg://"):
        sync_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    elif settings.database_url.startswith("postgresql+asyncpg://"):
        sync_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    elif settings.database_url.startswith("postgresql://"):
        sync_url = settings.database_url
    else:
        sync_url = settings.database_url
    
    from sqlalchemy import create_engine
    return create_engine(sync_url, pool_pre_ping=True)


# Sync session for migrations
from sqlalchemy.orm import sessionmaker

def get_sync_session():
    """Get synchronous session for migrations."""
    engine = get_sync_engine()
    if engine is None:
        return None
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def create_all_tables() -> None:
    """Create all tables (sync version for scripts)."""
    engine = get_sync_engine()
    if engine is not None:
        Base.metadata.create_all(bind=engine)


def drop_all_tables() -> None:
    """Drop all tables (sync version for scripts)."""
    engine = get_sync_engine()
    if engine is not None:
        Base.metadata.drop_all(bind=engine)