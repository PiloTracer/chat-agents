from sqlalchemy import (
    Column, Integer, String, DateTime, Text, ForeignKey, Boolean, LargeBinary, UniqueConstraint
)
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from app.db import Base
from datetime import datetime
from sqlalchemy.types import UserDefinedType

class Vector(UserDefinedType):
    cache_ok = True  # allow SQLAlchemy to cache query plans

    def get_col_spec(self):
        return "vector(1536)"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            return "[" + ",".join(str(x) for x in value) + "]"
        return process

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(16), default="admin")

class Agent(Base):
    __tablename__ = "agents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # e.g., local-regs
    title: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)

class AgentACL(Base):
    __tablename__ = "agent_acl"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(120), index=True)
    agent_slug: Mapped[str] = mapped_column(String(64), index=True)
    __table_args__ = (UniqueConstraint('username', 'agent_slug', name='uix_user_agent'),)

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_slug: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(255))
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

class Chunk(Base):
    __tablename__ = "chunks"
    from pgvector.sqlalchemy import Vector

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    agent_slug: Mapped[str] = mapped_column(String(64))  # <-- no index=True here
    text: Mapped[str] = mapped_column(Text)
    ord: Mapped[int] = mapped_column(Integer, default=0)
    embedding = Column(Vector(1536), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255))  # filename#page or slide
