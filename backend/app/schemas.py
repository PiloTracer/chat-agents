from pydantic import BaseModel, Field
from typing import List, Optional

class AgentCreate(BaseModel):
    slug: str
    title: str
    description: str

class AgentOut(AgentCreate):
    id: int

class DocumentOut(BaseModel):
    id: int
    filename: str
    agent_slug: str
    content_type: str

class AskRequest(BaseModel):
    question: str
    agent_slug: Optional[str] = None  # if None, router decides
    top_k: int = 8

class AskAnswer(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)  # [{source_ref, text}]
