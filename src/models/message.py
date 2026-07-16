from datetime import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.session import Base


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
