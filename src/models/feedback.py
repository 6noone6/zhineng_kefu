from datetime import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.session import Base


class MessageFeedback(Base):
    __tablename__ = "message_feedback"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sessions.id"), index=True)
    message_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    rating: Mapped[int] = mapped_column(Integer)  # 1 = negative, 5 = positive
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
