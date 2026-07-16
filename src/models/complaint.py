from datetime import datetime
import uuid

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.session import Base


class Complaint(Base):
    __tablename__ = "complaints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    details: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="Received")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
