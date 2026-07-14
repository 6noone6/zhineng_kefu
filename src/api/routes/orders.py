from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.core.rate_limit import limiter
from src.db import get_db
from src.models.user import User
from src.tools import order

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("/me/orders")
@limiter.limit("30/minute")
async def my_orders(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await order.list_my_orders(user_id=user.id, email=user.email)
    return result.to_dict()
