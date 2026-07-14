from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.security import verify_api_key

router = APIRouter(tags=["inference"])


class InferenceRequest(BaseModel):
    messages: list[dict[str, str]] = Field(..., min_length=1)
    max_new_tokens: int | None = None


class InferenceResponse(BaseModel):
    answer: str


@router.post("/inference", response_model=InferenceResponse)
async def qwen_inference(
    request: Request,
    body: InferenceRequest,
    _: Annotated[str, Depends(verify_api_key)] = ...,
):
    qwen = getattr(request.app.state, "qwen_service", None)
    if qwen is None:
        raise HTTPException(status_code=503, detail="Local Qwen model is not loaded")

    try:
        answer = await qwen.generate_from_messages(body.messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return InferenceResponse(answer=answer)
