"""Conversations and turns: create a conversation against a workspace, ask
a question (the one call that runs groundwork_api.chat.ask end to end),
list a conversation's turns for the chat scrollback, and fetch one turn's
full detail for the /trace view.

No workspace_id parameter appears on the ask route by construction,
matching chat.ask's own module docstring: a caller supplies a
conversation_id, this router loads it, and every downstream call is
scoped to whatever workspace that conversation actually belongs to, never
to a workspace_id a request body could name.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.chat import ConversationNotFoundError, ask
from groundwork_api.deps import get_session
from groundwork_api.models import Conversation, Turn, Workspace
from groundwork_api.schemas import (
    AskRequest,
    CitationVerificationOut,
    ClaimOut,
    ConversationOut,
    TurnOut,
)

router = APIRouter(tags=["conversations"])


def _turn_out(turn: Turn) -> TurnOut:
    """Turn.claims and Turn.citation_verifications are stored as a plain
    JSON column (list[dict[str, object]]), the only honest static type for
    a JSON column. model_validate is the precise tool for turning each
    loosely typed dict back into the strictly typed ClaimOut or
    CitationVerificationOut schema this endpoint promises its caller,
    rather than a cast or a blanket type: ignore standing in for real
    validation.
    """
    return TurnOut(
        id=turn.id,
        conversation_id=turn.conversation_id,
        workspace_id=turn.workspace_id,
        question=turn.question,
        retrieved_chunk_ids=turn.retrieved_chunk_ids,
        reranked=turn.reranked,
        chunking_strategy=turn.chunking_strategy,
        answer=turn.answer,
        claims=[ClaimOut.model_validate(raw) for raw in turn.claims],
        citation_verifications=[
            CitationVerificationOut.model_validate(raw) for raw in turn.citation_verifications
        ],
        extractive_fallback=turn.extractive_fallback,
        judge_scores=turn.judge_scores,
        latency_ms=turn.latency_ms,
        cost_usd=turn.cost_usd,
        created_at=turn.created_at,
    )


@router.post(
    "/workspaces/{workspace_id}/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    workspace_id: UUID, session: AsyncSession = Depends(get_session)
) -> Conversation:
    workspace = await session.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="workspace not found")
    conversation = Conversation(workspace_id=workspace_id)
    session.add(conversation)
    await session.commit()
    return conversation


@router.get(
    "/conversations/{conversation_id}/turns",
    response_model=list[TurnOut],
)
async def list_turns(
    conversation_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[TurnOut]:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found")
    result = await session.execute(
        select(Turn)
        .where(col(Turn.conversation_id) == conversation_id)
        .order_by(col(Turn.created_at))
    )
    return [_turn_out(turn) for turn in result.scalars()]


@router.post(
    "/conversations/{conversation_id}/turns",
    response_model=TurnOut,
    status_code=status.HTTP_201_CREATED,
)
async def ask_question(
    conversation_id: UUID,
    body: AskRequest,
    session: AsyncSession = Depends(get_session),
) -> TurnOut:
    try:
        turn = await ask(
            session,
            conversation_id=conversation_id,
            question=body.question,
            strategy=body.strategy,
            use_reranking=body.use_reranking,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found"
        ) from exc
    return _turn_out(turn)


@router.get("/turns/{turn_id}", response_model=TurnOut)
async def get_turn(turn_id: UUID, session: AsyncSession = Depends(get_session)) -> TurnOut:
    turn = await session.get(Turn, turn_id)
    if turn is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="turn not found")
    return _turn_out(turn)
