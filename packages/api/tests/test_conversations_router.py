"""groundwork_api.routers.conversations tests, over real HTTP: create a
conversation against a workspace, list turns, ask a question end to end
(routing into groundwork_api.chat.ask, already covered in detail by
test_chat.py; this file's job is proving the HTTP boundary itself, status
codes, request validation, and JSON shape, not re-deriving ask()'s own
retrieval and verification behavior), and fetch one turn's detail.

Seeded chunks are embedded with the real, process wide get_embedder(),
not a hand picked vector: the router gives ask() no way to inject a stub
embedder (there is no such parameter on the HTTP boundary, by design,
since a real deployment never should), so a chunk stored under a made up
vector would be judged against a genuinely computed query vector at ask
time and, measured directly, lands at 0.0 cosine similarity against this
environment's real tfidf hashing backend, an orthogonal, not merely
imperfect, mismatch. Embedding the real chunk text here is what a real
upload does, and it is the only way this test can ask a real question
and reach real generation rather than the relevance gate's refusal.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from groundwork_api.db import to_async_url
from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Conversation,
    Document,
    ExtractionMethod,
    Turn,
    Workspace,
)
from groundwork_core.config import Settings
from groundwork_core.ids import new_id
from groundwork_retrieve.embeddings import get_embedder

pytestmark = pytest.mark.requires_postgres


async def _seed_workspace(session: AsyncSession, *, name: str = "conv-test") -> Workspace:
    workspace = Workspace(name=name)
    session.add(workspace)
    await session.commit()
    return workspace


async def _seed_workspace_with_chunk(session: AsyncSession) -> tuple[Workspace, Chunk]:
    workspace = await _seed_workspace(session, name="conv-chunk-test")
    document = Document(
        workspace_id=workspace.id,
        filename="doc.pdf",
        sha256=new_id().hex * 2,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    session.add(document)
    await session.flush()
    text = "Sessions run fifty minutes."
    [embedding] = get_embedder().embed([text])
    chunk = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        embedding=embedding,
        embedding_backend="tfidf",
    )
    session.add(chunk)
    await session.commit()
    return workspace, chunk


async def _seed_conversation(session: AsyncSession, workspace: Workspace) -> Conversation:
    conversation = Conversation(workspace_id=workspace.id)
    session.add(conversation)
    await session.commit()
    return conversation


async def test_create_conversation_404s_when_the_workspace_does_not_exist(
    client: AsyncClient,
) -> None:
    response = await client.post(f"/workspaces/{new_id()}/conversations")
    assert response.status_code == 404


async def test_create_conversation_against_a_real_workspace(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace = await _seed_workspace(db_session)

    response = await client.post(f"/workspaces/{workspace.id}/conversations")

    assert response.status_code == 201
    assert response.json()["workspace_id"] == str(workspace.id)


async def test_list_turns_404s_when_the_conversation_does_not_exist(client: AsyncClient) -> None:
    response = await client.get(f"/conversations/{new_id()}/turns")
    assert response.status_code == 404


async def test_list_turns_is_empty_for_a_fresh_conversation(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace = await _seed_workspace(db_session)
    conversation = await _seed_conversation(db_session, workspace)

    response = await client.get(f"/conversations/{conversation.id}/turns")

    assert response.status_code == 200
    assert response.json() == []


async def test_ask_question_runs_the_full_pipeline_over_http(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace, chunk = await _seed_workspace_with_chunk(db_session)
    conversation = await _seed_conversation(db_session, workspace)

    response = await client.post(
        f"/conversations/{conversation.id}/turns",
        json={"question": "How long do sessions run?"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["conversation_id"] == str(conversation.id)
    assert body["workspace_id"] == str(workspace.id)
    assert body["retrieved_chunk_ids"] == [str(chunk.id)]
    assert "fifty minutes" in body["answer"]
    assert len(body["citation_verifications"]) == 1
    assert body["citation_verifications"][0]["chunk_id"] == str(chunk.id)


async def test_ask_question_404s_when_the_conversation_does_not_exist(
    client: AsyncClient,
) -> None:
    response = await client.post(f"/conversations/{new_id()}/turns", json={"question": "anything"})
    assert response.status_code == 404


async def test_ask_question_rejects_an_empty_question(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace = await _seed_workspace(db_session)
    conversation = await _seed_conversation(db_session, workspace)

    response = await client.post(f"/conversations/{conversation.id}/turns", json={"question": ""})

    assert response.status_code == 422


async def test_list_turns_returns_a_turn_after_asking(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace, _ = await _seed_workspace_with_chunk(db_session)
    conversation = await _seed_conversation(db_session, workspace)
    await client.post(
        f"/conversations/{conversation.id}/turns", json={"question": "How long do sessions run?"}
    )

    response = await client.get(f"/conversations/{conversation.id}/turns")

    assert response.status_code == 200
    assert len(response.json()) == 1


async def test_get_turn_404s_when_it_does_not_exist(client: AsyncClient) -> None:
    response = await client.get(f"/turns/{new_id()}")
    assert response.status_code == 404


async def test_get_turn_returns_the_full_detail(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace, _ = await _seed_workspace_with_chunk(db_session)
    conversation = await _seed_conversation(db_session, workspace)
    ask_response = await client.post(
        f"/conversations/{conversation.id}/turns", json={"question": "How long do sessions run?"}
    )
    turn_id = ask_response.json()["id"]

    response = await client.get(f"/turns/{turn_id}")

    assert response.status_code == 200
    assert response.json()["id"] == turn_id
    assert response.json()["question"] == "How long do sessions run?"


async def test_ask_question_actually_commits_so_a_separate_connection_can_see_it(
    client: AsyncClient, db_session: AsyncSession, test_settings: Settings
) -> None:
    """A real, previously undetected bug, found only by driving a real
    browser against a real running server, not by any test, including
    the one directly above this: ask_question's own router handler
    called chat.ask(), which only flushes the new Turn by design (see
    that route's own docstring), and never committed. Every request
    this file makes goes through the client fixture, whose get_session
    override hands out this one test's single db_session for every
    call (see conftest.py's client fixture docstring), so a flush from
    the POST above was already visible to a "later" GET on the same
    shared session regardless of whether a commit ever happened, the
    same reason test_get_turn_returns_the_full_detail passes either
    way. A real request in the real running app gets its own fresh
    session per deps.get_session() call, so a flush with no commit
    vanished the instant that request's session closed, and a turn a
    real user asked would 404 on every subsequent look, /trace
    included.

    This test is the one in this file that can actually tell a flush
    apart from a commit: it opens a second, fully independent
    connection to the same database, bypassing client's shared session
    override entirely, and confirms the turn is visible there too, not
    only through the same session that created it.
    """
    workspace, _ = await _seed_workspace_with_chunk(db_session)
    conversation = await _seed_conversation(db_session, workspace)

    response = await client.post(
        f"/conversations/{conversation.id}/turns",
        json={"question": "How long do sessions run?"},
    )
    assert response.status_code == 201
    turn_id = UUID(response.json()["id"])

    engine = create_async_engine(to_async_url(test_settings.database_url), pool_pre_ping=True)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as independent_session:
            fetched = await independent_session.get(Turn, turn_id)
    finally:
        await engine.dispose()

    assert fetched is not None, (
        "the turn was flushed but never committed by the request that created it, "
        "so a genuinely separate connection cannot see it"
    )
    assert fetched.question == "How long do sessions run?"
