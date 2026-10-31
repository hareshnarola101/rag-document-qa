from fastapi import APIRouter

from app.api.deps import ServicesDep, SessionDep
from app.qa.service import answer_question
from app.schemas import AgentResponse, AskRequest, AskResponse, SearchHit, SearchRequest

router = APIRouter(tags=["question answering"])


@router.post("/search", response_model=list[SearchHit])
async def search(
    body: SearchRequest, services: ServicesDep, session: SessionDep
) -> list[SearchHit]:
    """Hybrid retrieval only, no model call. Handy for debugging what /ask will see."""
    chunks = await services.retriever(session).search(
        body.query, body.top_k or services.settings.default_top_k, body.document_ids
    )
    return [SearchHit.model_validate(chunk, from_attributes=True) for chunk in chunks]


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, services: ServicesDep, session: SessionDep) -> AskResponse:
    return await answer_question(
        services.retriever(session),
        services.llm,
        body.question,
        top_k=body.top_k or services.settings.default_top_k,
        document_ids=body.document_ids,
    )


@router.post("/agent/ask", response_model=AgentResponse)
async def agent_ask(body: AskRequest, services: ServicesDep, session: SessionDep) -> AgentResponse:
    agent = services.agent(session, top_k=body.top_k, document_ids=body.document_ids)
    return await agent.run(body.question)
