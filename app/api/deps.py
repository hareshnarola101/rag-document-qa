from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.container import Services


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


async def get_session(
    services: Annotated[Services, Depends(get_services)],
) -> AsyncIterator[AsyncSession]:
    async with services.session_factory() as session:
        yield session


ServicesDep = Annotated[Services, Depends(get_services)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
