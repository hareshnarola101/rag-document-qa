from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.container import Services, build_services
from app.core.config import Settings
from app.main import create_app

PROJECT_DIR = Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def migrated_database() -> None:
    """Run the real Alembic migrations once per test session, from a clean slate."""
    from tests.conftest import TEST_DATABASE_URL

    config = Config(str(PROJECT_DIR / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    config.attributes["configure_logger"] = False
    config.attributes["embedding_dim"] = 256  # must match the `settings` fixture
    command.downgrade(config, "base")
    command.upgrade(config, "head")


@pytest.fixture
async def services(migrated_database: None, settings: Settings) -> AsyncIterator[Services]:
    services = build_services(settings)
    async with services.engine.begin() as connection:
        await connection.execute(text("TRUNCATE documents CASCADE"))
    yield services
    await services.aclose()


@pytest.fixture
async def client(services: Services) -> AsyncIterator[AsyncClient]:
    app = create_app(services=services)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http
