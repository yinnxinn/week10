from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


def create_app() -> FastAPI:
    settings.ensure_directories()
    app = FastAPI(title="Knowledge Base Demo", version="0.1.0")
    app.include_router(router, prefix="/api")
    return app


app = create_app()
