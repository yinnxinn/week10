from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

# Set default mirror for HuggingFace if not set
if "HF_ENDPOINT" not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def create_app() -> FastAPI:
    settings.ensure_directories()
    
    app = FastAPI(
        title=settings.api_title, 
        version=settings.api_version
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")
    return app


app = create_app()
