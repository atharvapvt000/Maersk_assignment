"""Entrypoint: `python -m app`. Configures JSON logging before uvicorn starts."""

import os

import uvicorn

from app.config import Settings
from app.logging_setup import configure_logging
from app.main import create_app


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
