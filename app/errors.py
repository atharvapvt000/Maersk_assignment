"""Application errors and the single JSON error envelope used for every non-2xx response.

{"error": {"code": "SHIPMENT_NOT_FOUND", "message": "...", "request_id": "...", "details": ...}}
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_setup import request_id_var


class AppError(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class BadRequestError(AppError):
    status_code = 400
    code = "BAD_REQUEST"


class UnprocessableError(AppError):
    status_code = 422
    code = "VALIDATION_ERROR"


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details is not None:
        body["details"] = details
    return {"error": body}


def error_response(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=jsonable_encoder(error_body(code, message, details))
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return error_response(422, "VALIDATION_ERROR", "Request validation failed", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Framework-level errors: unknown path (404), wrong method (405), etc.
        code = HTTPStatus(exc.status_code).name
        return error_response(exc.status_code, code, str(exc.detail))
