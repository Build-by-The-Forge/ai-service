from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.tenant import set_school_id


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        school_id = request.headers.get("X-School-ID")

        if school_id:
            set_school_id(int(school_id))

        response = await call_next(request)
        return response
