from pydantic import BaseModel
from typing import Optional


class Response(BaseModel):
    code: int
    ok: bool
    msg: str
    error: str
    data: Optional[dict] = None
