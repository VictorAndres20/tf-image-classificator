from pydantic import BaseModel


class Request(BaseModel):
    image: str
