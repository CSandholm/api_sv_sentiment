from pydantic import BaseModel


class Request(BaseModel):
    input: str


class Response(BaseModel):
    sentiment: int
