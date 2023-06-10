from enum import Enum
from pydantic import BaseModel, Field
import uuid


class SpeakerEnum(str, Enum):
    USER = "USER"
    AGENT = "AGENT"


def generate_uuid():
    return str(uuid.uuid4())


class Message(BaseModel):
    timestamp: float
    message: str
    uuid: str = Field(default_factory=generate_uuid)
    speaker: SpeakerEnum
    embeddings: list[float]
