from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str = ""
    score: float = 0.0
    chunk_id: str = ""
    lang: str = ""
