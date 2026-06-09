"""
RunSpec and QueueItemSpec — JSON-serializable structures for stateless orchestration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class QueueItemSpec:
    corpus_family: str
    jurisdiction_or_issuer: str
    scope_key: str = ""
    queue_order: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunSpec:
    run_id: str = ""
    original_prompt: str = ""
    run_mode: str = "single"
    corpus_family: str = ""
    clarification_needed: bool = False
    clarification_questions: list[str] = field(default_factory=list)
    jurisdictions_or_issuers: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    routing_strategy: str = "catalog_first"
    answer_contract: str = "evidence_with_citations"
    queue_items: list[QueueItemSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["queue_items"] = [q.to_dict() if isinstance(q, QueueItemSpec) else q for q in self.queue_items]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> RunSpec:
        qi = data.pop("queue_items", [])
        spec = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        spec.queue_items = [
            QueueItemSpec(**q) if isinstance(q, dict) else q for q in qi
        ]
        return spec
