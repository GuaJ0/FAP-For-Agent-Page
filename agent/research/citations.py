"""Pluggable, claim-level evidence sources for Research proposals.

The first implementation uses the bundled JSON catalog.  The Research Agent
depends only on ``CitationSource`` so a future literature-search or web-backed
source can be composed in without changing proposal parsing or prompts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

from agent.research.schemas import ResearchProposal

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "references.json"


class CitationValidationError(ValueError):
    """A proposal cites an unknown source or unsupported claim."""


@dataclass(frozen=True)
class CitationClaim:
    claim_id: str
    text: str


@dataclass(frozen=True)
class CitationRecord:
    citation_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    venue: str
    url: str
    claims: tuple[CitationClaim, ...]
    tags: tuple[str, ...] = ()

    def claim(self, claim_id: str) -> Optional[CitationClaim]:
        return next((claim for claim in self.claims if claim.claim_id == claim_id), None)


class CitationSource(Protocol):
    """Minimal boundary shared by local and future retrieval-backed sources."""

    def get(self, citation_id: str) -> Optional[CitationRecord]:
        ...

    def search(self, query: str, limit: int = 10) -> list[CitationRecord]:
        ...


class JsonCitationCatalog:
    def __init__(self, path: Path = DEFAULT_CATALOG_PATH):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or not isinstance(data.get("references"), list):
            raise CitationValidationError(f"invalid citation catalog schema in {self.path}")
        self._records: dict[str, CitationRecord] = {}
        for raw in data["references"]:
            record = CitationRecord(
                citation_id=raw["citation_id"],
                title=raw["title"],
                authors=tuple(raw["authors"]),
                year=int(raw["year"]),
                venue=raw["venue"],
                url=raw["url"],
                claims=tuple(CitationClaim(c["claim_id"], c["text"]) for c in raw["claims"]),
                tags=tuple(raw.get("tags", [])),
            )
            if record.citation_id in self._records:
                raise CitationValidationError(f"duplicate citation_id {record.citation_id!r}")
            if not record.claims:
                raise CitationValidationError(f"citation {record.citation_id!r} has no claims")
            claim_ids = [claim.claim_id for claim in record.claims]
            if len(claim_ids) != len(set(claim_ids)):
                raise CitationValidationError(f"citation {record.citation_id!r} has duplicate claim IDs")
            self._records[record.citation_id] = record

    def get(self, citation_id: str) -> Optional[CitationRecord]:
        return self._records.get(citation_id)

    def all(self) -> list[CitationRecord]:
        return list(self._records.values())

    def search(self, query: str, limit: int = 10) -> list[CitationRecord]:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not tokens:
            return self.all()[:limit]
        scored = []
        for record in self._records.values():
            haystack = " ".join([
                record.title,
                " ".join(record.authors),
                " ".join(record.tags),
                " ".join(claim.text for claim in record.claims),
            ]).lower()
            score = sum(token in haystack for token in tokens)
            if score:
                scored.append((score, record.year, record))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].citation_id))
        return [record for _, _, record in scored[:limit]]


class CompositeCitationSource:
    """Query several sources in priority order, deduplicating by citation ID."""

    def __init__(self, sources: Sequence[CitationSource]):
        self.sources = tuple(sources)

    def get(self, citation_id: str) -> Optional[CitationRecord]:
        for source in self.sources:
            record = source.get(citation_id)
            if record is not None:
                return record
        return None

    def search(self, query: str, limit: int = 10) -> list[CitationRecord]:
        found: dict[str, CitationRecord] = {}
        for source in self.sources:
            for record in source.search(query, limit=limit):
                found.setdefault(record.citation_id, record)
                if len(found) >= limit:
                    return list(found.values())
        return list(found.values())


@dataclass(frozen=True)
class ResolvedCitation:
    record: CitationRecord
    claim: CitationClaim
    application: str


def validate_proposal_citations(
    proposal: ResearchProposal,
    source: CitationSource,
) -> tuple[ResolvedCitation, ...]:
    resolved = []
    for evidence in proposal.rationale.evidence:
        record = source.get(evidence.citation_id)
        if record is None:
            raise CitationValidationError(f"unknown citation_id {evidence.citation_id!r}")
        claim = record.claim(evidence.claim_id)
        if claim is None:
            raise CitationValidationError(
                f"citation {evidence.citation_id!r} does not support claim_id {evidence.claim_id!r}"
            )
        resolved.append(ResolvedCitation(record, claim, evidence.application))
    return tuple(resolved)
