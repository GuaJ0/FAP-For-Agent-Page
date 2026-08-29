"""Research Agent implementation and its supporting contracts."""

from agent.research.agent import (
    DuplicateHypothesisError,
    LLMResearchAgent,
    ResearchAgentError,
    ResearchInputError,
    ResearchOutputError,
    ResearchUsageLog,
)

from agent.research.citations import (
    CitationClaim,
    CitationRecord,
    CitationSource,
    CitationValidationError,
    CompositeCitationSource,
    JsonCitationCatalog,
    ResolvedCitation,
    validate_proposal_citations,
)
from agent.research.context import (
    IncumbentSummary,
    IterationSummary,
    ResearchContext,
    build_research_context,
)
from agent.research.schemas import (
    EvaluationPlan,
    EvidenceReference,
    FeasibilityAssessment,
    ImplementationPlan,
    ProposalValidationError,
    Rationale,
    ResearchProposal,
)

__all__ = [
    "CitationClaim",
    "CitationRecord",
    "CitationSource",
    "CitationValidationError",
    "CompositeCitationSource",
    "DuplicateHypothesisError",
    "EvaluationPlan",
    "EvidenceReference",
    "FeasibilityAssessment",
    "ImplementationPlan",
    "IncumbentSummary",
    "IterationSummary",
    "JsonCitationCatalog",
    "LLMResearchAgent",
    "ProposalValidationError",
    "Rationale",
    "ResearchAgentError",
    "ResearchContext",
    "ResearchInputError",
    "ResearchOutputError",
    "ResearchProposal",
    "ResearchUsageLog",
    "ResolvedCitation",
    "build_research_context",
    "validate_proposal_citations",
]
