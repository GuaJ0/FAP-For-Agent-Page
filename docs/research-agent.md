# Research Agent (`agent/research/`)

The Research Agent is Agent 1 in the three-agent experimentation loop. Its
responsibility is to choose one concrete, literature-grounded experiment from
the current validation evidence and hand an implementation-ready proposal to
the Coding Agent. It does not write model code, run experiments, judge results,
route retries, or decide convergence.

## Inputs

Both Research modes implement the existing protocol:

```python
propose(history: list[RunRecord]) -> Idea
```

`build_research_context()` derives a validation-only view containing:

- the highest-scoring accepted incumbent and its iteration;
- every recorded hypothesis, status, decision, validation aggregate, seed
  count, delta, evaluator event, and wall-clock cost;
- remaining iteration and wall-clock budgets; and
- the configured minimum meaningful validation improvement.

The context builder deliberately does not read `logs/quarantine/`,
`docs/results.md`, or `solution/ideas.md`, because those locations contain or
may contain split-specific information outside the Research Agent boundary.
Both modes fail closed if hidden-test markers appear in agent-facing history.

## Output contract

Internally, every proposal is a strict, versioned `ResearchProposal` from
`agent/research/schemas.py`. It includes the hypothesis, mechanism, supported
evidence, implementation steps, hyperparameters, invariants, feasibility,
incumbent-relative evaluation plan, failure interpretation, and risks.

The proposal is rendered through `ResearchProposal.to_handoff_text()` and
returned using the existing shared type:

```python
Idea(
    hypothesis="[RESEARCH_PROPOSAL v1]\n...",
    parent_iteration=current_accepted_incumbent,
)
```

Both LLM and offline modes therefore produce the same deterministic Coding
handoff format. Success is always measured relative to the selected accepted
parent. Iteration 0 is not special unless it is actually the current incumbent.

## LLM mode

`LLMResearchAgent` receives the existing `LLMClient` abstraction by dependency
injection. It does not own an API client or make assumptions about the provider.

The agent:

1. builds and leak-checks Research context;
2. retrieves an evidence packet from its configured `CitationSource`;
3. requests one strict JSON `ResearchProposal`;
4. validates schema, parent iteration, citations, claims, and duplication; and
5. returns the shared handoff text.

Malformed or invalid output gets at most one repair call. The repair prompt
contains the original response and deterministic validation error. A second
failure raises `ResearchOutputError`; malformed output is never accepted
silently. Research token counts and estimated cost are written independently to
`logs/research_agent_usage.jsonl`; Coding Agent accounting is not modified.

## Offline mode

`OfflineResearchAgent` makes no model or network calls. It selects from a
ranked backlog of complete `ResearchProposal` templates covering contained
ranking-objective variations, behavior history, multi-task learning,
watch-time supervision, and interaction modelling.

It is not a round-robin string iterator. On every call it:

- rebuilds the current context;
- filters entries that do not fit remaining iterations or estimated wall time;
- uses research rank when the budget is comfortable and cost-first ordering
  when the budget is tight;
- skips unchanged proposals already reverted or abandoned;
- validates the selected proposal through the same schema and citation source
  as LLM mode; and
- parents the result to the current accepted incumbent.

When no untried, supported proposal fits, it raises
`OfflineBacklogExhausted` instead of cycling back to a dead end.

## Citation policy

The bundled `references.json` is the safe initial evidence source. Proposals
cite both a `citation_id` and one of that record's supported `claim_id` values;
free-form or invented citation claims are rejected.

Research code depends on the `CitationSource` protocol rather than directly on
the JSON file. `CompositeCitationSource` can combine the curated catalog with a
future literature-retrieval or web-research provider without changing proposal
schemas, validation, or either Research Agent mode.

## Duplicate and dead-end handling

Reverted and abandoned hypotheses are compared against each new proposal. A
legacy free-text hypothesis must state a material variation in its new
hypothesis wording. For structured handoffs, changed implementation steps or
hyperparameters can establish the variation. Merely changing an ID or parent
iteration does not.

The offline backlog intentionally contains distinct follow-ups—for example, a
hybrid pointwise/pairwise loss and a GAUC-weighted sampler are not treated as a
rerun of pure BPR because they test different mechanisms.

## Feasibility and dependencies

NumPy is the current starter implementation, not an official challenge limit.
Research proposals may recommend open-source ML libraries when justified. The
proposal must explicitly state dependency, installation, hardware, memory,
runtime, complexity, and remaining-budget implications. The Coding and Sandbox
owners decide whether those dependencies can be installed and executed.

## Limitations

- The offline backlog is finite and intentionally conservative.
- The bundled literature catalog is small; broader retrieval is a future
  `CitationSource`, not part of the current implementation.
- Evaluator interpretation is currently limited to commentary already present
  in `RunRecord.events`.
- Duplicate detection is deterministic text/implementation comparison, not a
  semantic model.
- Research context cannot recover facts that were never written to the shared
  validation history.

## Ownership boundaries

The Research Agent owns hypothesis selection, rationale, citations,
implementation guidance, and the Research handoff.

It does **not** own:

- Coding Agent generation, repair, dependencies, templates, or smoke tests;
- Evaluator execution, scoring, diagnostics, acceptance/reversion, or hidden
  test quarantine;
- orchestrator routing, retries, convergence, state, checkpoint selection, or
  human escalation; or
- wiring a particular Research implementation into `scripts/run_loop.py`.

Those integrations must remain with the corresponding teammates.
