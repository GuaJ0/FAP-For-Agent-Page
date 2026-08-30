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
2. summarizes current-run coverage across `features`, `architecture`,
   `objective_sampling`, `optimization_regularization`, and
   `inference_ensemble`;
3. retrieves a compact evidence packet from its configured `CitationSource`;
4. makes one bounded breadth call for exactly the configured number of shallow
   candidate directions (five by default; the absolute supported range remains
   three to eight), each with one bounded `primary_change` that identifies its
   intervention;
5. deterministically hard-filters malformed, unsafe, benchmark-manipulating,
   unsupported, duplicate, ambiguous-stage, and mislabelled-stage candidates;
6. requires at least three genuinely different candidates to survive those
   filters, otherwise invoking the single bounded breadth repair;
7. softly ranks survivors using structural mechanism novelty, relevant
   distinct-source evidence,
   stack coverage, remaining budget, and modest feasibility/upside priors;
8. gives only the selected direction to the existing detailed depth prompt;
9. safety-scans the final proposal and validates that it retains the selected
   stage, mechanism, and supporting evidence; and
10. returns the shared handoff text.

Coverage is a soft preference. An unexplored stage receives a diversity bonus,
while repeated or most-recently explored stages receive a bounded penalty. A
strong repeated-stage candidate can still win when its independently checked
evidence, novelty, and feasibility outweigh that penalty. Self-reported upside,
risk, and cost are weak priors rather than dominant signals; their maximum
combined advantage is smaller than one relevant evidence source. Under a tight
budget, candidate-declared cost receives no second adjustment beyond the
ordinary weak cost prior; objective feasibility signals can be added separately
in the future.
Evidence count is based on distinct citation IDs, capped at two, and a source
receives full ranking credit only when a confident primary mechanism family and
stack stage in its trusted title/claim match the candidate. Copied technical
tokens do not earn that credit. Multiple claims from one source do not inflate
it. Citation identity and claim provenance remain valid even when ranking
relevance is zero; this relevance heuristic is not proof of semantic support.
Only the one winning candidate reaches the Coding Agent.

Malformed or short breadth, or a batch with fewer than three surviving
candidates, gets at most one breadth repair. Valid survivors are retained; the
repair response supplies only fresh replacement IDs, and the combined pool is
passed through the same safety, evidence, mechanism, history-duplicate, and
intra-pool duplicate filters before the three-survivor invariant is checked
again. The depth proposal independently gets at most one schema/alignment/safety
repair. Its repair prompt repeats the binding selected stack stage, primary
change, and deterministically inferred primary family so vague wording can be
corrected without bypassing alignment. Both original and repaired outputs pass
the same safety boundary. Normal operation therefore uses two calls—one cheap
breadth call and one detailed depth call—and the configured default permits at
most four total calls. A second failure in either phase raises
`ResearchOutputError`; malformed output is never accepted silently. Usage rows
distinguish `research_breadth`, `research_breadth_repair`, `research_depth`, and
`research_depth_repair`. Research token counts and estimated cost remain in
`logs/research_agent_usage.jsonl`; Coding Agent accounting is not modified.

Generated breadth and depth fields share one safety boundary. It combines the
existing encoded-text/injection scanner with a normalized, bounded
action--target policy for forbidden test/competition evaluation feedback,
leaderboard feedback, and scorer manipulation. Unicode, punctuation,
hyphenation, common inflections, and scorer/evaluator synonyms are normalized.
Deterministic clause boundaries keep negation and preservation local: statements
such as "do not use final competition results" and "preserve evaluate.py
exactly" are allowed, but they cannot neutralize a later unsafe action in
another clause or field.

The bundled catalog remains the only active literature source. This
breadth-then-depth flow does not connect the Phase 4A retrieval foundations and
does not add web search, scholarly APIs, or full-text retrieval.

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
- Stack classification uses deterministic whole-token, presence-based matching
  through an internal structural fingerprint containing stack stage, primary
  family, primary intervention tags, and secondary tags. Repetition cannot vote
  one incompatible same-stage family above another. For new breadth candidates,
  `primary_change` is authoritative; optimizer, regularizer, sampler, and
  training details enrich secondary tags without overriding a clear DeepFM or
  BPR primary mechanism. Optimization/regularization remains primary when it is
  itself the stated intervention. Mixed core stages, conflicting same-stage
  families, unknown primary changes, and confident declaration/mechanism
  mismatches are rejected.
- Depth alignment infers title, hypothesis, target components, and individual
  implementation steps independently. Any ambiguous intervention field or
  confident core-family conflict fails closed; keep-constant controls, risks,
  evidence, hyperparameters, and evaluation prose do not vote on the family.
- Structured historical signatures fall through unknown hypothesis text to the
  title and implementation-change sections. Coherent ancillary details are
  retained as tags, while confident primary-family conflicts remain ambiguous.
- Breadth ranking uses intentionally coarse, documented weights. It improves
  exploration discipline but is not a learned estimate of experiment value.
  Structural signature novelty is the main novelty signal; raw lexical
  dissimilarity contributes only a small bounded amount.
- Evaluator interpretation is currently limited to commentary already present
  in `RunRecord.events`.
- Duplicate detection primarily compares canonical structural fingerprints and
  extracts intervention-bearing fields from structured history while excluding
  evidence, risks, constants, hyperparameters, and metric boilerplate. Renaming
  DeepFM or BPR does not create novelty. A conservative same-family proposal can
  remain eligible only when `primary_change` supplies a materially new primary
  intervention tag; ancillary prose cannot manufacture that variation. Lexical
  similarity is a small fallback for unknown signatures. Tied duplicate
  selection uses canonical structural/semantic content, not candidate ID or
  model output order.
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
