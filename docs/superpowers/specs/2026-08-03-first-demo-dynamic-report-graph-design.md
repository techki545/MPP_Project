# Dynamic First-Demo Report and Evidence Graph Design

## Goal

Restore the first Graph RAG demo's report logic and evidence-graph experience while
using the production knowledge base and dynamically retrieved evidence for every
clinical question.

The implementation must not hard-code the steroid/SMPP answer or its eight demo
documents. The first demo defines the presentation contract; each query supplies
new sources, claims, relations, reasoning, and conclusions.

## User-Facing Contract

Every successful query returns two visible report parts:

1. A collapsible evidence-reasoning process following the first demo's six-step
   sequence.
2. A conclusion-first synthesis that explains the evidence chain, chronology,
   safety, applicability boundaries, and evidence gaps.

The query evidence graph follows the first demo exactly:

- evidence documents are the visible nodes;
- evidence types form horizontal lanes in pyramid order;
- nodes show evidence type, year, and a shortened title;
- all accepted relations are visible by default;
- colored curved arrows show relation direction and a Chinese relation label;
- selecting a node highlights its direct neighbors and relations and mutes unrelated
  content;
- selecting a node also opens the corresponding evidence detail;
- long edges use automatic routing lanes above or below nodes to reduce crossings,
  without changing the first demo's interaction model.

The graph contains at most ten core documents chosen for synthesis readability.
All retrieved sources remain available in the source list and document detail view.

## Dynamic Evidence Pipeline

### 1. Question and retrieval

The existing query parser, filters, lexical/vector retrieval, evidence pyramid
ranking, chronology, and full-text signals remain the retrieval foundation. The
system selects a diverse core set that prioritizes:

1. relevant guidelines;
2. systematic reviews and meta-analyses;
3. randomized controlled trials;
4. observational studies;
5. narrative reviews;
6. case reports and other boundary evidence.

No evidence class is fabricated to fill an empty lane.

### 2. Source-grounded structured claims

The model may select only server-supplied Quote IDs. A selected quote resolves to an
exact document, chunk, and source text on the server. Model output uses closed fields
for question aspect, direction, population, intervention, comparator, outcome,
safety role, and limitations.

The server validates source IDs, quote IDs, numbers, doses, units, negation, and
field values. Unsupported optional details are discarded or downgraded to
`uncertain`; the source quote itself remains available for traceability.

### 3. Document-level relation projection

Validated claims are projected back to their source documents. The graph supports
the first demo's relation vocabulary:

- `supports`: evidence points in the same direction as a higher-level source;
- `supplements`: evidence adds a different outcome, population, safety detail, or
  real-world context without overturning the target;
- `updates`: newer evidence directly addresses a dated source's stated gap or a
  specific unresolved detail;
- `confirms`: newer or independent evidence agrees with a comparable earlier claim;
- `conflicts`: comparable claims have opposed directions;
- `cautions`: safety or boundary evidence limits application of another source.

Every edge carries source claim IDs and a concise rationale. Publication year alone
cannot create an `updates` edge. A lower-level source cannot automatically overturn
a higher-level source.

## Six-Step Reasoning Report

The reasoning section always uses these first-demo stages:

1. **Retrieve and inventory evidence**: summarize PICO, evidence counts, and the
   publication range.
2. **Review guidelines first**: establish the highest-level direction and identify
   explicit gaps or evidence cutoffs.
3. **Review systematic reviews**: test agreement with guidelines and add synthesized
   effect or safety detail.
4. **Focus on key RCTs**: determine whether newer direct evidence confirms, conflicts
   with, or locally updates an earlier conclusion.
5. **Use lower-level evidence for supplementation**: add real-world effects, safety
   signals, rare complications, and applicability boundaries without treating them
   as higher-level proof.
6. **Synthesize the judgment**: summarize agreement, conflicts, updates, limitations,
   and remaining uncertainty.

If a level is absent, its step explicitly states that no matching evidence was
retrieved and explains the consequence for certainty. The report does not invent a
source to preserve visual symmetry.

## Final Synthesis

The final answer begins with the conclusion and then explains:

1. the highest-level evidence chain supporting or opposing it;
2. any newer evidence that updates a narrower detail;
3. clinically relevant safety findings;
4. the population and scenario to which the conclusion applies;
5. important exclusions and unresolved evidence gaps.

Each substantive claim has an adjacent numbered citation. Quantitative values must
appear in the cited source text. The model may improve Chinese readability, but an
unsupported sentence is removed or replaced with a validated claim before display.

## Deterministic Recovery

Model timeout, malformed JSON, invalid citations, or rejected prose must not leave
the report or graph blank. The recovery path uses the same six-step structure and
document-level graph, generated from validated claims and relations. It explicitly
marks unresolved directions and does not make a treatment recommendation when the
available evidence does not support one.

## Frontend Behavior

The current technical question/document/claim/outcome graph is replaced in the
primary result view by a first-demo-compatible document graph. The production API
may retain claim-level provenance internally, but those technical nodes are not
shown in the main query graph.

The SVG layout is dynamic:

- lanes are ordered by evidence hierarchy;
- each lane sizes vertically to its document count;
- node dimensions remain stable;
- curved edges route through assigned upper or lower channels and avoid node boxes;
- labels sit on their associated path and use a background stroke for readability;
- horizontal scrolling is allowed when all six evidence types are present;
- mobile preserves readable node dimensions rather than shrinking text.

## Failure and Boundary Cases

- No guideline or RCT: show the empty evidence level and lower confidence.
- Conflicting evidence: show both sides and do not force consensus.
- Only weak evidence: return an uncertainty conclusion rather than a recommendation.
- Missing year: exclude the source from chronology-based `updates` logic.
- Excess graph sources: display the ten core sources and retain all sources below.
- Unsupported statistic or citation: reject the assertion while keeping the rest of
  the validated report.
- Embedding unavailable: allow lexical retrieval and clearly retain the degraded-mode
  status.

## Verification

Implementation is complete only after:

- unit tests cover claim normalization, all six document relations, chronology, and
  missing-level reasoning;
- API tests verify the six-step and document-graph response contract;
- multiple distinct clinical questions produce different nodes, relations, and
  conclusions;
- Playwright tests verify desktop and mobile rendering, horizontal overflow, node
  selection, neighbor highlighting, relation labels, and evidence-detail linkage;
- screenshots confirm that node text and relation labels do not overlap incoherently;
- the full test suite passes;
- one live DeepSeek query returns a non-empty six-step report, conclusion-first final
  answer, document-level graph, and no model error.
