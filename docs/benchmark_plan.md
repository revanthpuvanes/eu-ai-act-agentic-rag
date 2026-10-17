
# EU AI Act RAG — Evaluation Benchmark Plan

## Goal

Create 60 ground-truth questions before implementing either the naive RAG or agentic RAG system.

Both systems will be evaluated against the same frozen benchmark.

## Question distribution

| Category | Total |
|---|---:|
| single_hop | 20 |
| multi_hop | 15 |
| exceptions | 10 |
| definitions | 5 |
| unanswerable | 5 |
| temporal | 5 |
| **Total** | **60** |

## Dataset splits

- Development: 20 questions
- Final test: 40 questions

The first five questions (Q001-Q005) belong to the development split because they have already been inspected during benchmark preparation.

The remaining questions will be distributed across development and test to maintain category diversity.

## Ground-truth requirements

Each question must contain:

- Unique question ID
- Question text
- Category
- Development or test split
- Relevant article numbers
- Reference answer
- Supporting evidence from the official regulation, where applicable
- Verification status

Unanswerable questions must explicitly state why the corpus does not support an answer.

## Evaluation principles

1. Write questions before building or tuning either RAG system.
2. Use the same questions and source corpus for both systems.
3. Do not use test questions for prompt or retrieval tuning.
4. Check factual correctness and completeness separately.
5. Evaluate citation quality and unsupported claims.
6. Measure latency, token usage and cost from actual runs.
7. Never invent experimental results.
8. Record limitations and possible benchmark contamination.

## Current progress

Q001-Q005: verified, development split.

Q006-Q060: not yet written.

## Next milestone

Create the remaining 55 question specifications, then write and review their reference answers using the extracted EU AI Act corpus.
