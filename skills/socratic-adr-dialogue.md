---
name: socratic-adr-dialogue
description: Use when the human wants to design or decide something with you — architecture, a module, a protocol, a workflow, a naming or ownership question — or says "SAD". A one-point-at-a-time design dialogue whose decisions are written down together with the rejected options.
---
SAD = Socratic ADR Dialogue. Designing together, not being handed a verdict.
The conversation is part of the deliverable.

## How the dialogue runs

- **One point at a time.** The human pokes a concrete point; you go deep on
  that point only. No questionnaires, no menus of ten open questions.
- **Lay out the reasoning chain**: short, from constraints to conclusion —
  facts from the code or the evidence first, then what follows from them. End
  with a clear recommendation and the single question you need answered.
- **Take pushback seriously.** When they reject an argument, re-derive. A
  conclusion may flip; that is the process working. Do not cave to be
  agreeable and do not dig in — say what their point changes and what it does
  not.
- **Surface deviations from the project's conventions** (naming, error
  handling, IO shape, layout) as explicit decision points: "this design departs
  from convention X because Y". A deviation that is not shown is a deviation
  nobody ruled on.
- **Keep turns short** so they can interject at any moment. Do not launch a
  long tool chain in the middle of a design dialogue.

## Writing it down

- As soon as a point is ruled, update the design record (the repo's design-doc
  place, e.g. `docs/design/YYYY-MM-DD-slug.md`) and commit it.
- Mark each point: decided, open, deferred. Keep a list of **rejected options
  with the reason each was rejected** — the record exists so the same
  discussion is not re-run in six months.
- State what the human decided in their words, and where you recommended
  otherwise say so ("I recommended A; the ruling is B, which turns my objection
  into a design constraint: …").

## A ruling is not a go

- Closing a design point does not authorize implementation. Starting to
  build needs an explicit instruction from the human ("go", "start the long
  run"). When it is ambiguous, ask one short question.
- Likewise, a go to implement is not a go to merge or deploy: those are
  separate authorizations, asked for separately, never inferred from context or
  from an earlier "yes".
- Anything that comes up mid-way and is not the point at hand goes into the
  repo's backlog queue (enqueue, do not dive), so the thread of the dialogue is
  not lost.
