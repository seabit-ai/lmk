---
name: socratic-pairing-log
description: Use when you and the human work step by step on setup, configuration, an experiment or debugging — especially in consoles, browsers or machines where actions have consequences — or when they say "SPL". Keeps a shared, append-only, evidence-grade log so every step is visible and auditable.
---
SPL = Socratic Pairing Log. The human wants to SEE each step, not just the
result: where we are, what you are about to do and why, what came back, what
comes next. The log is the shared memory of that conversation.

## The file

- One notes file per piece of work, readable and writable by both of you:
  `<topic>.spl.md`, or whatever notes convention this repo already has (a
  dated folder with `notes.md`, `research/YYYY-MM-DD/`…). Follow the repo, do
  not invent a new place. Say the path once when you start.
- Header: goal, date, status, the context a stranger would need.

## The loop

1. **Say the next step** — the command or action and why — before you take
   it. One or two sentences in the reply, not a plan document.
2. **Do it** (or ask the human to, when it is theirs to do: logging in,
   approving, solving a captcha).
3. **Record it** — append the result with its evidence (the command, the
   output that matters, the value read off the screen).
4. **Hand back** — one line: where we are, what is next. The human can
   challenge any decision at any moment; that is the most valuable part.

## Append only

- Use `file_append`. Check the tail first, then add to the end.
- never edit or insert in the middle of the log with an anchored `file_edit`:
  the log must read in the order things happened, and an anchor match that
  fails (or lands on the wrong line) silently reorders history.
- A correction is a new entry that says what changed and why; leave the old
  entry in place.

## Evidence grade

- Every decision carries its reason (it must answer "why").
- Every command carries its output (it proves "it was done").
- Failed attempts are recorded too — they explain why we changed direction.
- Every configuration value, account, id or setting touched is written down.
- Record as you go. Never reconstruct from memory afterwards, and never wait
  to be asked.

## Consequential steps

Anything that changes state outside your own view — submit, apply, send,
delete, save, buy, a POST — gets its intent line BEFORE the action, and a
repeat of the same action gets a reason (why a second time). If the first
attempt's outcome was ambiguous, say so and check; do not click again
silently.

## When you are interrupted

If working on A you discover B (a prerequisite, a broken assumption) must be
fixed first: log where A stands (state, what was done, what remains), switch to
B, do it properly, then return to A from that record. The log is what lets you
come back cleanly.

## Closing

End with a short summary entry: what is done, what is open, where the
artifacts live. Pin the pointer with `memory_note` so the next session finds it.
