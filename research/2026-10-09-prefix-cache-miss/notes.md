# The 10,240 prefix-cache miss: solved (2026-10-09)

Answers the FR "make a prefix-cache miss self-explaining" (owner's nova repo,
`infra/lmk-feature-requests/prefix-cache-miss-visibility.md`). The FR reported a warmed
~82k-token prefix restoring only 10,240 tokens on what was believed to be a byte-identical
prompt, and asked for visibility because the logs could not distinguish (1) prompt
divergence / (2) record retention / (3) an addressing bug.

**Verdict: it was (1), prompt divergence, on both misses of the day.** No engine bug, no
retention loss, no addressing error. The engine behaved exactly as designed; the two numbers
that would have made this self-explaining from the logs are named in PCM-007, and they are
not the ones the FR asked for first.

Evidence lives on the owner's machine (`~/.lmk/logs/`, `~/src/nova/.kitten/logs/`,
`~/.lmk/cache/`); raw outputs carry session content and stay out of this repo. Scripts and
numbers are here. All numbers: m3u (M3 Ultra), qwen3.8-27b-4bit, kv16, dflash2,
lmk v0.7.2-38-g44c7d1d, cache dir `d3caf54bed4bc104e7ef91e9` (born 09-27).

## Findings

### PCM-001 — Both misses were prompt divergence, proven by on-disk chunk birth times
Recomputing the miss request's chunk keys (sha256 chain over 256-token chunks, exactly as
`prompt_cache/chunks.py`) and stat-ing the record files in the cache directory:

- chunks 1–6 born 09-27 04:07 (stable tools/system head, 12 days old)
- chunks 7–42 born 10-09 10:33:20–10:33:53 (the morning turn's cold prefill)
- **chunk 40's `state_checkpoint` born 10:33:46, mtime 10-09 15:53:34 — touched by the
  15:53 miss's restore.** That file IS the 10,240 boundary.
- chunks 43+ born 15:53:34+ — written by the missing request's own re-prefill (which is why
  round .3 then restored 82,176: "healed" = self-repopulated).

So the 15:53 request matched the day's existing chain for exactly 42 chunks and diverged in
chunk 43 — the prompt changed, nothing was lost. APFS `st_birthtime` on record files is a
free forensic channel nobody had used.

### PCM-002 — The Qwen3.8 template renders ~10k tokens of tools BEFORE the system text
Verified by rendering a marker conversation: the template emits the reasoning-effort
preamble, then `# Tools` with every function schema, then the system message content.
nova's toolset ≈ 9.8k tokens. Consequences:

- An edit to the **system prompt** diverges ~10k tokens deep (here: token 10,835 — tools
  plus the first 1,014 tokens of system text were shared).
- An edit to a **tool schema** diverges near token 0.

"The system prompt is the first thing in the prompt" is false for this family. Any
client-side reasoning about where a prefix diverges must account for template section order.

### PCM-003 — The divergence: the client's system prompt changed between 15:50 and 15:53
Message-0 md5 across the day's wiredump (client request log):

- 10:33–15:50, every request including all warmups: md5 `69d88042` (15,186 chars)
- 15:53 miss and everything after: md5 `091a0e12` (15,453 chars) — a 267-char section
  inserted at char 4,601 = system-text token 1,014 (browser-identities block)

The FR's "byte-identical prompts" compared the miss (.1) against the NEXT round (.3) — both
post-change. The pair that mattered (last pre-change request vs the miss) was never diffed.
Lesson for miss triage: diff the miss against the **last request that hit deep**, not
against a later request.

### PCM-004 — Restorable length snaps to state-checkpoint boundaries, not to the token match
The planner (`restore_planner.py`) needs kv_deltas for every chunk in the chain plus an
**exact-boundary `state_checkpoint`** (the GDN/linear-layer state is opaque; only saved
where `chunk.end == snapshot_len`, i.e. at prefill-step ends — 2048-aligned — and at exact
decode crossings). On this prompt: tokens matched to 10,835 (42 full chunks, kv_deltas all
present), but the last checkpointed boundary was chunk 40 → restored 10,240. The gap
between "matched" and "restorable" is a designed granularity cost, invisible in today's
logs, and it is why observed miss values land on suspiciously round 2048-multiples.

### PCM-005 — "No eviction pressure" was true for the window, and is false going forward
Engine stderr: `lifetime_evicted_mib=0.0` from model load until the miss — zero evictions
ever, so retention was impossible as a cause here. But the miss's own re-prefill pushed the
store over the 200 GB cap for the first time, and it has been evicting continuously since
(tens of GB per active hour at 100k+ context). Future misses CAN be retention; PCM-007's
instrumentation is what will tell them apart.

### PCM-006 — The morning cached=0 miss: tool schemas changed, and short matches restore 0
The 10:32 client restart changed tool definitions (new browser profile param) → divergence
within the tools section's first chunks. Chunks 1–6 still matched 12-day-old records, but
with no checkpoint at or below chunk 6 the plan is empty → `cached=0`. The FR's explanation
("system prompt gained the identities section") was wrong in detail: the 10:38 request still
carried the OLD system prompt (md5 `69d88042`); it was the tools that moved. Also a second
client-side fact: the system prompt did NOT change at the 10:32 restart — it changed at
15:53, mid-session, when the client's turn path refreshed a snapshot its warmup path had
been holding stale (warmups kept sending the old prompt until the end; client-side bug,
reported to nova).

### PCM-007 — The log line that makes a miss self-explaining (supersedes FR items 1 and 4)
Three numbers per request close the case without wiredumps:

- `promptTokens` (already logged)
- `matchedTokens` — longest prefix whose chunk chain is indexed, **ignoring** checkpoint
  availability (pure token match against the index)
- `restoredTokens` (already logged) — what the checkpoint granularity allowed

Read-out: `matched ≪ prompt` → the prompt diverged at ~matched; go diff the client's last
deep-hit request at that offset. `matched ≈ prompt, restored ≪ matched` → retention or
checkpoint granularity; the record lifecycle log (FR item 2) then names which. On this
incident the line would have read `prompt=82292 matched≈10752 restored=10240` — divergence
at ~10.7k, inside the system text per PCM-002, five minutes to root cause instead of a day.

The FR's item 1 (hot/disk split + record length) and item 4 (hash compare at the stop
block) would NOT have discriminated this case: at the stop block the incoming chunk's hash
simply isn't in the index, which reads as "absent" for divergence and retention alike.
`matchedTokens` is also nearly free: the planner already walks the chain; it only needs to
keep walking past the first missing checkpoint and report how far token identity went.

## Decision log (hypotheses tried and dropped, in order)
- Engine deleted/evicted records in the 15:50→15:53 window → no: zero evictions ever
  (stderr counters), index only grew, no saves ran in the window.
- Restore load failure evicting a record and re-planning shorter → no: a load failure is
  swallowed as a full miss (`cached=0`), never re-planned; and the observed 10,240 came from
  the plan itself.
- Store rebuilt/process restarted in the window → no: single store lifetime since model
  load; `LmkCacheRestored` last fired Sept 27.
- Hot-cache mislabeled as disk → no: hot restore for this model family effectively never
  serves trimmed prefixes (GDN state is untrimmable), and source labels are set explicitly.
- A pre-existing chain from another session / yesterday matching the new prompt → no:
  yesterday's client build rendered a third system-prompt variant (md5 `3093fd4c`); no other
  session's traffic shares this thread's history. The matching chain turned out to be the
  same day's OWN morning chain, shared through the tools section (PCM-002).

## Scripts
`scripts/` holds the forensic tools (paths passed as arguments; no session content inside):
message-array diff, system-message md5 scan, tokenizer divergence locator, cumulative
tokens per message, chunk-key recomputation + record stat (birthtime/mtime), template
section-order probe. The chunk-key script reproduces the engine's chunking exactly and is
the seed of the FR's item 3 (deterministic repro harness).
