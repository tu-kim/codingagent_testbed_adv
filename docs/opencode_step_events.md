# OpenCode step lifecycle and profiling events

Reference for what an "AI SDK step" actually contains, in what order events fire, and how `deploy/patches/opencode-profile.patch` decomposes a step into LLM / tool / overhead time.

The events in this document are emitted by the Vercel AI SDK v5 `streamText` stream and consumed in `opencode/packages/opencode/src/session/processor.ts:220` (`handleEvent`). Names match the AI SDK's `type` field on each stream chunk.

## What a "step" is

A **step** in AI SDK v5 brackets:

1. One LLM HTTP request → streamed response (text, reasoning, tool args)
2. Every tool execution triggered by that response
3. Post-tool bookkeeping inside the SDK before the next step begins

`start-step` and `finish-step` events bracket the whole thing. A single agent **turn** (one iteration of OpenCode's `runLoop`) is exactly one step from the SDK's view; the testbed uses `step` and `turn` interchangeably in profile output.

A **query** is a full `runLoop` invocation = multiple steps until the model stops calling tools or hits a stop condition.

## OpenCode call path (agent loop)

Where the profile hooks sit in the actual OpenCode control flow. All line
references are to the vendored `opencode/packages/opencode/src/session/`
(post-`opencode-profile.patch`); a "turn" is one iteration of the outer
`while (true)` loop in `prompt.ts`, which is one `handle.process(...)`
call, which is one AI-SDK `streamText` step — the correspondence is 1:1.

```
runLoop (prompt.ts)
  Profile.query.start                         ← just above `while (true)`
  while (true):                               ← one iteration = one TURN
    status=busy; step++                        (prompt.ts top of loop)
    getModel · subtask/compaction branch checks
    insertReminders · build assistant message
    sessions.updateMessage(msg)                ← DB write (persist turn)
    processor.create(...)
    resolveTools(...)                          ← per-turn tool array (hooks attached)
    plugin "chat.messages.transform"
    Effect.all([sys.*, toModelMessagesEffect]) ← PROMPT ASSEMBLY (wire format)
    Profile.turn.start(step, {system,messages})← stamped just before process()
    result = handle.process({...}):            ← consumes the streamText stream
        Profile.llm.start                       (processor.ts case "start-step")
        …text/reasoning/tool-input deltas…
        Profile.tool.start → execute() → Profile.tool.end   (per tool, inside process)
        Profile.llm.streamFinish                (case "finish")
        Profile.llm.end                         (case "finish-step"; response_id here)
    Profile.turn.end(step)                     ← just after process() returns
    inspect handle.message.finish:
        "tool-calls" → continue                 (another turn)
        terminal (stop/…) → break               (query ends)
  Profile.query.end
```

### The inter-turn gap (`turn.end(N)` → `turn.start(N+1)`)

Everything between one turn's `handle.process` returning and the next
turn's `handle.process` starting is **pure agent-loop plumbing — no LLM
streaming and no tool execution happen here** (both already completed
inside turn N, before `turn.end(N)`). What runs, in order:

1. Loop-continue decision from `handle.message.finish` (`tool-calls` →
   continue, terminal → break).
2. `step++`, `getModel()`, subtask/compaction branch checks, overflow →
   compaction fork.
3. New assistant message construction + `sessions.updateMessage` — a **DB
   write**.
4. `resolveTools()` — rebuild this step's tool array (each tool re-wrapped
   with the `Profile.tool.*` hooks).
5. reminder injection + `<system-reminder>` wrapping + plugin transforms.
6. **Prompt assembly** — `Effect.all([sys.skills, sys.environment,
   instruction.system, toModelMessagesEffect(msgs)])` (wire-format build;
   the chat template itself is applied later, frontend-side in Rust).
7. `Profile.turn.start(N+1)` fires, immediately before the next
   `handle.process`.

This is the **scaffold** slice of the canonical decomposition (see
`analyze_profiles.py` / `scripts/arm/`): time that is neither the server
LLM wall nor tool execution. Two things worth pinning:

- **Tool execution is INSIDE the turn**, not the gap: the AI SDK invokes
  `execute()` while consuming the stream, so `tool.start`/`tool.end` sit
  between `llm.end` and `turn.end(N)`.
- **`post_overhead_s` is also inside turn N** (snapshot.track/patch + DB
  write at `finish-step`, `processor.ts:453-514`), NOT in the gap. So the
  full non-LLM/non-tool time of a turn = `post_overhead_s` (inside the
  turn) + this inter-turn gap. The gap itself is not a single profile
  field — measure it as `turn.start(N+1).ts − turn.end(N).ts`.

Sub-agent/subtask and compaction loop iterations `continue` **before**
reaching `Profile.turn.start`, so they produce no turn bracket — a
profiled "turn" is specifically a real model-generation iteration.

## Event types within a step

Events fire in the order received from the model stream. Multiple parts (text blocks, reasoning blocks, tool calls) can occur in one step. The "Lifecycle" column shows when each event is fired by the SDK relative to the model's output stream.

### Step brackets

| Event | When fired | Notes |
|-------|------------|-------|
| `start-step` | Step begins, just before HTTP request to model sent | Step counter increments per loop iteration in OpenCode |
| `finish-step` | Step ends, AFTER all tools in this step have completed and the SDK has finished bookkeeping | Includes `usage` (prompt/completion tokens) and `finishReason` |

### Top-level stream brackets (NOT per-step)

| Event | When fired | Notes |
|-------|------------|-------|
| `start` | Top of the entire `streamText` invocation | Fires once for the whole runLoop iteration that sets up the stream |
| `finish` | Bottom of the entire `streamText` invocation | Fires once after all steps complete |

In OpenCode these aren't used to time anything — they're outside the per-step granularity.

### Text emission

| Event | When fired | Multiplicity per step |
|-------|------------|------------------------|
| `text-start` | Model emits the first token of a text block | 0..N |
| `text-delta` | Each text token chunk | many per text-start |
| `text-end` | Model emits the end of a text block | 0..N, matches text-start |

For a final answer turn (no tool calls), the model emits one text block: `text-start → … deltas … → text-end → finish-step`.

### Reasoning emission (Anthropic extended thinking, Qwen3 reasoning, etc.)

| Event | When fired |
|-------|------------|
| `reasoning-start` | Model begins reasoning block |
| `reasoning-delta` | Each reasoning chunk |
| `reasoning-end` | Reasoning block ends |

Treated identically to text from a timing standpoint (still LLM-streaming).

### Tool call lifecycle (per tool)

Order is strict: every tool fires these in sequence.

| Event | When fired | Notes |
|-------|------------|-------|
| `tool-input-start` | Model begins streaming this tool's args | One per tool |
| `tool-input-delta` | Each chunk of tool args | Many per tool |
| `tool-input-end` | Model finishes streaming this tool's args | One per tool |
| `tool-call` | AI SDK has fully parsed the tool args JSON | Immediately precedes `execute()` invocation |
| `tool-result` | Tool's `execute()` returned successfully | Carries `output.output` and `output.metadata` |
| `tool-error` | Tool's `execute()` threw (alternative to `tool-result`) | Carries `error` |

Important: **`tool-call` is the closest stream-event observable to "AI SDK invoked `execute()`"**. The SDK invokes `execute()` synchronously from its own state machine; OpenCode's tool wrapper (which fires `Profile.tool.start`) runs inside that invocation. So `Profile.tool.start` timestamp ≈ `tool-call` timestamp ≈ "model fully done streaming this tool".

### Error / abort

| Event | When fired |
|-------|------------|
| `error` | Any stream error |

## Typical orderings within one step

**Final-answer step (text only, no tools):**
```
start-step
  text-start → text-delta* → text-end
finish-step
```

**Tool-only step (no preamble text):**
```
start-step
  tool-input-start → tool-input-delta* → tool-input-end → tool-call
  [execute() runs]
  tool-result
finish-step
```

**Mixed step (model emits text, then a tool):**
```
start-step
  text-start → text-delta* → text-end          ← text part of model output ends here
  tool-input-start → … → tool-input-end → tool-call   ← tool args streamed AFTER text
  [execute() runs]
  tool-result
finish-step
```

In the mixed case, the model is **continuously streaming** between `text-end` and `tool-call`. `text-end` is NOT the end of the LLM round-trip; the LLM is still working on the tool args.

**Multi-tool step (parallel tools):**
```
start-step
  text-start → text-end                         (optional)
  tool-input-start(A) → … → tool-call(A)
  tool-input-start(B) → … → tool-call(B)
  [execute(A) and execute(B) both run; may overlap]
  tool-result(A), tool-result(B) in completion order
finish-step
```

Whether tools A and B run in parallel depends on AI SDK config and OpenCode wrapper semantics; in practice for Qwen3-Coder serial tools are the common case.

## Profile timing decomposition

`deploy/patches/opencode-profile.patch` instruments select events to derive the following per-step quantities. See `opencode/packages/opencode/src/profile/profile.ts` for the implementation.

| Quantity | Definition | Hook |
|----------|------------|------|
| `llm.start` ts | Timestamp of `start-step` | `processor.ts case "start-step"` |
| `streamEnd` (internal) | First `Profile.tool.start` of this step if any tool ran, else last `text-end`, else `finish-step` time | `Profile.tool.start` and `processor.ts case "text-end"` |
| `llm.duration_s` | `streamEnd − llm.start` ⇒ pure LLM streaming time | computed in `Profile.llm.end` |
| `llm.step_duration_s` | `finish-step − start-step` ⇒ whole AI SDK step bracket | computed in `Profile.llm.end` |
| `tool.start` ts | When OpenCode's tool wrapper enters its `execute()` body | `Effect.gen` enter in `prompt.ts` builtin + MCP wrappers |
| `tool.end` ts | When the wrapper's `Effect.onExit` fires | `Effect.onExit` in same wrappers |
| `tool.duration_s` | `tool.end − tool.start` per tool call | per call |
| `turn.start` ts | Just before `handle.process` runs in `prompt.ts` | `Profile.turn.start` |
| `turn.end` ts | Just after `handle.process` returns | `Profile.turn.end` |
| `turn.duration_s` | `turn.end − turn.start` | `Profile.turn.end` |
| `turn.llm_wall_s` | Σ over this step of `llm.duration_s` (only one step per turn in practice) | reads `llmDurationByStep` |
| `turn.tool_wall_s` | Σ over this step of `tool.duration_s` | accumulated in `Profile.tool.end` |
| `turn.post_overhead_s` | `max(0, turn.duration_s − llm_wall_s − tool_wall_s)` | computed in `Profile.turn.end` |

### Why `first tool.start` is the LLM streamEnd marker

A naive choice would be to mark "LLM done" at `text-end`. That is **wrong for tool-call steps**, because:

- The model HTTP response is a single contiguous token stream.
- After emitting text, the model **continues streaming** the tool_call structure with no pause.
- `text-end` only signals the end of the text block, not the end of the model's contribution to this step.

After `tool-input-end` the SDK still needs to parse args. The cleanest moment after which the model can no longer be talking is when the SDK actually **invokes `execute()`** for the first tool, which is the moment `Profile.tool.start` fires (the tool wrapper's enter point). So that timestamp is taken as `streamEnd` for tool steps.

For text-only steps no `tool.start` ever fires; fallback to last `text-end` timestamp.

### Decomposition equation

```
turn.duration_s ≈ pre_turn_overhead
                + llm_wall_s        (start-step → streamEnd, pure model streaming)
                + tool_wall_s       (Σ tool.duration_s)
                + post_tool_gap     (last tool.end → finish-step)
                + post_turn_overhead (finish-step → turn.end)
```

In `turn.end` we collapse the three non-LLM/non-tool slices into `post_overhead_s`:

```
post_overhead_s = max(0, turn.duration_s − llm_wall_s − tool_wall_s)
```

The `max(0, …)` clamp handles the **parallel-tool caveat**: when two tools run concurrently, `tool_wall_s` is their sum but elapsed wall time is closer to their max, so the naive subtraction can go negative. A `post_overhead_s = 0.0` reading in a turn that also has more than one tool call is a strong signal that those tools ran in parallel — fall back to raw `tool.start`/`tool.end` pairs for that case.

## Cross-reference

- Patch file: `deploy/patches/opencode-profile.patch`
- Apply / check / revert: `scripts/apply_opencode_patches.sh [--check|--revert]`
- Aggregate per-session profiles: `scripts/aggregate_profiles.sh <workspace_root>`
- Profile module source (under patch): `opencode/packages/opencode/src/profile/profile.ts`
- AI SDK event consumer (under patch for hooks): `opencode/packages/opencode/src/session/processor.ts`
- Turn / query / tool brackets (under patch for hooks): `opencode/packages/opencode/src/session/prompt.ts`
