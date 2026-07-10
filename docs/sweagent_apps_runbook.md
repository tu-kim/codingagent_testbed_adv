# Running APPS through the SWE-agent scaffold

How we drove the SWE-bench/APPS samples through the **SWE-agent** scaffold
(instead of OpenCode) against the *same* Dynamo → vLLM backend, to explain
why tool-time share differs across scaffolds (OpenCode ≈1% vs the ~35%
SWE-agent numbers in the literature). Everything here is what actually
worked on the GPU host after the debugging pass — the driver is
`scripts/run_sweagent_apps.py`, the decomposition is
`scripts/analyze_sweagent_traj.py`, correctness is `scripts/evaluate_apps.py`.

The point of the comparison: tool share is a joint property of
*(scaffold granularity × execution environment × model/serving speed)*, not
of the benchmark. This run holds the samples (identical deterministic
selection via `testbed.apps.load_samples`), the model, and the serving stack
constant and swaps **only** the scaffold, so the residual difference is
attributable to the scaffold + its execution environment.

---

## 1. Install SWE-agent — FROM SOURCE

The PyPI name `sweagent` is an **unrelated squatted 0.0.1 package** whose
`togetherunidiff` dependency doesn't even resolve. Install the real thing
from GitHub:

```bash
source .venv/bin/activate
pip install "git+https://github.com/SWE-agent/SWE-agent.git"
# or, pinned to a release tag, clone + editable:
#   git clone https://github.com/SWE-agent/SWE-agent.git
#   pip install -e SWE-agent

# Verify the CLI surface the driver was written against (1.x):
sweagent run --help
```

If `sweagent run --help` disagrees with any flag, the fix goes in **one
place**: `build_sweagent_cmd()` in `scripts/run_sweagent_apps.py`. Validate a
new version with `--dry-run` (prints the exact per-task commands, no side
effects) before a real run.

## 2. Deployment: docker (default) vs local

`--deployment docker` is the **upstream-supported path** and matches the
container round-trip the SWE-agent literature numbers include — use it.

`--deployment local` was meant to isolate the pure scaffold factor (no
container), **but it does not work for non-root users**: upstream hardcodes
copying the repo to the deployment root `/{repo_name}` and writing
`/root/model.patch`, so on a host that means writing to `/` →
`PermissionError: [Errno 13] Permission denied: '/session-apps-...'`. The
configurable-base-dir fix (upstream **PR #1132**) was closed unmerged with
"use mini-swe-agent for fully-local runs". So:

- `docker` → what we used. Requires Docker + network reachable from inside
  the container (see §3).
- `local` → only if your `/` is writable (i.e. you are root). Otherwise use
  mini-swe-agent as a separate third arm.

## 3. Docker prerequisites (the part that actually bit us)

SWE-agent spins up a container per task and, via **swe-rex**, pip-installs
its tool bundle *inside* that container at startup. That means the container
needs **outbound network** to PyPI. Symptoms when it can't reach out: the
run hangs with nothing on the Dynamo frontend, and DEBUG shows
`docker run ...` followed by a stalled `pipx`/`pip` bootstrap.

Two things had to be right:

**a) Proxy — if the host is behind one** (the same network where
`git submodule` returned HTTP 403), the container inherits nothing by
default. Configure Docker's proxy at both layers:

- daemon + client proxy in `~/.docker/config.json` (`proxies` block), so
  `docker run` injects `HTTP_PROXY`/`HTTPS_PROXY` into the container, **and**
- `NO_PROXY=localhost,127.0.0.1` — otherwise litellm's call from inside the
  scaffold to the Dynamo endpoint gets sent through the proxy and fails.

**b) Do NOT use `sweagent/swe-agent:latest`.** That tag is the OLD 0.x
miniconda / py3.9 image; its swe-rex install is script-less and breaks
(`swe-rex executable not found`). Let the driver use the default modern
`python:3.11`-based flow (network working), or pre-bake a
`swerex-py311` image if you want to skip the per-task pip bootstrap. A
prebaked image is also the fix if the pip round-trip is too slow/flaky —
it removes the startup install entirely.

> The per-task container create + swe-rex bootstrap + tool install is
> **~35–54 s/task** of pure environment startup (measured). That is almost
> the entire "others" bucket in the decomposition — see §6.

## 4. OPENCODE_PROFILE does NOT apply here

SWE-agent is a different scaffold — there is no OpenCode process in this
path, so `OPENCODE_PROFILE=1` does nothing. Timing comes from two other
sources instead:
- **tool** = per-step execution time recorded in SWE-agent's `.traj`,
- **llm** = the Dynamo frontend log's `request completed` `elapsed_ms`,
  joined to each task by its `task_start/end_unix_s` window (the run is
  strictly sequential, so the join is unambiguous).

So the Dynamo frontend must be logging normally; no profiler patch is
needed on the SWE-agent side.

## 5. Run it

The backend (workers → frontend → nothing-else-needed; OpenCode is not
used) must be up first, exactly as for an OpenCode run:

```bash
deploy/testbed.sh up nats
deploy/testbed.sh up etcd
deploy/testbed.sh up workers
deploy/testbed.sh up frontend
```

Dry-run first (flag-surface check, no side effects):

```bash
scripts/run_sweagent_apps.py --split competition --num-samples 20 --seed 42 \
    --out results/sweagent-apps1 --dry-run
```

Then the real run (strictly sequential — matches the OpenCode comparison
mode; one request in flight at a time so the frontend-log join is clean):

```bash
scripts/run_sweagent_apps.py --split competition --num-samples 20 --seed 42 \
    --out results/sweagent-apps1 \
    --deployment docker \
    --max-steps 50 \
    --task-timeout-s 1800
```

Defaults resolved from `deploy/testbed.yaml`:
- `--api-base` → `http://<dynamo.host>:<dynamo.port>/v1`
- `--model`    → `model.served_name` (sent to litellm as `openai/<served_name>`)
- `--workspace-root` → `<workspace_root>/sweagent`

Cost limits are **zeroed** deliberately — litellm can't price a local model
and `sweagent` aborts on unpriced models; the step budget is enforced via
`per_instance_call_limit` (`--max-steps`) instead.

### What each task does (driver flow)

1. Materialize the APPS workspace (`PROBLEM.md` + `solution.py` via
   `apps.prepare_workspace`, always `reset=True`) and `git init` + commit —
   SWE-agent operates on a git repo; the commit is the base its patch is
   diffed against.
2. `sweagent run` as a subprocess (model → litellm openai-compatible; env →
   `--env.repo.path <ws>` + `--env.deployment.type <docker|local>`; task →
   `--problem_statement.text <rendered prompt>` + `.id <instance_id>`).
   Wall-clocked; `task_start/end_unix_s` recorded for the frontend join.
3. Locate the produced patch `<out>/trajs/<id>/<id>.patch` and `git apply`
   it back onto the pristine workspace — so `scripts/evaluate_apps.py` scores
   this run dir **identically** to an OpenCode run. (`git -C <ws> apply`
   chdirs first, so the driver passes the **resolved absolute** patch path —
   a relative one fails with "can't open patch".)
4. Append a TaskRecord-shaped line to `trace.jsonl`.

Outputs under `--out`: `config.json` (workload=apps, agent=swe-agent,
sample tuple, sweagent knobs, workspace_root — evaluate_apps-compatible),
`trace.jsonl` (one record/task incl. `task_start/end_unix_s`, `traj_dir`,
`patch_applied`), `trajs/<id>/` (raw `.traj`, patch, logs).

## 6. Analyze

**Correctness** (executes model code — run in a container):

```bash
scripts/evaluate_apps.py --run results/sweagent-apps1
```

**Time decomposition** — llm / tool / others, plus the env-setup split:

```bash
scripts/analyze_sweagent_traj.py --run results/sweagent-apps1 \
    --frontend logs/frontend.log
```

This prints two share tables:
- **pooled share** (total-based): `others` ≈ 30% of wall, which is almost
  entirely `env_head_s` (the docker create + swe-rex bootstrap + tool
  install of §3), tail ≈ 1 s.
- **active-time share** (env head/tail excluded): **this is the row to
  compare against OpenCode.** The literature's ~35% tool numbers are a
  *share of active time*, and OpenCode's turn-level shares carry no per-task
  env startup either — so only the active-time row is apples-to-apples.

The env-startup being ~the whole "others" bucket is the headline finding:
the raw wall-clock tool share looks inflated on docker mostly because of
container/tool bootstrap, not because the scaffold spends more time in tools
per se — you have to strip head/tail to compare scaffolds fairly.

## 7. Gotchas checklist

- Installed `sweagent` from **GitHub**, not PyPI (name-squat).
- `--deployment local` fails for non-root (hardcoded copy to `/`); default
  to `docker`.
- Container needs outbound network for the swe-rex pip bootstrap — set
  Docker proxies + `NO_PROXY=localhost,127.0.0.1` on proxied hosts.
- Do **not** use `sweagent/swe-agent:latest` (old py3.9 image, broken
  swe-rex); use the default python:3.11 flow or a prebaked py311 image.
- `OPENCODE_PROFILE` is irrelevant here; timing = `.traj` steps (tool) +
  frontend `request completed` (llm), joined by the sequential task window.
- All sweagent CLI flags live in `build_sweagent_cmd()`; re-validate with
  `--dry-run` + `sweagent run --help` after any version bump.
- Compare scaffolds on the **active-time** share table, not the total-based
  one.
