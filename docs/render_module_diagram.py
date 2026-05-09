"""Render docs/module-diagram.png — two-band view of the testbed.

Upper band: in-process Python driver. Lower band: out-of-process service
stack. Lifecycle (testbed.sh up/down) is summarized between the bands rather
than drawn as N spaghetti arrows.

Run: `python3 docs/render_module_diagram.py`. No graphviz needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


_OUT = Path(__file__).resolve().parent / "module-diagram.png"


# ----- Palette -----
COLOR_INPUT   = "#fff3b0"
COLOR_PYTHON  = "#cfe7ff"
COLOR_SHELL   = "#d4f5d4"
COLOR_SERVICE = "#ffd6cc"
COLOR_AUX     = "#e8e8e8"
COLOR_OUTPUT  = "#e8d5ff"

EDGE = "#333333"
BAND = "#fafafa"
BAND_BORDER = "#e0e0e0"


def box(ax, x, y, w, h, label, *, color, fontsize=10, weight="normal"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.0, facecolor=color, edgecolor=EDGE,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, weight=weight)
    return (x, y, w, h)


def _bbox_anchor(b, target_xy):
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    tx, ty = target_xy
    dx, dy = tx - cx, ty - cy
    if abs(dx) * h > abs(dy) * w:
        return (x + w if dx > 0 else x, cy)
    return (cx, y + h if dy > 0 else y)


def _center(b):
    x, y, w, h = b
    return (x + w / 2, y + h / 2)


def arrow(ax, src, dst, *, label=None, label_xy=None, color=EDGE,
          style="-|>", lw=1.3, ls="-", curve=0.0, label_size=8.5):
    """Draw an arrow. If label_xy is given (data coordinates), place the
    label there explicitly — much more reliable than offset-from-midpoint."""
    sp = _bbox_anchor(src, _center(dst)) if len(src) == 4 else src
    dp = _bbox_anchor(dst, _center(src)) if len(dst) == 4 else dst
    arr = FancyArrowPatch(
        sp, dp,
        arrowstyle=style, mutation_scale=13,
        linewidth=lw, linestyle=ls, color=color,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)
    if label and label_xy:
        ax.text(label_xy[0], label_xy[1], label,
                ha="center", va="center",
                fontsize=label_size, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def main() -> None:
    # 18 wide × 13 tall gives generous room for labels between boxes.
    fig, ax = plt.subplots(figsize=(18, 13), dpi=150)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 15.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Two band backgrounds
    ax.add_patch(Rectangle((0.2, 7.4), 17.6, 7.4,
                           facecolor=BAND, edgecolor=BAND_BORDER, linewidth=0.5))
    ax.add_patch(Rectangle((0.2, 1.6), 17.6, 5.0,
                           facecolor=BAND, edgecolor=BAND_BORDER, linewidth=0.5))

    # Band labels
    ax.text(0.4, 14.55, "Python driver  (in-process)",
            fontsize=11, color="#666", weight="bold")
    ax.text(0.4, 6.4, "Runtime services  (out-of-process)",
            fontsize=11, color="#666", weight="bold")

    # Title
    ax.text(9, 15.15,
            "codingagent_testbed — module / process diagram",
            ha="center", va="center", fontsize=17, weight="bold")
    ax.text(9, 14.80,
            "SWE-bench → runner.py (Poisson) → OpenCode → Dynamo → vLLM PD",
            ha="center", va="center", fontsize=11, style="italic", color="#555")

    # ============== UPPER BAND ==============

    # Row A — external inputs (y = 13.2)
    yaml_box = box(ax, 0.5, 13.2, 3.6, 1.0,
                   "deploy/testbed.yaml\n(single source of truth)",
                   color=COLOR_INPUT, weight="bold")
    env_box  = box(ax, 4.5, 13.2, 2.6, 1.0,
                   ".env\n(secrets / overrides)",
                   color=COLOR_INPUT)
    hf_box   = box(ax, 14.0, 13.2, 3.5, 1.0,
                   "SWE-bench dataset\n(huggingface.co)",
                   color=COLOR_INPUT)

    # Row B — config + lifecycle controller + template (y = 11.4)
    config_box = box(ax, 0.5, 11.4, 3.6, 1.0,
                     "src/testbed/config.py\n"
                     "pydantic schema · TESTBED__* env overrides",
                     color=COLOR_PYTHON)
    sh_box     = box(ax, 4.9, 11.4, 7.4, 1.0,
                     "deploy/testbed.sh\n"
                     "yq → spawn(workers, frontend, opencode) · PGID teardown",
                     color=COLOR_SHELL, weight="bold")
    tmpl_box   = box(ax, 13.0, 11.4, 4.5, 1.0,
                     "deploy/opencode.json.tmpl\n"
                     "rendered → opencode/opencode.json",
                     color=COLOR_SHELL)

    # Row C — Python driver row (y = 9.4)
    cli_box       = box(ax, 0.5, 9.4, 2.6, 1.0,
                        "cli.py\nclick: run / smoke",
                        color=COLOR_PYTHON)
    runner_box    = box(ax, 3.4, 9.4, 3.4, 1.0,
                        "runner.py\nPoisson driver · pre-clone repo",
                        color=COLOR_PYTHON, weight="bold")
    poisson_box   = box(ax, 7.1, 9.4, 2.0, 1.0,
                        "poisson.py\narrival_offsets",
                        color=COLOR_PYTHON)
    swe_box       = box(ax, 9.4, 9.4, 2.7, 1.0,
                        "swebench.py\nload_samples · render_prompt",
                        color=COLOR_PYTHON)
    oc_client_box = box(ax, 12.4, 9.4, 4.0, 1.0,
                        "opencode.py\nhttpx async — /session, /message",
                        color=COLOR_PYTHON)

    # Row D — driver output (y = 7.8)
    out_box = box(ax, 0.5, 7.8, 6.4, 0.8,
                  "results/<run>/  ·  config.json   trace.jsonl   summary.json",
                  color=COLOR_OUTPUT)

    # ============== LOWER BAND ==============

    # Row E — primary services (y = 4.0)
    opencode_proc = box(ax, 0.5, 4.0, 3.8, 1.6,
                        "OpenCode  :4096\n"
                        "bun run dev serve\n"
                        "?directory=<abs>  HTTP Basic\n"
                        "OPENCODE_CONFIG=<abs>",
                        color=COLOR_SERVICE, weight="bold")
    dynamo_proc   = box(ax, 5.2, 4.0, 4.6, 1.6,
                        "Dynamo frontend  :8000/v1\n"
                        "python -m dynamo.frontend\n"
                        "--router-mode\n"
                        "--discovery-backend etcd",
                        color=COLOR_SERVICE, weight="bold")
    prefill_proc  = box(ax, 10.7, 4.2, 3.2, 1.4,
                        "vLLM prefill\n"
                        "kv_producer\n"
                        "--disagg prefill",
                        color=COLOR_SERVICE)
    decode_proc   = box(ax, 14.2, 4.2, 3.3, 1.4,
                        "vLLM decode\n"
                        "kv_consumer\n"
                        "--dyn-tool-call-parser",
                        color=COLOR_SERVICE)

    # Row F — aux infra (y = 2.0)
    etcd_box = box(ax, 10.7, 2.0, 3.2, 0.85,
                   "etcd  :2379\n(discovery)",
                   color=COLOR_AUX)
    nats_box = box(ax, 14.2, 2.0, 3.3, 0.85,
                   "NATS  :4222\n(request / event plane)",
                   color=COLOR_AUX)

    # ============== Edges ==============

    # ---- Upper band ----
    arrow(ax, yaml_box, config_box,
          label="load",
          label_xy=(2.3, 12.85))
    arrow(ax, yaml_box, sh_box,
          label="yq",
          label_xy=(4.5, 12.85))
    arrow(ax, env_box, config_box,
          label="TESTBED__*",
          label_xy=(4.05, 12.85), curve=-0.18)
    arrow(ax, env_box, sh_box, curve=-0.05)

    # testbed.sh renders the template
    arrow(ax, sh_box, tmpl_box,
          label="render", label_xy=(12.65, 11.85))

    # config feeds cli
    arrow(ax, config_box, cli_box,
          label="resolved cfg", label_xy=(2.3, 10.85))

    # cli → runner → submodules
    arrow(ax, cli_box, runner_box)
    arrow(ax, runner_box, poisson_box)
    arrow(ax, runner_box, swe_box)
    arrow(ax, runner_box, oc_client_box,
          label="HTTP", label_xy=(9.5, 10.05))

    # swebench fetches SWE-bench
    arrow(ax, swe_box, hf_box,
          label="datasets.load_dataset",
          label_xy=(12.95, 10.85), curve=0.18)

    # runner → results
    arrow(ax, runner_box, out_box,
          label="TaskRecord", label_xy=(3.8, 8.85))

    # ---- Bridge: opencode.py → OpenCode service ----
    arrow(ax, oc_client_box, opencode_proc,
          label="POST /session\nPOST /session/:id/message",
          label_xy=(3.5, 7.0), curve=-0.25, lw=1.5,
          label_size=8.5)

    # tmpl → OpenCode (config consumed at startup via env var)
    arrow(ax, tmpl_box, opencode_proc,
          label="OPENCODE_CONFIG=<abs>",
          label_xy=(13.0, 7.0), curve=0.35,
          color="#666", label_size=8)

    # ---- Lower band ----
    # OpenCode → Dynamo (label ABOVE row at y=5.95, well clear of box tops at 5.6)
    arrow(ax, opencode_proc, dynamo_proc,
          label="POST /v1/chat/completions   (OpenAI)",
          label_xy=(4.75, 6.05), label_size=9.5, lw=1.6)

    # Dynamo → workers (label above row)
    arrow(ax, dynamo_proc, prefill_proc,
          label="NATS request plane",
          label_xy=(10.25, 6.05), label_size=8.5)
    arrow(ax, dynamo_proc, decode_proc, curve=-0.10)

    # prefill → decode KV transfer
    arrow(ax, prefill_proc, decode_proc,
          label="KV transfer (NixlConnector)",
          label_xy=(14.05, 6.05), label_size=9, lw=1.7, color="#b0431a")

    # frontend uses etcd / NATS — labels in the gap between bands
    arrow(ax, dynamo_proc, etcd_box,
          label="watch / discover",
          label_xy=(11.0, 3.45), color="#555", label_size=8.5)
    arrow(ax, dynamo_proc, nats_box, curve=-0.05, color="#555")

    # ============== Lifecycle annotation (between bands) ==============
    ax.annotate(
        "↓  testbed.sh up/down spawns & PGID-kills every service in the lower band\n"
        "    (etcd & NATS are external prereqs; up etcd / up nats are conveniences)",
        xy=(8.6, 11.35), xytext=(8.6, 6.95),
        ha="center", va="center",
        fontsize=8.5, color="#666", style="italic",
        arrowprops=dict(arrowstyle="-|>", color="#aaa",
                        linestyle=":", lw=1.0,
                        connectionstyle="arc3,rad=0.0"),
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )

    # ============== Legend ==============
    legend_y = 0.5
    legend_items = [
        (COLOR_INPUT, "config / external data"),
        (COLOR_PYTHON, "Python driver (in-process)"),
        (COLOR_SHELL, "shell / lifecycle"),
        (COLOR_SERVICE, "runtime services (out-of-process)"),
        (COLOR_AUX, "supporting infra"),
        (COLOR_OUTPUT, "generated artifacts"),
    ]
    x0 = 0.5
    for color, label in legend_items:
        rect = FancyBboxPatch((x0, legend_y), 0.5, 0.5,
                              boxstyle="round,pad=0.02,rounding_size=0.06",
                              facecolor=color, edgecolor=EDGE, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x0 + 0.6, legend_y + 0.25, label,
                va="center", fontsize=9.5)
        x0 += 0.6 + len(label) * 0.09 + 0.5

    fig.savefig(_OUT, dpi=150, bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
