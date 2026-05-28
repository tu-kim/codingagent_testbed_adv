#!/usr/bin/env python3
"""Sample GPU (DCGM) + CPU/host (psutil) metrics during agent runs.

Writes one NDJSON line per sample with a wall-clock `ts` that lines up
with OpenCode profile NDJSON's `ts`, so the two timelines can be
joined for offline analysis (e.g. GPU SM-active% overlaid on session
turn boundaries).

GPU fields collected by default (full DCGM `DCGM_FI_*` names):
  Compute / occupancy
    DCGM_FI_PROF_SM_ACTIVE          per-SM active fraction       (0.0-1.0)
    DCGM_FI_PROF_SM_OCCUPANCY       warps active / max warps     (0.0-1.0)
    DCGM_FI_PROF_PIPE_TENSOR_ACTIVE tensor pipe active fraction  (0.0-1.0)
    DCGM_FI_PROF_PIPE_FP32_ACTIVE   FP32 pipe active fraction
    DCGM_FI_PROF_PIPE_FP16_ACTIVE   FP16 pipe active fraction
    DCGM_FI_DEV_GPU_UTIL            legacy NVML 0-100 %
  Memory / bandwidth
    DCGM_FI_PROF_DRAM_ACTIVE        HBM read+write busy fraction (0.0-1.0)
    DCGM_FI_DEV_FB_USED             frame buffer used (MiB)
    DCGM_FI_DEV_FB_TOTAL            frame buffer total (MiB)
    DCGM_FI_DEV_MEM_COPY_UTIL       memory copy engine %         (0-100)
  Interconnect
    DCGM_FI_PROF_PCIE_RX_BYTES      cumulative PCIe RX bytes
    DCGM_FI_PROF_PCIE_TX_BYTES      cumulative PCIe TX bytes
    DCGM_FI_PROF_NVLINK_RX_BYTES    cumulative NVLink RX bytes
    DCGM_FI_PROF_NVLINK_TX_BYTES    cumulative NVLink TX bytes
  Misc
    DCGM_FI_DEV_POWER_USAGE         W
    DCGM_FI_DEV_GPU_TEMP            C
    DCGM_FI_DEV_SM_CLOCK            MHz
    DCGM_FI_DEV_MEM_CLOCK           MHz

The PCIe / NVLink byte counters are CUMULATIVE. Downstream analysis
should take per-sample deltas divided by interval to get bandwidth.

CPU side (via psutil):
  host        cpu_util_pct, load_1min, n_cores, mem (used / total / available)
  processes[] each watched PID gets cpu_util_pct (0-100 × n_cores when
              multi-threaded), rss_bytes, n_threads, name (from .pid stem)

PIDs to watch are auto-discovered from `*.pid` files under the
directory passed to `--pids-from` (testbed.sh writes each component's
PID there). The set is refreshed each sample so newly-spawned
components are picked up and dead PIDs drop out silently.

Usage:
  scripts/monitor_resources.py \\
      --output logs/resource.ndjson \\
      --interval 1.0 \\
      --pids-from logs/

Stop with SIGTERM (testbed.sh down_monitor) or Ctrl-C.

Dependencies:
  - DCGM Python bindings: install `datacenter-gpu-manager` (apt) and
    ensure /usr/local/dcgm/bindings/python3 is reachable.
  - psutil: `pip install psutil` (already a dev dep candidate).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path


# DCGM field NAMES we collect by default. Resolved to integer IDs at
# init time via dcgm_fields module so this list survives library
# version bumps that renumber IDs.
DEFAULT_DCGM_FIELDS = [
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_MEM_COPY_UTIL",
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_NVLINK_RX_BYTES",
    "DCGM_FI_PROF_NVLINK_TX_BYTES",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_FB_TOTAL",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_MEM_CLOCK",
]


# DCGM_FI_PROF_* fields that depend on the perfworks profiling
# subsystem. On DCGM 3.x these are auto-activated when watched via
# samples.WatchFields, but ONLY when they're isolated in their own
# field group (mixing them with non-profiling fields in a single
# group is the configuration where activation gets skipped). Keeping
# them on a dedicated field group + short updateFreq is the
# documented-working recipe.
#
# Fields NOT in this set (PROF_PCIE/NVLINK_*_BYTES, all DCGM_FI_DEV_*)
# are hardware counters / NVML and don't need perfworks. They go on
# the regular field group with the caller's updateFreq.
PROF_SAMPLED_FIELDS = frozenset({
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
})


def _dcgm_candidate_paths() -> list[str]:
    """Where to look for NVIDIA's DCGM Python bindings.

    Honors `DCGM_BINDINGS_PATH` env override first (colon-separated),
    then walks a list of known apt / tarball install layouts. The
    actual import (`import dcgm_fields`) does the final check; we
    only nudge sys.path here so the bindings become findable when
    they exist outside the default Python search path.
    """
    candidates: list[str] = []
    env_override = os.environ.get("DCGM_BINDINGS_PATH", "")
    if env_override:
        candidates.extend(p for p in env_override.split(":") if p)
    candidates += [
        "/usr/local/dcgm/bindings/python3",
        "/usr/local/dcgm/bindings/python",
        # version-suffixed (datacenter-gpu-manager-4 etc)
        "/usr/local/dcgm-4/bindings/python3",
        "/usr/local/dcgm-3/bindings/python3",
        # tarball install (NVIDIA's standalone .deb extracts here)
        "/opt/dcgm/bindings/python3",
        "/opt/dcgm/bindings/python",
        # debian/ubuntu apt layout — `datacenter-gpu-manager-N` package
        # places bindings under /usr/share/<pkgname>/bindings/python3
        # and adds a .pth so user-mode python finds them automatically;
        # under sudo the .pth often isn't picked up (different
        # site-packages or different python binary), so we list the
        # known suffixes explicitly.
        "/usr/share/datacenter-gpu-manager/bindings/python3",
        "/usr/share/datacenter-gpu-manager-4/bindings/python3",
        "/usr/share/datacenter-gpu-manager-3/bindings/python3",
        # pip-installed user wheel locations are already on sys.path
    ]
    # Glob for any version-suffixed dirs we missed under common parents.
    for parent in ("/usr/local", "/opt", "/usr/share"):
        try:
            for pattern in ("dcgm*", "datacenter-gpu-manager*"):
                for entry in Path(parent).glob(pattern):
                    p = entry / "bindings" / "python3"
                    if p.is_dir():
                        candidates.append(str(p))
        except OSError:
            continue
    return candidates


def _import_dcgm():
    """Import NVIDIA's DCGM Python bindings. Tries pip-installed first
    (no path mangling), then a list of known apt/tarball locations,
    then `DCGM_BINDINGS_PATH` env. Raises SystemExit with diagnostic
    info if nothing works."""
    # 1) pip-installed path -- already on sys.path
    try:
        import dcgm_fields  # noqa: F401
        import dcgm_structs
        import pydcgm
        return dcgm_structs, dcgm_fields, pydcgm
    except ImportError:
        pass

    # 2) inject known apt/tarball paths
    tried_paths: list[str] = []
    for p in _dcgm_candidate_paths():
        if not Path(p).is_dir():
            continue
        tried_paths.append(p)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import dcgm_fields  # noqa: F811
        import dcgm_structs  # noqa: F811
        import pydcgm  # noqa: F811
    except ImportError as e:
        raise SystemExit(
            f"DCGM Python bindings not found: {e}\n"
            f"Searched the following directories (existed but didn't yield "
            f"importable bindings):\n  "
            + ("\n  ".join(tried_paths) if tried_paths else "(none)")
            + "\n\nInstall DCGM with `sudo apt install datacenter-gpu-manager` "
            "(or `-4` for newer). If DCGM is installed in a non-standard "
            "location, set DCGM_BINDINGS_PATH=<dir containing dcgm_fields.py>. "
            "Locate manually:\n"
            "  dpkg -L datacenter-gpu-manager | grep python\n"
            "  sudo find /usr /opt -name 'dcgm_fields.py'\n"
        )
    return dcgm_structs, dcgm_fields, pydcgm


def _import_psutil():
    try:
        import psutil
    except ImportError as e:
        raise SystemExit(
            f"psutil not found: {e}\nInstall with `pip install psutil` "
            "(or pass --no-cpu to skip CPU sampling)."
        )
    return psutil


# ---------- GPU side (DCGM) ----------


class DcgmSampler:
    """Wraps a DCGM field group and exposes a single `sample()` method
    that returns the latest values for all watched fields, one entry
    per GPU."""

    def __init__(self, field_names: list[str], update_freq_us: int = 1_000_000):
        ds, df, pd = _import_dcgm()
        self._ds, self._df, self._pd = ds, df, pd

        # Resolve names → ids. Missing names raise immediately so
        # field-name typos blow up at startup, not silently mid-run.
        unknown = [n for n in field_names if not hasattr(df, n)]
        if unknown:
            raise SystemExit(
                f"unknown DCGM field name(s): {unknown}. Check "
                "dcgm_fields.py in the installed DCGM bindings."
            )
        self._field_names = list(field_names)
        self._field_ids = [getattr(df, n) for n in field_names]

        # AUTO mode = embedded host engine (no separate nv-hostengine
        # daemon required for short-lived sampling).
        self._handle = pd.DcgmHandle(
            opMode=ds.DCGM_OPERATION_MODE_AUTO,
            ipAddress=None,
        )
        system = self._handle.GetSystem()
        self._group = pd.DcgmGroup(
            self._handle,
            groupName="testbed_monitor_group",
            groupType=ds.DCGM_GROUP_DEFAULT,  # all GPUs
        )

        # One field group with ALL fields — used for GetLatest. DCGM
        # caches field values per-field (not per-watch), so a single
        # GetLatest on the unified group returns every field regardless
        # of which API watched it.
        self._field_group = pd.DcgmFieldGroup(
            self._handle, name="testbed_monitor_fg", fieldIds=self._field_ids,
        )

        # Split watches by required subsystem. Names in
        # PROF_SAMPLED_FIELDS need the perfworks profiling subsystem
        # activated via dcgmProfWatchFields; otherwise they silently
        # return 0. Everything else (DCGM_FI_DEV_*, PROF_*_BYTES
        # hardware counters) uses the regular dcgmWatchFields path.
        prof_names = [n for n in field_names if n in PROF_SAMPLED_FIELDS]
        regular_names = [n for n in field_names if n not in PROF_SAMPLED_FIELDS]
        self._prof_field_ids = [getattr(df, n) for n in prof_names]
        self._regular_field_ids = [getattr(df, n) for n in regular_names]

        if self._regular_field_ids:
            regular_fg = pd.DcgmFieldGroup(
                self._handle,
                name="testbed_monitor_regular_fg",
                fieldIds=self._regular_field_ids,
            )
            self._group.samples.WatchFields(
                regular_fg,
                updateFreq=update_freq_us,
                maxKeepAge=600.0,
                maxKeepSamples=0,
            )

        if self._prof_field_ids:
            prof_fg = pd.DcgmFieldGroup(
                self._handle,
                name="testbed_monitor_prof_fg",
                fieldIds=self._prof_field_ids,
            )
            # Cap update period at 1s — perfworks windows longer than
            # ~1s average over irrelevantly-large intervals and round
            # short compute bursts down to 0. The samples.WatchFields
            # path here auto-activates perfworks in DCGM 3.x BECAUSE
            # the field group contains ONLY PROF sampled fields (the
            # original bug was mixing them into the regular group,
            # which suppressed activation).
            prof_update_freq = min(update_freq_us, 1_000_000)
            self._group.samples.WatchFields(
                prof_fg,
                updateFreq=prof_update_freq,
                maxKeepAge=600.0,
                maxKeepSamples=0,
            )

        self._gpu_ids = list(system.discovery.GetAllSupportedGpuIds())

    def sample(self) -> list[dict]:
        latest = self._group.samples.GetLatest(self._field_group)
        rows: list[dict] = []
        for gpu_id in self._gpu_ids:
            entry: dict = {"index": int(gpu_id)}
            gpu_data = (
                latest.values.get(gpu_id) if hasattr(latest, "values") else None
            )
            if not gpu_data:
                rows.append(entry)
                continue
            for name, fid in zip(self._field_names, self._field_ids):
                value = _extract_dcgm_value(gpu_data.get(fid))
                if value is None or _is_dcgm_blank(value):
                    continue
                entry[name] = _to_json_value(value)
            rows.append(entry)
        return rows

    def close(self) -> None:
        with suppress(Exception):
            self._handle.Shutdown()


# DCGM sentinel values for missing/unsupported readings.
_DCGM_BLANK_VALUES = {
    -1,
    -2,
    -3,
    -4,
    -5,
    -6,
    0x7FFFFFFFFFFFFFFF,           # max int64 ~ blank int
    9223372036854775792,
}


def _is_dcgm_blank(v) -> bool:
    if isinstance(v, (int, float)) and v in _DCGM_BLANK_VALUES:
        return True
    if isinstance(v, float) and (v != v):  # NaN
        return True
    return False


def _extract_dcgm_value(fv):
    """Drill into the wrapper that `samples.GetLatest(...)` returns.

    Despite the name, that API returns a `DcgmFieldValueTimeSeries`
    (a list of `DcgmFieldValue_v1` records, even when only one sample
    is buffered) rather than a bare scalar. Touch `.value` directly and
    json.dumps() blows up with
        TypeError: Object of type DcgmFieldValueTimeSeries is not JSON serializable

    Defensively support three shapes the DCGM Python bindings have
    used across versions:
      1. TimeSeries with `.values` list  → take the last `.value`
      2. Singleton `DcgmFieldValue_v1`   → take `.value` directly
      3. Already-unwrapped int/float/str → return as-is
    """
    if fv is None:
        return None
    # Shape 1: TimeSeries
    items = getattr(fv, "values", None)
    if items is not None and not isinstance(items, (int, float, str, bytes)):
        try:
            seq = list(items)
        except TypeError:
            seq = None
        if seq:
            last = seq[-1]
            return getattr(last, "value", last)
        # empty series — no sample yet
        if hasattr(fv, "value"):
            return fv.value
        return None
    # Shape 2: singleton FieldValue
    if hasattr(fv, "value"):
        return fv.value
    # Shape 3: bare value
    return fv


def _to_json_value(v):
    """Coerce DCGM-returned values to JSON-serializable Python types.
    Most DCGM fields are int / float. A few (driver_version, model
    name) come back as bytes -> decode. Unknown objects stringify so
    one weird field can't crash the whole NDJSON line."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


# ---------- CPU + per-process (psutil) ----------


class CpuSampler:
    """Tracks host CPU/memory plus a set of watched processes. The PID
    set is refreshed each sample from a directory of `*.pid` files
    (testbed.sh's component PID layout)."""

    def __init__(self, pids_dir: Path | None):
        self._psutil = _import_psutil()
        self._pids_dir = pids_dir
        self._procs: dict[str, "self._psutil.Process"] = {}
        # warm up cpu_percent so the next call returns real numbers
        self._psutil.cpu_percent(interval=None)
        self._refresh_pids()

    def _refresh_pids(self) -> None:
        if not self._pids_dir:
            return
        seen: set[str] = set()
        for f in self._pids_dir.glob("*.pid"):
            name = f.stem
            seen.add(name)
            try:
                pid = int(f.read_text().strip().split("\n", 1)[0])
            except (ValueError, OSError):
                continue
            existing = self._procs.get(name)
            if existing and existing.pid == pid and existing.is_running():
                continue
            try:
                p = self._psutil.Process(pid)
                p.cpu_percent(interval=None)  # warm up per-process
                self._procs[name] = p
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                pass
        # Drop processes whose .pid file disappeared.
        for stale in [n for n in self._procs if n not in seen]:
            self._procs.pop(stale, None)

    def sample(self) -> dict:
        self._refresh_pids()
        mem = self._psutil.virtual_memory()
        host = {
            "cpu_util_pct": self._psutil.cpu_percent(interval=None),
            "load_1min": (os.getloadavg()[0] if hasattr(os, "getloadavg") else None),
            "n_cores": self._psutil.cpu_count(),
            "mem_total_bytes": mem.total,
            "mem_used_bytes": mem.used,
            "mem_available_bytes": mem.available,
        }
        processes: list[dict] = []
        for name in list(self._procs.keys()):
            p = self._procs[name]
            try:
                with p.oneshot():
                    cpu_pct = p.cpu_percent(interval=None)
                    rss = p.memory_info().rss
                    n_threads = p.num_threads()
                processes.append({
                    "name": name,
                    "pid": p.pid,
                    "cpu_util_pct": cpu_pct,
                    "rss_bytes": rss,
                    "n_threads": n_threads,
                })
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                self._procs.pop(name, None)
        return {"host": host, "processes": processes}


# ---------- main loop ----------


def run_sampler(
    output: Path,
    interval_s: float,
    dcgm: DcgmSampler | None,
    cpu: CpuSampler | None,
    stop_fn,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", buffering=1) as f:  # line-buffered
        while not stop_fn():
            sample: dict = {"ts": time.time(), "interval_s": interval_s}
            if dcgm is not None:
                try:
                    sample["gpus"] = dcgm.sample()
                except Exception as e:
                    sample["gpu_error"] = str(e)
            if cpu is not None:
                try:
                    sample.update(cpu.sample())
                except Exception as e:
                    sample["cpu_error"] = str(e)
            f.write(json.dumps(sample) + "\n")
            n += 1
            # Sleep accounting for sample collection time so we don't drift.
            target = sample["ts"] + interval_s
            now = time.time()
            remaining = target - now
            if remaining > 0:
                time.sleep(remaining)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True, type=Path,
                    help="NDJSON output file (one sample per line)")
    ap.add_argument("--interval", type=float, default=1.0, metavar="SECONDS",
                    help="Sampling period (default 1.0s)")
    ap.add_argument("--fields", nargs="*", default=DEFAULT_DCGM_FIELDS,
                    help="DCGM field names to collect (default: rich set)")
    ap.add_argument("--pids-from", type=Path, default=None,
                    help="Directory holding *.pid files for per-process tracking")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Skip DCGM sampling (host CPU/mem only)")
    ap.add_argument("--no-cpu", action="store_true",
                    help="Skip CPU sampling (GPU only)")
    args = ap.parse_args(argv)

    dcgm = None if args.no_gpu else DcgmSampler(args.fields,
                                                  update_freq_us=int(args.interval * 1e6))
    cpu = None if args.no_cpu else CpuSampler(args.pids_from)

    stop = {"flag": False}

    def _handle_sig(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    try:
        n = run_sampler(args.output, args.interval, dcgm, cpu, lambda: stop["flag"])
    finally:
        if dcgm is not None:
            dcgm.close()
    print(f"monitor_resources: wrote {n} samples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
