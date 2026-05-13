#!/usr/bin/env bash
# Apply (or revert) testbed-side patches against the vendored opencode/
# submodule. The submodule pointer stays at the upstream-tagged commit
# (e.g. v1.14.41) and patches live under deploy/patches/ so the parent
# repo can be cloned without depending on a local-only opencode SHA.
#
# Usage:
#   scripts/apply_opencode_patches.sh           # apply all patches (idempotent)
#   scripts/apply_opencode_patches.sh --check   # report which patches are pending
#   scripts/apply_opencode_patches.sh --revert  # reverse-apply all patches

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$REPO_ROOT/deploy/patches"
SUBMODULE="$REPO_ROOT/opencode"

if [[ ! -d "$SUBMODULE/.git" && ! -f "$SUBMODULE/.git" ]]; then
  echo "opencode submodule not initialized at $SUBMODULE" >&2
  echo "  run: git submodule update --init opencode" >&2
  exit 1
fi

mode="apply"
case "${1-}" in
  --revert) mode="revert" ;;
  --check)  mode="check" ;;
  "")       mode="apply" ;;
  *) echo "usage: $(basename "$0") [--check|--revert]" >&2; exit 2 ;;
esac

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
shopt -u nullglob

if [[ ${#patches[@]} -eq 0 ]]; then
  echo "no patches in $PATCH_DIR"
  exit 0
fi

for p in "${patches[@]}"; do
  base="$(basename "$p")"
  case "$mode" in
    check)
      if git -C "$SUBMODULE" apply --check "$p" >/dev/null 2>&1; then
        echo "$base: PENDING (forward-apply would succeed)"
      elif git -C "$SUBMODULE" apply --check --reverse "$p" >/dev/null 2>&1; then
        echo "$base: APPLIED"
      else
        echo "$base: CONFLICT (neither forward nor reverse applies cleanly)"
      fi
      ;;
    apply)
      # Idempotent: skip if already applied (reverse-check succeeds means it's in).
      if git -C "$SUBMODULE" apply --check --reverse "$p" >/dev/null 2>&1; then
        echo "$base: already applied, skipping"
        continue
      fi
      if ! git -C "$SUBMODULE" apply --check "$p" >/dev/null 2>&1; then
        echo "$base: cannot apply -- submodule may be at unexpected SHA" >&2
        git -C "$SUBMODULE" log -1 --oneline >&2
        exit 1
      fi
      git -C "$SUBMODULE" apply "$p"
      echo "$base: applied"
      ;;
    revert)
      if git -C "$SUBMODULE" apply --check "$p" >/dev/null 2>&1; then
        echo "$base: not applied, skipping"
        continue
      fi
      if ! git -C "$SUBMODULE" apply --check --reverse "$p" >/dev/null 2>&1; then
        echo "$base: cannot reverse-apply" >&2
        exit 1
      fi
      git -C "$SUBMODULE" apply --reverse "$p"
      echo "$base: reverted"
      ;;
  esac
done
