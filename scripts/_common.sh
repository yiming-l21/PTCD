#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTCD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

json_get() {
  local file="$1"
  local key="$2"
  local default_value="${3:-}"
  python - "$file" "$key" "$default_value" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]

if not path.exists():
    print(default)
    raise SystemExit(0)

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(default)
    raise SystemExit(0)

value = data
for part in key.split("."):
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        print(default)
        raise SystemExit(0)

if isinstance(value, bool):
    print("1" if value else "0")
else:
    print(value)
PY
}

ensure_dataset_arg() {
  local dataset="${1:-mvsa-s}"
  case "${dataset}" in
    mvsa-s|mvsa-m|t2015|t2017|masad|tumemo)
      printf '%s\n' "${dataset}"
      ;;
    *)
      echo "[ERR] unknown dataset: ${dataset}" >&2
      echo "      expected one of: mvsa-s, mvsa-m, t2015, t2017, masad, tumemo" >&2
      return 1
      ;;
  esac
}
