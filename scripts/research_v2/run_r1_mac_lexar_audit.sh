#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "BLOCKED: this wrapper is intended for the Mac connected to Lexar" >&2
  exit 20
fi

if [[ ! -d /Volumes/Lexar/ProtoEM-CT ]]; then
  echo "BLOCKED: /Volumes/Lexar/ProtoEM-CT is not mounted" >&2
  exit 21
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

HEAD="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
DIRTY="$(git status --porcelain)"

if [[ "$BRANCH" != "research-v2/r1-pixel-metric-audit" ]]; then
  echo "BLOCKED: wrong branch: $BRANCH" >&2
  exit 22
fi
if [[ -n "$DIRTY" ]]; then
  echo "BLOCKED: repository working tree is not clean" >&2
  git status --short >&2
  exit 23
fi

MANIFEST="/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/manifest.json"
SPLIT="/Volumes/Lexar/ProtoEM-CT/runs/phase2_real_lits_v2/split.json"
[[ -f "$MANIFEST" ]] || { echo "BLOCKED: locked manifest missing" >&2; exit 24; }
[[ -f "$SPLIT" ]] || { echo "BLOCKED: locked split missing" >&2; exit 25; }
command -v uv >/dev/null || { echo "BLOCKED: uv is not installed" >&2; exit 26; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/Volumes/Lexar/ProtoEM-CT/runs/research_v2_r1_real_audit_${STAMP}"
mkdir -p "$OUT"

cat > "$OUT/r1_execution_identity.json" <<EOF
{
  "schema_version": "research_v2_r1_execution_identity.v1",
  "git_commit": "$HEAD",
  "branch": "$BRANCH",
  "working_tree_clean": true,
  "host_os": "macOS",
  "medical_arrays_or_overlays_in_git": false
}
EOF

{
  echo "===== R1 EXECUTION IDENTITY ====="
  echo "git_commit=$HEAD"
  echo "branch=$BRANCH"
  echo "working_tree_clean=true"
  echo ""
  echo "===== PYTHON 3.11 LOCKED ENVIRONMENT ====="
  uv --version
  uv python install 3.11
  uv sync --frozen --python 3.11 --all-groups
  uv run --frozen --python 3.11 python --version
  echo ""
  echo "===== R1 TARGETED FIXTURES ====="
  uv run --frozen --python 3.11 pytest -q \
    tests/unit/test_research_v2_r1_audit.py \
    tests/unit/test_research_v2_r1_lesion_matching.py
} 2>&1 | tee "$OUT/r1_preflight.log"

set +e
uv run --frozen --python 3.11 python scripts/research_v2/run_r1_mac_lexar_audit.py \
  --manifest "$MANIFEST" \
  --split "$SPLIT" \
  --output-dir "$OUT/audit" \
  > >(tee "$OUT/r1_real_audit.stdout.log") \
  2> >(tee "$OUT/r1_real_audit.stderr.log" >&2)
AUDIT_RC=$?
set -e

echo "$AUDIT_RC" > "$OUT/r1_real_audit.exit_code.txt"

if [[ "$AUDIT_RC" -ne 0 ]]; then
  echo "R1_STATUS=BLOCKED"
  echo "R1_EVIDENCE_DIR=$OUT"
  exit "$AUDIT_RC"
fi

cat <<EOF
R1_STATUS=READY_FOR_LOCAL_OVERLAY_REVIEW
R1_EVIDENCE_DIR=$OUT
R1_AUDIT_DIR=$OUT/audit

NEXT:
  Open the 10 PNG files under:
    $OUT/audit/overlays_local_only
  Then run:
    uv run --frozen --python 3.11 python scripts/research_v2/review_r1_overlays.py "$OUT/audit"
  If every overlay is confirmed, package sanitized evidence with:
    uv run --frozen --python 3.11 python scripts/research_v2/package_r1_evidence.py "$OUT/audit"

Do not upload the overlay PNGs or raw NIfTI files.
Upload the sanitized ZIP plus:
  $OUT/r1_execution_identity.json
  $OUT/r1_preflight.log
  $OUT/r1_real_audit.exit_code.txt
EOF
