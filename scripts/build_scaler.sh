#!/usr/bin/env bash
#
# build_scaler.sh — pull → fit → compress → store an eosframes scaler.
#
# Given a model identifier and a version, this script:
#   1. pulls the model's precalculations from the isaura store (cloud → local)
#      using the Ersilia reference library as the molecule set,
#   2. exports them to a CSV and fits a type-aware robust scaler with eosframes,
#   3. compresses the fitted transformer artifact into a zip,
#   4. stores it under output/<ref-library>/<model>/<version>/scaler-<major>.zip
#
# Usage:
#   scripts/build_scaler.sh <model_id> <version>
#   PROJECT_NAME=isaura-private scripts/build_scaler.sh eos4e40 v1
#
# Prerequisites:
#   - `isaura` and `eosframes` installed on PATH
#   - local isaura MinIO engine running (`isaura engine --start`)
#
# All progress goes to stderr; stdout prints only the final zip path.

set -euo pipefail

log() { printf '%s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- args -------------------------------------------------------------------
if [[ $# -ne 2 ]]; then
    die "usage: $(basename "$0") <model_id> <version>  (e.g. eos4e40 v1)"
fi

MODEL_ID="$1"
VERSION="$2"
PROJECT_NAME="${PROJECT_NAME:-isaura-public}"

# Mirror the eosframes naming contract: model_id = eos<digit><3 alnum>,
# version = 'v' + integer. Fail fast with a clear message.
[[ "$MODEL_ID" =~ ^eos[0-9][A-Za-z0-9]{3}$ ]] \
    || die "invalid model id '$MODEL_ID' (expected eos<digit><3 alphanumeric>, e.g. eos4e40)"
[[ "$VERSION" =~ ^v[0-9]+$ ]] \
    || die "invalid version '$VERSION' (expected v<integer>, e.g. v1)"

# --- tooling checks ---------------------------------------------------------
command -v isaura    >/dev/null 2>&1 || die "isaura not found on PATH (pip install the isaura package)"
command -v eosframes >/dev/null 2>&1 || die "eosframes not found on PATH (pip install eosframes)"

# --- constants --------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REF_BASENAME="ersilia_reference_library_v0"
REF_LIB="$ROOT/data/${REF_BASENAME}.csv"
[[ -f "$REF_LIB" ]] || die "reference library not found: $REF_LIB"

# --- work dir (convention-valid file names required by eosframes) -----------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

DATA="$WORK/${MODEL_ID}_${VERSION}.csv"
SCALER="$WORK/${MODEL_ID}_${VERSION}_transformer.json"

# --- 1. pull: cloud → local MinIO -------------------------------------------
log "[1/4] pulling ${MODEL_ID} ${VERSION} from ${PROJECT_NAME} …"
isaura pull -i "$REF_LIB" -m "$MODEL_ID" -v "$VERSION" -pn "$PROJECT_NAME" >&2

# --- 2. read/export the precalculations to a CSV ----------------------------
log "[2/4] reading precalculations → $(basename "$DATA") …"
isaura read -i "$REF_LIB" -m "$MODEL_ID" -v "$VERSION" -pn "$PROJECT_NAME" -o "$DATA" >&2

# --- 3. fit the eosframes scaler --------------------------------------------
log "[3/4] fitting eosframes scaler …"
eosframes fit "$DATA" -s "$SCALER" >&2

# --- 4. compress + store ----------------------------------------------------
EOS_MAJOR="$(python -c 'import eosframes; print(eosframes.__version__.split(".")[0])')"
[[ -n "$EOS_MAJOR" ]] || die "could not determine eosframes major version"

OUTDIR="$ROOT/output/${REF_BASENAME}/${MODEL_ID}/${VERSION}"
mkdir -p "$OUTDIR"
ZIP="$OUTDIR/scaler-${EOS_MAJOR}.zip"

log "[4/4] compressing → ${ZIP} …"
rm -f "$ZIP"                       # overwrite on re-run (zip would otherwise append)
# -j flattens paths so the transformer sits at the zip root.
( cd "$WORK" && zip -j -q "$ZIP" "$(basename "$SCALER")" )

log "done."
printf '%s\n' "$ZIP"
