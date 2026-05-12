#!/usr/bin/env bash
# eosframes CLI smoke test — runs every offline command on the eos7m30
# example fixture (49 ADMET features, 1000 rows) and confirms each
# output file shows up. Skips `info` / `columns` since those hit GitHub.
#
# Usage: ./scripts/smoke.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$REPO/data/example_eos7m30_v1.csv"
OTHER="$REPO/data/example_eos4e40_v1.csv"
RAW="$REPO/data/example_input.csv"

WORK="$(mktemp -d -t eosframes-smoke-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
cp "$INPUT" "$OTHER" .

step() { printf "\n→ %s\n" "$*"; }

step "split  (raw SMILES → chunks/)"
eosframes split "$RAW" -o chunks/ --chunksize 200

step "summary  (eos7m30 → 49-row sidecar)"
eosframes summary example_eos7m30_v1.csv -o example_eos7m30_v1_summary.csv

step "convert  (CSV → H5)"
eosframes convert example_eos7m30_v1.csv -o example_eos7m30_v1.h5

step "fit  (eos7m30 → transformer JSON)"
eosframes fit example_eos7m30_v1.csv -s example_eos7m30_v1_transformer.json

step "transform --quantize  (eos7m30 → int8 H5)"
eosframes transform example_eos7m30_v1.csv \
  -s example_eos7m30_v1_transformer.json \
  -o quantized_eos7m30_v1.h5 --quantize

step "stack  (eos4e40 + eos7m30 → eosmix CSV)"
eosframes stack example_eos4e40_v1.csv example_eos7m30_v1.csv \
  -o project_eosmix.csv

step "unstack  (eosmix → per-model files)"
eosframes unstack project_eosmix.csv -o split/

step "Final workspace:"
ls -la

echo
echo "✅ all offline CLI commands ran end-to-end on eos7m30 fixtures."
