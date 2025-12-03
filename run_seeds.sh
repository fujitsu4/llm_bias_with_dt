# run_seeds.sh
# Author : Zakaria JOUILIL
# Description  : The script:
#  - ignores the first line (timestamp)
#  - reads the SEEDS list on second line
#  - for each seed: if outputs/attention/attention_top5_untrained_seed_${seed}.csv exists -> skip
#    else run the Python script for that seed
#  - stops after --max-runs jobs in this invocation (default 5)
# Usage (Colab):
# !apt-get install dos2unix -y
# !dos2unix run_seeds.sh
# !bash run_seeds.sh /content/llm_bias_with_dt/outputs/attention/seeds_list.txt --max-runs 3


set -euo pipefail

SEEDS_FILE="${1:-seeds_list.txt}"
MAX_RUNS=5

# optional arg --max-runs N
if [[ "${2:-}" == "--max-runs" && -n "${3:-}" ]]; then
  MAX_RUNS="$3"
fi

if [[ ! -f "$SEEDS_FILE" ]]; then
  echo "ERROR: seeds file not found: $SEEDS_FILE"
  exit 2
fi

# read the second line (SEEDS = [...]) and strip prefix + brackets
SEED_LINE=$(sed -n '2p' "$SEEDS_FILE" | sed -e 's/^SEEDS *= *//')
# remove surrounding [ ]
SEED_LINE="${SEED_LINE#[}"
SEED_LINE="${SEED_LINE%]}"

# split by comma into an array
IFS=',' read -ra RAW <<< "$SEED_LINE"

# prepare output dir
OUT_DIR="outputs/attention"
mkdir -p "$OUT_DIR"

runs=0

for raw in "${RAW[@]}"; do
  # trim whitespace
  seed="$(echo "$raw" | xargs)"
  if [[ -z "$seed" ]]; then
    continue
  fi

  out_file="${OUT_DIR}/attention_top5_untrained_seed_${seed}.csv"

  if [[ -f "$out_file" ]]; then
    echo "[SKIP] seed ${seed} -> ${out_file} already exists"
    continue
  fi

  if (( runs >= MAX_RUNS )); then
    echo "[INFO] Reached max runs (${MAX_RUNS}) in this invocation. Stopping."
    exit 0
  fi

  echo "[RUN] seed=${seed} -> launching compute_attention_top5.py"
  python /content/llm_bias_with_dt/src/attention/compute_attention_top5.py --model untrained --seed "${seed}" --input_csv /content/llm_bias_with_dt/outputs/bert/bert_final_features.csv

  # check exit status
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "[ERROR] run for seed ${seed} failed with status ${status}. Stopping."
    exit $status
  fi

  ((runs++))
done

echo "[DONE] Completed up to ${runs} runs in this invocation."
exit 0