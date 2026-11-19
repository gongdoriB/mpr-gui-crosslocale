#!/usr/bin/env bash
# Method 6-language chain on ONE RTX 4090 (self-edit): [OCR once] -> per model: snapshot -> self-edit (V-D-S, V-D-H) -> build its solve inputs -> solve.
#   KEYS="qwen3_vl_8b" PILOT=1 bash method6_chain.sh     pilot: OCR of the 24 pilot images, edit + solve the 120 pilot pairs (4 items x 30 directions)
#   KEYS="qwen3_vl_8b qwen3_vl_4b" bash method6_chain.sh  full: OCR of all 948 images (skipped when inventories6.jsonl is complete), edit 4,740 pairs, solve via queue_method6.sh
# Files in /workspace: method6_package_20261119.tar.gz (this script's package; sha in METHOD6_PKG_SHA), method6_solve_addon_20261119.tar.gz, queue_method6.sh.
# Model keys: qwen3_vl_8b qwen3_vl_4b uitars_15_7b (repo mpr-gui-crosslocale)  internvl35_8b (repo mpr-ivl). Each model edits its own questions.
# Status: /workspace/QUEUE_METHOD6_CHAIN_STATUS. Outputs: /workspace/method6_work/. Nothing outside /workspace/mpr-method, /workspace/method6_work,
# <repo>/followup_rel_6lang/method6, <repo>/followup_rel_6lang/results/method6_* and the models' HF caches is written. The instance is NOT stopped.
# Env: PILOT=1, KEEP_MODEL_CACHE (default 1), SKIP_OCR_GPU_CHECK=1 (never by default).
set -Eeuo pipefail
W=/workspace
PKG_TAR=$W/method6_package_20261119.tar.gz; PKG_SHA="${METHOD6_PKG_SHA:?set METHOD6_PKG_SHA}"
SOLVE_TAR=$W/method6_solve_addon_20261119.tar.gz
M=$W/mpr-method; MURL=https://github.com/dltmdwoOS/mpr-gui-crosslocale.git; MBRANCH=method-selective-alignment-dev; MCOMMIT=533633c4d059b04da17bad473b4758575d55e7c5
WORK=$W/method6_work; P=$WORK/package; ST=$W/QUEUE_METHOD6_CHAIN_STATUS
KEYS="${KEYS:?KEYS=...}"; PILOT="${PILOT:-0}"; export KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-1}"
say() { echo "$(date -u +%FT%TZ) method6-chain: $*" | tee -a "$ST"; }
trap 'say "FAILED at ${CURRENT:-?} (exit $?)"' ERR
mkdir -p "$WORK/logs"
# guards (review 2026-11-19): nothing else on the GPU, no other queue, enough disk
CURRENT=guards
if pgrep -f '^(/bin/)?(ba)?sh .*(queue_[A-Za-z0-9_]+\.sh|run_all_overnight\.sh)' >/dev/null; then echo "another queue is running"; exit 1; fi
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc 0-9); [ "${U:-99999}" -lt 1500 ] || { echo "GPU busy (${U} MiB)"; exit 1; }
F=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9); [ "$F" -ge "${MIN_FREE_GB:-30}" ] || { echo "only ${F} GB free (< ${MIN_FREE_GB:-30})"; exit 1; }

CURRENT=package
echo "$PKG_SHA  $PKG_TAR" | sha256sum -c - >/dev/null
rm -rf "$P"; mkdir -p "$P"; tar -xzf "$PKG_TAR" -C "$P"; (cd "$P" && sha256sum -c --quiet SHA256SUMS)
say "start KEYS=[$KEYS] PILOT=$PILOT; package verified"

CURRENT=method_repo
if [ ! -d "$M/.git" ]; then GIT_TERMINAL_PROMPT=0 git clone -q --branch "$MBRANCH" "$MURL" "$M"; fi
(cd "$M" && git cat-file -e "$MCOMMIT^{commit}" 2>/dev/null || git fetch -q origin "$MBRANCH"; cd "$M" && git checkout -q "$MCOMMIT" && [ "$(git rev-parse HEAD)" = "$MCOMMIT" ] && [ -z "$(git status --porcelain --untracked-files=no)" ])
say "method repo $M at ${MCOMMIT:0:7}, clean"

repo_of() { case "$1" in qwen3_vl_8b|qwen3_vl_4b|uitars_15_7b) echo "$W/mpr-gui-crosslocale" ;; internvl35_8b) echo "$W/mpr-ivl" ;; *) return 1 ;; esac; }
spec_of() { case "$1" in
  qwen3_vl_8b)   echo "Qwen/Qwen3-VL-8B-Instruct 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b qwen3_vl_8b_4090_channels_last.yaml" ;;
  qwen3_vl_4b)   echo "Qwen/Qwen3-VL-4B-Instruct ebb281ec70b05090aa6165b016eac8ec08e71b17 qwen3_vl_4b_4090_channels_last.yaml" ;;
  uitars_15_7b)  echo "ByteDance-Seed/UI-TARS-1.5-7B 683d002dd99d8f95104d31e70391a39348857f4e uitars_1_5_7b_4090.yaml" ;;
  internvl35_8b) echo "OpenGVLab/InternVL3_5-8B 9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79 internvl3_5_8b_4090.yaml" ;;
esac; }
DATA=$W/mpr-gui-crosslocale            # images (followup_rel_4lang/assets, followup_rel_6lang/assets) are checked by sha256 in run_ocr6.py
INV=$WORK/inventories6.jsonl; [ "$PILOT" = 1 ] && INV=$WORK/inventories6_pilot.jsonl
ONLY_IMG=(); [ "$PILOT" = 1 ] && ONLY_IMG=(--only-images "$P/inputs/pilot_images.txt")
NIMG=948; [ "$PILOT" = 1 ] && NIMG=24

# ---- OCR (once) ----------------------------------------------------------------------------------------------------------------
ocr_done() { [ -s "$INV" ] && [ "$(wc -l < "$INV")" -ge "$NIMG" ]; }
if ! ocr_done; then
  CURRENT=ocr_env
  OV=$M/.venv-ocr; BASE_SP=$(ls -d "$W"/mpr-gui-crosslocale/.venv/lib/python3.*/site-packages)
  if [ ! -f "$OV/.ready" ]; then
    "$W/mpr-gui-crosslocale/.venv/bin/python" -m venv "$OV"           # same Python as the harness venv whose site-packages is reused
    SP=$(ls -d "$OV"/lib/python3.*/site-packages)
    echo "$BASE_SP" > "$SP/zz_torch_from_harness.pth"      # torch 2.9.1+cu130 and its CUDA 13 libraries from the harness venv (plain path, its .pth files are not run)
    "$OV/bin/python" -m pip install -q --no-cache-dir rapidocr==3.10.0 numpy==1.26.4 opencv-python-headless==4.11.0.86 opt-einsum==3.3.0 requests \
      httpx protobuf Pillow decorator astor networkx typing_extensions safetensors >> "$WORK/logs/ocr_env.log" 2>&1
    "$OV/bin/python" -m pip install -q --no-cache-dir --no-deps --index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ paddlepaddle-gpu==3.3.0 >> "$WORK/logs/ocr_env.log" 2>&1
    "$OV/bin/python" -c 'import torch, paddle, rapidocr, cv2, numpy; assert paddle.__version__=="3.3.0" and numpy.__version__=="1.26.4"; assert paddle.is_compiled_with_cuda(); print("torch", torch.__version__, "paddle", paddle.__version__)' | tee -a "$WORK/logs/ocr_env.log"
    touch "$OV/.ready"
  fi
  CURRENT=ocr_gpu_check
  if [ "${SKIP_OCR_GPU_CHECK:-0}" != 1 ]; then
    "$OV/bin/python" -c 'import torch, paddle; paddle.set_device("gpu:0"); x = paddle.to_tensor([1.0, 2.0]); print("paddle gpu ok", float((x * 2).sum()), paddle.get_device())' 2>&1 | tail -1 | tee -a "$WORK/logs/ocr_env.log"
  fi
  CURRENT=ocr_models
  "$OV/bin/python" "$P/method6/prepare_models6.py" "$P/method6/configs/gpu_ocr_6lang_v1.json" "$M/ppocr_models" >> "$WORK/logs/ocr_models.log" 2>&1
  say "OCR env + PP-OCRv5 files ready"
  CURRENT=ocr
  (cd "$M" && "$OV/bin/python" "$P/method6/run_ocr6.py" --repo "$M" --data-root "$DATA" --inputs "$P/inputs/method6_inputs.jsonl" \
     --config "$P/method6/configs/gpu_ocr_6lang_v1.json" --model-dir "$M/ppocr_models" --out "$INV" "${ONLY_IMG[@]}") >> "$WORK/logs/ocr.log" 2>&1
  say "OCR done: $(tail -1 "$WORK/logs/ocr.log")"
fi
CURRENT=ocr_check   # stop when OCR failed silently (dev: all 680 inventories success)
python3 - "$INV" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
st = Counter(r["status"] for r in rows); by = {}
for r in rows:
    c = by.setdefault(r["gui_language"], Counter()); c[r["status"]] += 1
print("OCR status", dict(st), {k: dict(v) for k, v in sorted(by.items())})
assert st.get("failed", 0) == 0, "OCR failures"
for k, v in by.items():
    assert v.get("empty", 0) <= 0.2 * sum(v.values()), f"too many empty inventories for {k}"
PY
ocr_done

# ---- per model --------------------------------------------------------------------------------------------------------------------
for KEY in $KEYS; do
  R=$(repo_of "$KEY"); read -r MID REV CFG <<<"$(spec_of "$KEY")"
  export HF_HOME="$R/.cache/huggingface" XDG_CACHE_HOME="$R/.cache" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
  SNAP=$HF_HOME/hub/models--${MID//\//--}/snapshots/$REV
  CURRENT=snapshot_$KEY
  [ -s "$SNAP/config.json" ] || (cd "$R" && .venv/bin/hf download "$MID" --revision "$REV" >> "$WORK/logs/download_$KEY.log" 2>&1)
  [ -s "$SNAP/config.json" ]
  say "$KEY snapshot ready ($MID @ ${REV:0:8})"
  OUTE=$WORK/edit_$KEY; PAIRS=(); NP=4740
  [ "$PILOT" = 1 ] && { OUTE=$WORK/edit_${KEY}_pilot; PAIRS=(--pairs "$P/inputs/pilot_pairs.txt"); NP=120; }
  CURRENT=edit_$KEY
  if [ ! -f "$OUTE/COMPLETE" ]; then
    say "$KEY self-edit started ($NP pairs, V-D-S + V-D-H)"
    (cd "$M" && HF_HUB_OFFLINE=1 "$R/.venv/bin/python" -u "$P/method6/run_edit6.py" --model "$KEY" --repo "$M" --inputs "$P/inputs/method6_inputs.jsonl" \
       --inventories "$INV" --out "$OUTE" --editor-path "$SNAP" "${PAIRS[@]}") >> "$WORK/logs/edit_$KEY.log" 2>&1
  fi
  say "$KEY self-edit done: $(python3 -c "import json;c=json.load(open('$OUTE/cost.json'));print({k:c[k] for k in ('wall_seconds_this_invocation','editor_calls','cache_hits','output_limit_reached')})")"
  CURRENT=build_$KEY
  tar -xzf "$SOLVE_TAR" -C "$R"; (cd "$R/followup_rel_6lang/method6" && sha256sum -c --quiet SHA256SUMS)
  "$R/.venv/bin/python" "$R/followup_rel_6lang/method6/build_method6_conditions.py" "$KEY" "$OUTE" --pairs-expected "$NP" > "$WORK/logs/build_$KEY.log" 2>&1
  say "$KEY solve inputs built: $(python3 -c "import json;l=json.load(open('$R/followup_rel_6lang/method6/inputs/$KEY/method6_lock.json'));print(l['rows'], l['stats'])")"
  CURRENT=solve_$KEY
  if [ "$PILOT" = 1 ]; then
    O=$R/followup_rel_6lang/results/method6_pilot_$KEY
    (cd "$R" && source .venv/bin/activate && python followup_rel_6lang/method6/run_method6.py --model "$KEY" --model-config "followup_rel_6lang/method6/models/$CFG" \
       --output-dir "$O" --resume && python followup_rel_6lang/method6/check_method6_gate.py smoke "$O" --model "$KEY" --batch "$(python3 -c "import json;print(json.load(open('followup_rel_6lang/method6/contracts/'+{'qwen3_vl_8b':'ref_qwen3_rel_20261107.json','qwen3_vl_4b':'ref_qwen3_4b_expA_20261117.json','uitars_15_7b':'ref_uitars_rel_20261114.json','internvl35_8b':'ref_internvl35_rel_20261109.json'}['$KEY']))['batch_size'])")" --full-n 9480) >> "$WORK/logs/solve_pilot_$KEY.log" 2>&1 || true
    say "$KEY pilot solve: $(python3 -c "import json;g=json.load(open('$O/gate_smoke.json'));print(g['status'], g['by_condition(n,correct)'], 'unparsed', g['parse_failures'], 'ref', {k:(v['label_agreement'],v['common_rows']) for k,v in g['references'].items()}, 'ETA full h', g['timing'].get('eta_full_hours'), g['problems'])" 2>&1 | tail -1)"
  else
    ONLY="$KEY" bash "$W/queue_method6.sh" >> "$WORK/logs/queue_$KEY.log" 2>&1 || true
    say "$KEY solve queue: $(tail -1 "$W/QUEUE_METHOD6_STATUS")"
  fi
done
CURRENT=package_out
OUTT=$W/queue_results/method6_chain_$( [ "$PILOT" = 1 ] && echo pilot || echo full )_$(date -u +%Y%m%d_%H%M).tar.gz
mkdir -p "$W/queue_results"
tar -czf "$OUTT" -C "$WORK" --exclude='*/cache' . && sha256sum "$OUTT" | tee -a "$ST"
say "FINISHED KEYS=[$KEYS] PILOT=$PILOT; package $OUTT"
