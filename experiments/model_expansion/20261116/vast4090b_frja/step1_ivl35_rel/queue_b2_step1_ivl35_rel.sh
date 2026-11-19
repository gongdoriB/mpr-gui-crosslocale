#!/usr/bin/env bash
# STEP 1 of the vast4090b fr/ja queue: InternVL3.5-8B (OpenGVLab/InternVL3_5-8B), REL O/R/C/F `new_only` (29,148 condition rows -> 26,057 canonical inferences),
# batch size 1, ONE RTX 4090, in the EXISTING /workspace/mpr-ivl. No new addon: it uses the files the InternVL2.5 REL6 queue already unpacked there
# (run_rel_followup_6lang.py / rel_followup_6lang.py / run_inference_6lang.py / data_bundle_6lang.py / ivl25_rel6/check_rel6_gate.py, all verified by sha256 first).
# Conditions of the earlier 4-language InternVL3.5-8B REL run (run_contract_internvl35_rel_20261109.json): config internvl3_5_8b_4090.yaml (eager, max_num 8, 2304 visual tokens),
# batch_size 1, strict_batch_size false (so NO --strict-batch-size), same 12 inference-code hashes as the repo here, same system prompt, torch 2.9.1+cu130 / transformers 4.57.6 / timm 1.0.31.
# Usage (tmux, /workspace):  bash queue_b2_step1_ivl35_rel.sh --plan   |   bash queue_b2_step1_ivl35_rel.sh
# Env: KEEP_MODEL_CACHE=1, FORCE_SLOW_NET=1, IGNORE_ETA=1, MIN_FREE_GB (30), NET_MIN_MBPS (20), REL_H (full step limit, hours, default 9).
# Status /workspace/QUEUE_B2_STATUS (shared name with the later full queue), done marker /workspace/QUEUE_B2_DONE. Rerun = resume. Nothing is written into followup_rel_4lang/results.
set -uo pipefail
W="${W:-/workspace}"
KEY=internvl35_8b
MODEL_ID=OpenGVLab/InternVL3_5-8B
REVISION=9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79       # earlier REL (2026-11-09) and Exp A (2026-11-14) runs
CFG_SHA=7fed517cf7905dd8158487e25754b716c48c31c55b91469381d0332e4beebf03   # 01_code/configs/models/internvl3_5_8b_4090.yaml of the earlier runs
SYS_SHA=956a386091c7a4a1e92547599e9edddda5ad4c9f9656bcddf842da1e69a26709   # sha256 of the system prompt in the earlier run contract
BASE_COMMIT=666440011f538cd9336794c9965429de640843c3
N_FULL=26057; ROWS_FULL=29148; OUTNAME=rel6new_internvl35_8b
MODEL_MIN_BYTES="${MODEL_MIN_BYTES:-16000000000}"; MODEL_MAX_BYTES="${MODEL_MAX_BYTES:-18000000000}"   # exact byte sum not known locally (확인 필요); ~17.07 GB on disk in an earlier queue note
MIN_FREE_GB="${MIN_FREE_GB:-30}"; NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"; GPU_BUSY_MIB="${GPU_BUSY_MIB:-1500}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}" EXPECT_TORCH="${EXPECT_TORCH:-2.9.1+cu130}"
T_SETUP=$((60*60)); T_MODEL=$((120*60)); T_SMOKE=$((40*60)); T_FULL=$(( ${REL_H:-9} * 3600 ))
OTHERS="${OTHERS_RE:-run_all_overnight\.sh|queue_[A-Za-z0-9_]+\.sh}"; SELF_RE='queue_b2_[A-Za-z0-9_]*\.sh'
R="$W/mpr-ivl"; RES="$W/queue_results/b2_step1_ivl35_rel"; LOG="$RES/logs"
ST="$W/QUEUE_B2_STATUS"; DONE="$W/QUEUE_B2_DONE"
OUTD="$R/followup_rel_6lang/results/$OUTNAME"
CACHE_DIR="$R/.cache/huggingface/hub/models--${MODEL_ID//\//--}"
DAY=$(date -u +%Y%m%d); FAILED=(); FINISHED=0
PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1
say() { local l; l="$(date -u +%FT%TZ) b2_step1_ivl35_rel: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }
if [ "$PLAN_ONLY" = 1 ]; then cat <<PLAN
Plan (nothing is executed):
 0. refuse to start (no waiting) if another queue/overnight script runs [$OTHERS], a compute process uses the GPU, or GPU memory used >= ${GPU_BUSY_MIB} MiB; flock against a second copy;
    network check (15 s ranged download from huggingface.co; STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1)
 1. checks: RTX 4090; free disk >= ${MIN_FREE_GB} GB (skipped when the model is cached); $R with .git and .venv
 2. setup (limit $((T_SETUP/60)) min): the InternVL2.5 REL6 files in $R are present and match ivl25_rel6/SHA256SUMS; internvl3_5_8b_4090.yaml sha256 = ${CFG_SHA:0:12}...;
    venv equals the earlier run (torch \$EXPECT_TORCH, transformers 4.57.6, accelerate 1.15.0, timm 1.0.31, sentencepiece 0.2.0, torchvision 0.24.1+cu130); the 12 inference-code hashes and the system prompt equal the earlier InternVL3.5 REL run contract; 4-/6-language data + images verify by sha256
 3. model (limit $((T_MODEL/60)) min): hf download $MODEL_ID @ $REVISION (3 tries); 4 safetensors shards, byte sum in ${MODEL_MIN_BYTES}..${MODEL_MAX_BYTES} (exact size 확인 필요); tokenizer: A,B,C,D single tokens with ids 32,33,34,35 (as in the earlier run)
 4. smoke (--max-items 2, new_only, batch 1, no --strict-batch-size as in the earlier run) + gate (visual tokens 1792..2304, labels not all identical, A-D mass >= 0.5, parse failures <= 50 %)
 5. full: $N_FULL inferences / $ROWS_FULL condition rows, --resume (limit $((T_FULL/3600)) h; not started when the smoke ETA is longer, IGNORE_ETA=1 overrides) + completion gate
 6. package $RES/b2_step1_ivl35_rel_results_<date>.tar.gz (+ .sha256, + per-file sha256 list); delete this model's cache only after success (KEEP_MODEL_CACHE=1 keeps); write $DONE (OK | FAILED steps: ...)
 Results: $OUTD (never followup_rel_4lang/results). ETA (estimate): earlier InternVL3.5-8B REL median 0.748 s/inf on a 4090 -> 26,057 x 0.75 = 5.4 h + download/load ~0.3 h.
PLAN
exit 0; fi
mkdir -p "$LOG"
command -v flock >/dev/null && { exec 9> "$W/.queue_b2.lock"; flock -n 9 || { echo "another queue_b2 script is already running"; exit 1; }; }
retry() { local n=$1 i; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "retry $i/$n failed: $*" >&2; [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i )); done; return 1; }
step() { local name=$1 limit=$2 fn=$3 rc
  say "step $name started (limit $((limit/60)) min)"
  timeout -k 60 "$limit" bash -c "set -Eeo pipefail; $fn" >> "$LOG/$name.log" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then say "step $name ok"; return 0; fi
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then say "step $name TIMEOUT"
  elif [ $rc -eq 3 ] && [ "$name" = full ]; then say "STOP step full: smoke ETA longer than the step limit (IGNORE_ETA=1 overrides)."; FAILED+=("full_eta"); return $rc
  else say "step $name FAILED (exit $rc), see $LOG/$name.log"; fi
  FAILED+=("$name"); return $rc; }
package() { local out="$RES/b2_step1_ivl35_rel_results_${DAY}.tar.gz" stage="$RES/_stage"
  rm -rf "$stage"; mkdir -p "$stage"
  if [ -d "$OUTD" ]; then (cd "$R" && find "followup_rel_6lang/results/$OUTNAME" -type f -exec sha256sum {} + > "$stage/results_files.sha256") 2>/dev/null || true; fi
  (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
  [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
  cp -r "$LOG" "$stage/queue_logs" 2>/dev/null || true; cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true; df -h "$W" > "$stage/disk_end.txt" 2>&1 || true
  if [ -d "$OUTD" ]; then tar -czf "$out" -C "$R" "followup_rel_6lang/results/$OUTNAME" -C "$stage" . ; else tar -czf "$out" -C "$stage" . ; fi \
    && sha256sum "$out" > "$out.sha256" && say "packaged $out ($(du -h "$out" | cut -f1)); sha256 $(cut -d' ' -f1 "$out.sha256")"
  local rc=$?; rm -rf "$stage"; return $rc; }
finish() { [ "$FINISHED" = 1 ] && return 0; FINISHED=1
  local verdict="OK" pk=0; package && pk=1 || say "packaging FAILED"
  [ "$pk" = 0 ] && FAILED+=("package")
  [ ${#FAILED[@]} -gt 0 ] && verdict="FAILED steps: ${FAILED[*]}"
  if [ -d "$CACHE_DIR" ]; then
    if [ "${KEEP_MODEL_CACHE:-0}" = 1 ]; then say "model cache kept (KEEP_MODEL_CACHE=1)"
    elif [ "$verdict" = OK ]; then rm -rf "$CACHE_DIR"; say "model cache deleted; free disk: $(df -h --output=avail "$W" | tail -1 | tr -d ' ')"
    else say "model cache KEPT (not finished); a rerun resumes without downloading"; fi; fi
  say "FINISHED $verdict. Results: $OUTD ; package $RES/b2_step1_ivl35_rel_results_${DAY}.tar.gz"
  echo "$verdict" > "$DONE"; }
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE_B2_DONE written, rerun to resume"; FINISHED=1; exit 130' INT TERM
abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }
rm -f "$DONE"
say "queue_b2_step1_ivl35_rel started ($KEY, new_only, batch 1)"
# ---- 0. nothing else may run ------------------------------------------------------------------
others=""; for p in $(pgrep -f "^(/bin/)?(ba)?sh .*($OTHERS)" 2>/dev/null); do [ "$p" = "$$" ] && continue; a=$(ps -o args= -p "$p" 2>/dev/null) || continue; echo "$a" | grep -q -E "$SELF_RE" && continue; others="$others $p:$(echo "$a" | cut -c1-60)"; done
[ -n "$others" ] && abort others "another queue/overnight script is running ($others). Nothing was started."
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v -i 'no running' | sed '/^$/d')
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
[ -z "$used" ] && abort gpu "nvidia-smi not available"
if [ -n "$apps" ] || [ "$used" -ge "$GPU_BUSY_MIB" ]; then abort gpu "the GPU is in use (${used} MiB; processes: $(echo "$apps" | head -3 | tr '\n' ';')). Nothing was started."; fi
say "GPU idle (${used} MiB, no compute process)"
net_speed() { local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'; }
hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/$MODEL_ID/resolve/$REVISION/model-00001-of-00004.safetensors}")
say "network check (${NET_SECS}s): huggingface.co ${hf} MB/s; minimum $NET_MIN_MBPS MB/s"
printf 'huggingface.co %s MB/s\n' "$hf" > "$RES/net_check.txt"; slow=0
case "$hf" in ERR:4*) say "WARNING: test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;; ERR*) slow=1 ;; *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;; esac
if [ "$slow" = 1 ] && [ "${FORCE_SLOW_NET:-0}" != 1 ]; then abort network "network too slow or unreachable. Nothing was downloaded. FORCE_SLOW_NET=1 overrides"; fi
# ---- 1. checks -------------------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$LOG/gpu.log" 2>&1
grep -q 4090 "$LOG/gpu.log" || abort gpu "GPU is not an RTX 4090 ($(cat "$LOG/gpu.log"))"
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9); df -h "$W" > "$LOG/disk_start.log"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || [ -s "$CACHE_DIR/snapshots/$REVISION/config.json" ] || abort disk "only ${FREE_GB} GB free in $W (need ${MIN_FREE_GB}+)"
{ [ -d "$R/.git" ] && [ -f "$R/.venv/bin/activate" ]; } || abort repo "$R (repo with .venv) does not exist"
say "checks ok: $(cat "$LOG/gpu.log"); ${FREE_GB} GB free"
# ---- 2. setup --------------------------------------------------------------------------------
setup() {
  cd "$R"
  git cat-file -e "$BASE_COMMIT^{commit}"; git merge-base --is-ancestor "$BASE_COMMIT" HEAD
  grep -q 'INTERNVL_FAMILIES' 01_code/src/mpr_crosslocale/inference/run_cross_locale.py
  ( cd followup_rel_6lang/ivl25_rel6 && sha256sum -c SHA256SUMS )       # the unchanged InternVL2.5 REL6 files (modules, runner, data, gate)
  [ "$(sha256sum 01_code/configs/models/internvl3_5_8b_4090.yaml | cut -d' ' -f1)" = "$CFG_SHA" ]
  export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch"
  source .venv/bin/activate
  REPO_DIR="$PWD" python - <<'PY'
import importlib.metadata as md, hashlib, json, os
from pathlib import Path
import yaml, mpr_crosslocale
want = {"torch": os.environ["EXPECT_TORCH"], "transformers": "4.57.6", "accelerate": "1.15.0", "timm": "1.0.31", "sentencepiece": "0.2.0", "torchvision": "0.24.1+cu130"}
for k, v in want.items():
    if k in os.environ.get("EXPECT_SKIP", "").split(","): continue          # local test hook
    assert md.version(k) == v, f"{k} {md.version(k)} != {v} (earlier InternVL3.5 run); nothing is installed here"
assert os.path.realpath(mpr_crosslocale.__file__).startswith(os.path.realpath(os.environ["REPO_DIR"]))
from mpr_crosslocale.inference.run_rel_followup import code_identity
from mpr_crosslocale.inference.system_prompts import build_system_prompt
ref = json.loads(os.environ["CODE_REF"])
now = code_identity()
diff = sorted(k for k in set(now) | set(ref) if not k.startswith("run_rel_T") and ref.get(k) != now.get(k))
assert not diff, f"inference code differs from the earlier InternVL3.5 REL run: {diff}"
sp, _, tpl = build_system_prompt("english_fixed", "en")
assert hashlib.sha256(sp.encode()).hexdigest() == os.environ["SYS_SHA"] and tpl == "cross_locale_system_v1", "system prompt differs"
cfg = yaml.safe_load(Path("01_code/configs/models/internvl3_5_8b_4090.yaml").read_text())
assert cfg["model_id"] == os.environ["MODEL_ID"] and cfg["revision"] == os.environ["REVISION"] and cfg["attn_implementation"] == "eager" and cfg["max_num"] == 8, cfg
print("venv, 12 inference-code hashes, system prompt and config equal the earlier InternVL3.5 REL run")
PY
  python -c 'import mpr_crosslocale.data.rel_followup_6lang, mpr_crosslocale.inference.run_rel_followup_6lang; print("6-language modules import ok")'
  python followup_rel_4lang/data_bundle.py verify || { python followup_rel_4lang/data_bundle.py unpack; retry 6 python followup_rel_4lang/data_bundle.py fetch-images --workers 2; python followup_rel_4lang/data_bundle.py verify; }
  python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang > "$LOG/bundle6_verify.log" \
    || { retry 6 python followup_rel_6lang/data_bundle_6lang.py fetch-images --data-root followup_rel_6lang --workers 2; python followup_rel_6lang/data_bundle_6lang.py verify --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang; }
}
model() {
  cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
  source .venv/bin/activate
  retry 3 hf download "$MODEL_ID" --revision "$REVISION"
  MODEL_ID="$MODEL_ID" REVISION="$REVISION" RES="$RES" python - <<'PY'
import glob, json, os
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MODEL_ID"], revision=os.environ["REVISION"], local_files_only=True)
fs = glob.glob(os.path.join(snap, "model-*.safetensors")); size = sum(os.path.getsize(p) for p in fs)
print("snapshot", snap, "safetensors files", len(fs), "bytes", size)
assert len(fs) == 4 and int(os.environ["MIN_B"]) <= size <= int(os.environ["MAX_B"]), (len(fs), size)
cfg = json.load(open(os.path.join(snap, "config.json")))
print("architectures", cfg.get("architectures"), "| llm", (cfg.get("llm_config") or {}).get("architectures"))   # printed only, not asserted (확인 필요)
open(os.path.join(os.environ["RES"], "model_size.txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\n")
from mpr_crosslocale.models.internvl25 import load_internvl_tokenizer
tok = load_internvl_tokenizer(os.environ["MODEL_ID"], revision=os.environ["REVISION"], trust_remote_code=True)
ids = {l: list(tok.encode(l, add_special_tokens=False)) for l in "ABCD"}
print("label ids", ids)
assert ids == {"A": [32], "B": [33], "C": [34], "D": [35]}, ids   # as in the earlier InternVL3.5 predictions (label_token_ids)
PY
  du -sh "$HF_HOME"
}
RUNCMD() { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false; source .venv/bin/activate; }
MOCKF=""; [ "${LOCAL_MOCK:-0}" = 1 ] && MOCKF="--mock-model"
rel_cmd() { python followup_rel_6lang/run_inference_6lang.py --model-config 01_code/configs/models/internvl3_5_8b_4090.yaml --data-root followup_rel_6lang --legacy-data-root followup_rel_4lang --batch-size 1 --languages-subset new_only "$@"; }
preflight_plan() {   # dry-run plan must be exactly the new_only plan
  rel_cmd --dry-run --output-dir "$OUTD/dry_run" > "$OUTD/logs/dry_run.log"
  python - "$OUTD/dry_run/dry_run_plan.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert (p["actual_inferences"], p["condition_rows"], p["items"], p["batch_size"], p["languages_subset"]) == (26057, 29148, 366, 1, "new_only"), p
print("plan ok", p["actual_inferences"], p["condition_rows"], p["experiment_fingerprint"][:12])
PY
}
smoke() { RUNCMD; mkdir -p "$OUTD/logs"; preflight_plan
  if ! grep -q '"status": "passed"' "$OUTD/smoke/gate_rel_smoke.json" 2>/dev/null; then
    rel_cmd --max-items 2 --resume $MOCKF --output-dir "$OUTD/smoke" 2>&1 | tee -a "$OUTD/logs/smoke.log"; [ "${PIPESTATUS[0]}" = 0 ]
    python followup_rel_6lang/ivl25_rel6/check_rel6_gate.py rel smoke "$OUTD/smoke" --model "$KEY" --full-n "$N_FULL" > "$OUTD/logs/smoke_gate.log"
  fi
}
full() { RUNCMD; mkdir -p "$OUTD/logs"
  grep -q '"status": "passed"' "$OUTD/full/gate_rel_complete.json" 2>/dev/null && { echo "full already done"; return 0; }
  preflight_plan
  if [ "${IGNORE_ETA:-0}" != 1 ]; then
    python -c "$ETA_PY" "$OUTD/smoke/gate_rel_smoke.json" "$T_FULL" "$OUTD/full/predictions.jsonl" "$N_FULL"; rc=$?
    [ "$rc" = 3 ] && exit 3
  fi
  rel_cmd --resume $MOCKF --output-dir "$OUTD/full" 2>&1 | tee -a "$OUTD/logs/full.log"; [ "${PIPESTATUS[0]}" = 0 ]
  python followup_rel_6lang/ivl25_rel6/check_rel6_gate.py rel complete "$OUTD/full" "$N_FULL" --model "$KEY" --full-n "$N_FULL" --full-rows "$ROWS_FULL" > "$OUTD/logs/complete_gate.log"
}
smoke_report() { python3 - "$OUTD/smoke/gate_rel_smoke.json" <<'PY' 2>/dev/null
import json, sys
try: g = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
s, t = g.get("sanity", {}), g.get("timing", {})
l = f"acc {s.get('accuracy')} unparsed {g.get('parse_failures')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h visual tokens {s.get('visual_token_count')}"
if g.get("warnings"): l += " WARN " + "; ".join(g["warnings"])
if g.get("problems"): l += " PROBLEMS " + "; ".join(g["problems"])
print(l, end=" | ")
PY
}
export -f retry setup model RUNCMD rel_cmd preflight_plan smoke full
export W R RES LOG OUTD OUTNAME BASE_COMMIT KEY MODEL_ID REVISION CFG_SHA SYS_SHA T_FULL N_FULL ROWS_FULL MOCKF MIN_B="$MODEL_MIN_BYTES" MAX_B="$MODEL_MAX_BYTES"
export ETA_PY='import json, sys
gate, limit, preds, total = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
try: eta = json.load(open(gate))["timing"]["eta_full_seconds"]
except Exception: eta = None
if eta is None: print("no ETA in smoke gate; starting without ETA check"); sys.exit(0)
try: done = sum(1 for l in open(preds) if chr(34)+"status"+chr(34)+": "+chr(34)+"success"+chr(34) in l)
except OSError: done = 0
rem = eta * max(0, total - done) / total
print("ETA %.1f h vs limit %.1f h" % (rem/3600, limit/3600)); sys.exit(3 if rem > limit else 0)'
export CODE_REF='{"run_rel_followup.py":"fbdb1b46591ed9c88e3d6f76119876a864ed549b665057af8e25cf92233fae6f","qwen25vl.py":"3f0e6a2f504488065fe799fc8aea0f75663ad0ea7e91e1ecb7c51781184bd3ad","qwen3_memory.py":"fe00351f368cdaca85418f7d0f3d03b638cffd4c61a86258a875e646ea66c4b6","internvl25.py":"f618b729970e1567d8776e5757feaf6df59d07db24bf8f75973502e35fa1a095","prompts.py":"5db5085c28e4f7b4495992008bdebe30f47f4809c307bacd47ffd22c91d73ba9","system_prompts.py":"afbfe2dc4ea3e69c6d90bb2b9f65490e266343ef6533468857b36fbb843bf554","run_cross_locale.py":"c051c390642dd469674d80d533c20a3f7cf77946f08918c408e7cbf68cdde559","answer_parser.py":"f13ffd717563ddc41f085c87b8dfceba9cf80c8151c8c3aed6da66a7523817a0","label_scoring.py":"b46e5c7e361030ef3cda661f2d8669942994a33626dafabbbfd3e4e74d00aecd","option_scoring.py":"f37979a48c5fe2f3e8450627ed7282e93135ab3de497df787e109da54a15727c","runtime.py":"42c831898441b8fa42d12a81003657489d0fe3a63a13eaaa9dbd3492e3ff1dd0","rel_followup.py":"67580d17f4bc099de2bfcc3e834552f8d18f14aaac090ed477cf5d7f1631a244"}'
mkdir -p "$OUTD/logs" 2>/dev/null
if step setup "$T_SETUP" setup && step model "$T_MODEL" model; then
  if step smoke "$T_SMOKE" smoke; then say "smoke gate: $(smoke_report)"; step full "$T_FULL" full || true
  else say "smoke failed: $(smoke_report)see $LOG/smoke.log"; fi
fi
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
