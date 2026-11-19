#!/usr/bin/env bash
# InternVL2.5-8B (OpenGVLab/InternVL2_5-8B, language model InternLM2.5-7B-chat), Experiment A only (O/T/R, 1,896 pairs x 3 = 5,688
# inferences, batch size 1) on the Vast.ai RTX 4090, run as the LAST item of the queue chain:
#   wait for /workspace/QUEUE6_DONE (written at the very end of queue_6x6.sh) -> no other queue/overnight process -> GPU idle -> network check
#   -> checks -> setup in the EXISTING /workspace/mpr-ivl (addon unpack, repo/venv/data checks) -> pinned model
#   -> smoke (108 inferences) + gate -> Exp A full (5,688) + gate -> package -> delete this model's cache -> /workspace/QUEUE7_DONE
# Usage (inside tmux, in /workspace, with internvl25_expA_addon_20261114.tar.gz next to this script):
#   bash queue_internvl25_expA.sh --plan   # only print what would run; changes nothing
#   bash queue_internvl25_expA.sh          # waits for QUEUE6_DONE first
# Environment: WAIT_FOR (default $W/QUEUE6_DONE), MAX_WAIT_H (default 168), FORCE_SLOW_NET=1, IGNORE_ETA=1, KEEP_MODEL_CACHE=1,
#   MIN_FREE_GB (default 30), NET_MIN_MBPS (default 20).
# Rerunning resumes (addon unpack is idempotent, predictions are reused). The instance is NOT stopped or destroyed.
# Progress: tail /workspace/QUEUE7_STATUS ; per-step logs: /workspace/queue_results/internvl25_expA/logs/
# NOTE (r1, 2026-11-15): if queue_6x6.sh STOPs early it writes no QUEUE6_DONE; this script now continues when queue_6x6.sh is not running and its last status line is STOP/interrupted/FINISHED (2 checks). Older note: if it stopped
# early this script keeps waiting (and says so in the status every 30 min) until MAX_WAIT_H; then check by hand and, if the GPU is free,
# `touch /workspace/QUEUE6_DONE`. queue_6x6.sh deletes a stale QUEUE6_DONE when it starts; do not start this script after a stale file.
# Only this model's cache folder is deleted; /workspace/mpr-ivl venv, other caches and other results are not touched.
set -uo pipefail

W="${W:-/workspace}"
BASE_COMMIT=666440011f538cd9336794c9965429de640843c3   # mpr-ivl = this commit + 5 InternVL3.5 patches (0001-0005); 0001 also supports internvl2_5
MODEL_ID=OpenGVLab/InternVL2_5-8B
REVISION=e9e4c0dc1db56bfab10458671519b7fa3dd29463      # HF main of 2025-03-25 (checked via HF API 2026-11-14); same as the 2026-10-05 REL run
MODEL_BYTES=16150800432                                 # sum of the four model-0000x-of-00004.safetensors (HF API)
ADDON=internvl25_expA_addon_20261114.tar.gz
ADDON_SHA=32031bdf30e153b369f6e85e32f03848747f1afd546386fe8457e40f2e7cffda
WAIT_FOR="${WAIT_FOR:-$W/QUEUE6_DONE}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"                        # model 16.2 GB x 1.3 + 8 GB for results and temp files
MAX_WAIT_H="${MAX_WAIT_H:-168}"                         # queue_6x6.sh step timeouts add up to about 154 h in the worst case (7 models)
POLL_S="${POLL_S:-60}"
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}"
export EXPECT_TORCH="${EXPECT_TORCH:-2.9.1+cu130}"      # only changed in local tests
T_SETUP=$((45*60)); T_MODEL=$((120*60)); T_SMOKE=$((40*60)); T_EXPA=$((3*3600))
# r2 (2026-11-15): only processes whose command line STARTS with bash/sh count; a tmux server started as
# "tmux new-session ... bash run_all_overnight.sh" keeps that text forever and blocked the queues on the server.
OTHERS="${OTHERS_RE:-run_all_overnight\.sh|queue_6x6\.sh|queue_minicpm\.sh|queue_sizes_uitars\.sh}"   # OTHERS_RE: local test hook

R="$W/mpr-ivl"
RES="$W/queue_results/internvl25_expA"
LOG="$RES/logs"
ST="$W/QUEUE7_STATUS"; DONE="$W/QUEUE7_DONE"
OUTD="$R/followup_rel_4lang/results/expA_internvl25_8b"
CACHE_DIR="$R/.cache/huggingface/hub/models--${MODEL_ID//\//--}"
DAY=$(date -u +%Y%m%d)
FAILED=()
FINISHED=0

PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1

say() { local l; l="$(date -u +%FT%TZ) ivl25_expA: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$ST"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<PLANEOF
Plan (nothing is executed):
 0. wait for $WAIT_FOR (any content; poll ${POLL_S}s, give up after ${MAX_WAIT_H}h), then while any of [$OTHERS] still runs,
    then until GPU memory used < 1.5 GB (max 30 min), then network check (curl ${NET_SECS}s ranged download from huggingface.co (the model's first
    shard) and download.pytorch.org (informational); STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1; MB/s go to $ST)
 1. checks: $ADDON present and sha256 = $ADDON_SHA; GPU is RTX 4090; free disk >= ${MIN_FREE_GB} GB; $R exists with .git and .venv
 2. setup (limit $((T_SETUP/60)) min): repo is a descendant of $BASE_COMMIT and carries the InternVL patches; unpack addon into $R and verify its
    SHA256SUMS; venv has torch 2.9.1+cu130, transformers 4.57.6, timm, einops, sentencepiece (else pip install with the repo's constraints file);
    config in the addon is byte-identical to 01_code/configs/models/internvl2_5_8b.yaml; Exp A inputs verify; data_bundle verify (fetch images if missing)
 3. model (limit $((T_MODEL/60)) min): hf download $MODEL_ID @ $REVISION (~16.2 GB, 3 tries, partial files resume); safetensors bytes = $MODEL_BYTES;
    config.json llm_config.architectures = InternLM2ForCausalLM; tokenizer loads and A, B, C, D are single tokens (ids 290, 309, 289, 302 as in the earlier run)
 4. smoke Exp A: 3 items = 108 inferences (limit $((T_SMOKE/60)) min) + gate (rules: all rows ok, parse failures <= 50%, labels not all identical, A-D mass >= 0.5,
    visual tokens 768..1792, english_fixed system prompt replaced the default Chinese one in every rendered prompt)
 5. Exp A full: 5,688 inferences (limit $((T_EXPA/3600)) h; not started if the smoke ETA exceeds it, IGNORE_ETA=1 overrides) + gate (all 5,688 rows ok)
 6. package to $RES/internvl25_expA_results_<date>.tar.gz; delete only $CACHE_DIR on success (kept after a failure); append to $ST;
    write $DONE (OK or FAILED steps: ...)
PLANEOF
  exit 0
fi

mkdir -p "$LOG"
command -v flock >/dev/null && { exec 9> "$W/.queue7_ivl25.lock"; flock -n 9 || { echo "another queue_internvl25_expA.sh is already running"; exit 1; }; }

retry() {  # retry <tries> <cmd...>: network steps on flaky hosts (SSL EOF / gnutls handshake errors seen on the previous server)
  local n=$1 i; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "retry $i/$n failed: $*" >&2
    [ "$i" = "$n" ] || sleep $(( ${RETRY_SLEEP:-30} * i ))
  done
  return 1
}

step() {  # step <name> <limit_s> <function>; runs the function in a subshell with a time limit
  local name=$1 limit=$2 fn=$3 rc
  say "step $name started (limit $((limit/60)) min)"
  timeout -k 60 "$limit" bash -c "set -Eeo pipefail; $fn" >> "$LOG/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then say "step $name ok"; return 0; fi
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then say "step $name TIMEOUT"
  elif [ $rc -eq 3 ] && [ "$name" = full_expa ]; then say "STOP step $name: smoke ETA longer than the step limit (see $OUTD/STATUS). IGNORE_ETA=1 overrides."; FAILED+=("${name}_eta"); return $rc
  else say "step $name FAILED (exit $rc), see $LOG/$name.log"; fi
  FAILED+=("$name"); return $rc
}

package() {
  local out="$RES/internvl25_expA_results_${DAY}.tar.gz" stage="$RES/_stage"
  rm -rf "$stage"; mkdir -p "$stage"
  if [ -d "$R" ]; then
    cp -r "$OUTD" "$stage/" 2>/dev/null || true
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git log --oneline -7 > "$stage/git_log.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
  fi
  cp -r "$LOG" "$stage/queue_logs" 2>/dev/null || true
  cp "$RES/STATUS" "$stage/" 2>/dev/null || true
  nvidia-smi > "$stage/nvidia_smi.txt" 2>&1 || true
  df -h "$W" > "$stage/disk_end.txt" 2>&1 || true
  tar -czf "$out" -C "$stage" . && sha256sum "$out" > "$out.sha256" && say "packaged $out ($(du -h "$out" | cut -f1))"
  rm -rf "$stage"
}

finish() {  # runs once, also on error exits
  [ "$FINISHED" = 1 ] && return 0
  FINISHED=1
  package || say "packaging FAILED"
  local verdict="OK"; [ ${#FAILED[@]} -gt 0 ] && verdict="FAILED steps: ${FAILED[*]}"
  if [ -d "$CACHE_DIR" ]; then
    if [ "${KEEP_MODEL_CACHE:-0}" = 1 ]; then say "model cache kept (KEEP_MODEL_CACHE=1)"
    elif [ "$verdict" = OK ]; then rm -rf "$CACHE_DIR"; say "model cache deleted (run finished); free disk: $(df -h --output=avail "$W" | tail -1 | tr -d ' ')"
    else say "model cache KEPT (not finished). A rerun resumes without downloading."; fi
  fi
  say "FINISHED $verdict. Results: $RES/internvl25_expA_results_${DAY}.tar.gz"
  echo "$verdict" > "$DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no QUEUE7_DONE written, rerun the script to resume"; FINISHED=1; exit 130' INT TERM

abort() { local key=$1; shift; say "STOP: $*"; FAILED+=("$key"); exit 1; }   # abort <key> <message>; key = name in "FAILED steps:"

# ---- 0. wait for the earlier queues ------------------------------------------------------------
rm -f "$DONE"
say "queue_internvl25_expA started; waiting for $WAIT_FOR"
waited=0
while [ ! -e "$WAIT_FOR" ]; do
  sleep "$POLL_S"; waited=$((waited + POLL_S))
  if [ $((waited % 1800)) -lt "$POLL_S" ]; then
    pgrep -f '^(/bin/)?(ba)?sh .*queue_6x6\.sh' >/dev/null 2>&1 && q6=running || q6="NOT running (stopped early? check)"
    say "still waiting for $(basename "$WAIT_FOR") (${waited}s); queue_6x6.sh $q6; last QUEUE6_STATUS: $(tail -1 "$W/QUEUE6_STATUS" 2>/dev/null | cut -c1-160)"
  fi
  # r1: queue_6x6.sh exits without QUEUE6_DONE when it STOPs early. If it is not running and its last status line is a STOP or
  # an interrupt, count it; after 2 such checks in a row (>= 2 polls apart) treat it as finished. The OTHERS/GPU-idle checks below still apply.
  if ! pgrep -f '^(/bin/)?(ba)?sh .*queue_6x6\.sh' >/dev/null 2>&1 && tail -1 "$W/QUEUE6_STATUS" 2>/dev/null | grep -q -E 'STOP|interrupted|FINISHED'; then
    q6stop=$(( ${q6stop:-0} + 1 ))
    if [ "$q6stop" -ge 2 ]; then say "queue_6x6.sh is not running and ended with: $(tail -1 "$W/QUEUE6_STATUS" | cut -c1-160); continuing without $(basename "$WAIT_FOR")"; break; fi
  else q6stop=0; fi
  [ "$waited" -ge "${MAX_WAIT_S:-$((MAX_WAIT_H*3600))}" ] && abort wait "$(basename "$WAIT_FOR") did not appear within ${MAX_WAIT_H}h"   # MAX_WAIT_S: local test hook
done
say "$(basename "$WAIT_FOR") found: $(head -c 200 "$WAIT_FOR" | tr '\n' ' ')"
pgrep -f "^(/bin/)?(ba)?sh .*($OTHERS)" >/dev/null 2>&1 && say "another queue/overnight process is still running ($(pgrep -fl "^(/bin/)?(ba)?sh .*($OTHERS)" | head -2 | cut -c1-80 | tr '\n' ';')); waiting for it to end"
while pgrep -f "^(/bin/)?(ba)?sh .*($OTHERS)" >/dev/null 2>&1; do
  sleep "$POLL_S"; waited=$((waited + POLL_S))
  [ "$waited" -ge $((MAX_WAIT_H*3600)) ] && abort others "another queue/overnight process still running after ${MAX_WAIT_H}h"
done
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
  [ -z "$used" ] && abort gpu "nvidia-smi not available"
  [ "$used" -lt 1500 ] && break
  sleep "${GPU_POLL_S:-60}"
done
[ "${used:-99999}" -lt 1500 ] || abort gpu "GPU still busy (${used} MiB used)"

# ---- 0b. network check (same method as queue_6x6.sh): 15 s ranged download -------------------------
net_speed() {  # $1 url -> prints "MB/s" or "ERR:<http code>"
  local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'
}
net_check() {  # download.pytorch.org is measured and logged only; torch is not downloaded unless the venv is broken
  local hf pt slow=0
  hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/$MODEL_ID/resolve/$REVISION/model-00001-of-00004.safetensors}")
  pt=$(net_speed "${NET_PT_URL:-https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl}")
  say "network check (${NET_SECS}s range request): huggingface.co ${hf} MB/s, download.pytorch.org ${pt} MB/s; minimum $NET_MIN_MBPS MB/s (huggingface.co decides)"
  printf 'huggingface.co %s MB/s\ndownload.pytorch.org %s MB/s\n' "$hf" "$pt" > "$RES/net_check.txt"
  case "$hf" in
    ERR:4*) say "WARNING: the huggingface test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;;
    ERR*) slow=1 ;;
    *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;;
  esac
  if [ "$slow" = 1 ]; then
    if [ "${FORCE_SLOW_NET:-0}" = 1 ]; then say "network below $NET_MIN_MBPS MB/s but FORCE_SLOW_NET=1: continuing"; return 0; fi
    abort network "network too slow or unreachable (need >= $NET_MIN_MBPS MB/s from huggingface.co). Nothing was downloaded. Use another instance, or FORCE_SLOW_NET=1"
  fi
}
net_check

# ---- 1. checks --------------------------------------------------------------------------------
[ -s "$W/$ADDON" ] || abort addon "$ADDON missing in $W"
echo "$ADDON_SHA  $W/$ADDON" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort addon "$ADDON sha256 differs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$LOG/gpu.log" 2>&1
grep -q 4090 "$LOG/gpu.log" || abort gpu "GPU is not an RTX 4090 ($(cat "$LOG/gpu.log"))"
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
df -h "$W" > "$LOG/disk_start.log"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || [ -s "$CACHE_DIR/snapshots/$REVISION/config.json" ] || abort disk "only ${FREE_GB} GB free in $W (need ${MIN_FREE_GB}+)"
{ [ -d "$R/.git" ] && [ -f "$R/.venv/bin/activate" ]; } || abort repo "$R (repo with .venv) does not exist; it is created by the overnight job / queue_6x6.sh (InternVL3.5). Not rebuilt here"
say "checks ok: $(cat "$LOG/gpu.log"); ${FREE_GB} GB free"

# ---- 2. setup -------------------------------------------------------------------------------
setup() {
  cd "$R"
  git cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null
  git merge-base --is-ancestor "$BASE_COMMIT" HEAD
  echo "HEAD $(git rev-parse --short HEAD), $(git rev-list --count "$BASE_COMMIT"..HEAD) commits after ${BASE_COMMIT:0:7}"
  grep -q 'INTERNVL_FAMILIES' 01_code/src/mpr_crosslocale/inference/run_cross_locale.py          # patch 0001: internvl2_5 and 3.5 share the adapter path
  grep -q 'self.model.system_message if system_prompt is None' 01_code/src/mpr_crosslocale/models/internvl25.py   # patch 0001: an explicit system prompt REPLACES the default
  tar -xzf "$W/$ADDON"
  ( cd followup_rel_4lang/expA_ivl25 && sha256sum -c SHA256SUMS )
  cmp followup_rel_4lang/expA_ivl25/internvl2_5_8b.yaml 01_code/configs/models/internvl2_5_8b.yaml
  [ -s followup_rel_4lang/expA/expA_inputs/EXPA_SHA256SUMS ]
  ( cd followup_rel_4lang/expA && sha256sum -c expA_inputs/EXPA_SHA256SUMS )
  export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" PIP_CACHE_DIR="$R/.cache/pip" PIP_NO_CACHE_DIR=1
  source .venv/bin/activate
  venv_ok() { REPO_DIR="$PWD" python - <<'PY' >/dev/null 2>&1
import os, torch, transformers, timm, einops, sentencepiece, mpr_crosslocale
assert torch.__version__ == os.environ["EXPECT_TORCH"], torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__
assert os.path.realpath(mpr_crosslocale.__file__).startswith(os.path.realpath(os.environ["REPO_DIR"]))
PY
  }
  if ! venv_ok; then
    echo "venv check failed; installing with the repo's pins"
    C=followup_rel_4lang/rstem_ropts_inputs/internvl35_constraints_20261109.txt
    retry 3 python -m pip install -r 01_code/requirements-qwen3-4090.txt -c "$C"
    retry 3 python -m pip install -e './01_code[models]' -c "$C"
    venv_ok
  fi
  python -c 'import torch,transformers,timm,sentencepiece;print("torch",torch.__version__,"transformers",transformers.__version__,"timm",timm.__version__,"sentencepiece",sentencepiece.__version__)'
  python followup_rel_4lang/data_bundle.py verify || { python followup_rel_4lang/data_bundle.py unpack; retry 6 python followup_rel_4lang/data_bundle.py fetch-images --workers 2; python followup_rel_4lang/data_bundle.py verify; }
}

model() {
  cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
  source .venv/bin/activate
  # 3 tries; partial files stay in the cache and hf resumes them (also after a timeout of this step)
  retry 3 hf download "$MODEL_ID" --revision "$REVISION"
  MODEL_ID="$MODEL_ID" REVISION="$REVISION" MODEL_BYTES="$MODEL_BYTES" RES="$RES" python - <<'PY'
import glob, json, os
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MODEL_ID"], revision=os.environ["REVISION"], local_files_only=True)
fs = glob.glob(os.path.join(snap, "model-*.safetensors"))
size = sum(os.path.getsize(p) for p in fs)
print("snapshot", snap, "safetensors files", len(fs), "bytes", size)
assert len(fs) == 4 and size == int(os.environ["MODEL_BYTES"]), size
cfg = json.load(open(os.path.join(snap, "config.json")))
arch = cfg["llm_config"]["architectures"]
print("llm_config.architectures", arch, "| _name_or_path", cfg["llm_config"].get("_name_or_path"), "| template", cfg.get("template"))
assert arch == ["InternLM2ForCausalLM"] and cfg.get("template") == "internvl2_5", (arch, cfg.get("template"))
open(os.path.join(os.environ["RES"], "model_size.txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\nllm={arch} {cfg['llm_config'].get('_name_or_path')}\n")
# fail fast on a tokenizer / transformers mismatch before the GPU is used (same loader as the adapter)
from mpr_crosslocale.models.internvl25 import load_internvl_tokenizer
tok = load_internvl_tokenizer(os.environ["MODEL_ID"], revision=os.environ["REVISION"], trust_remote_code=True)
ids = {l: list(tok.encode(l, add_special_tokens=False)) for l in "ABCD"}
print("label ids", ids)
assert all(len(v) == 1 for v in ids.values()), ids
assert {k: v[0] for k, v in ids.items()} == {"A": 290, "B": 309, "C": 289, "D": 302}, ids   # ids recorded in the 2026-10-05 REL run
PY
  du -sh "$HF_HOME"
}

# r3 (2026-11-15): Vast sets HF_HOME=/workspace/.hf_home in /etc/environment; run_expA_ivl25.sh keeps an inherited HF_HOME,
# so smoke looked there and failed (model is in $R/.cache/huggingface). Export the same cache vars as the model step.
# (exports are inline: steps run in a fresh bash that only gets the step function itself)
smoke_expa() { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; bash followup_rel_4lang/expA_ivl25/run_expA_ivl25.sh smoke; }
full_expa()  { cd "$R"; export XDG_CACHE_HOME="$R/.cache" HF_HOME="$R/.cache/huggingface" TORCH_HOME="$R/.cache/torch" CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1; EXPA_LIMIT_S="$T_EXPA" bash followup_rel_4lang/expA_ivl25/run_expA_ivl25.sh full; }
export -f retry setup model smoke_expa full_expa
export W R RES BASE_COMMIT ADDON MODEL_ID REVISION MODEL_BYTES T_EXPA

smoke_report() {  # accuracy, speed, ETA and warnings from the smoke gate
  python3 - "$OUTD/expa_smoke/gate_expa_smoke.json" <<'PY' 2>/dev/null
import json, sys
try:
    g = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
s, t = g.get("sanity", {}), g.get("timing", {})
line = f"expa: acc {s.get('accuracy')} median {t.get('median_s_per_inference')} s/inf ETA full {t.get('eta_full_hours')} h visual tokens {s.get('visual_token_count')}"
if g.get("warnings"):
    line += " WARN " + "; ".join(g["warnings"])
if g.get("problems"):
    line += " PROBLEMS " + "; ".join(g["problems"])
print(line, end=" | ")
PY
}

if step setup "$T_SETUP" setup && step model "$T_MODEL" model; then
  if step smoke_expa "$T_SMOKE" smoke_expa; then
    say "smoke gate: $(smoke_report)"
    step full_expa "$T_EXPA" full_expa || true
  else
    say "smoke failed: $(smoke_report)see $LOG/smoke_expa.log"
  fi
fi
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
