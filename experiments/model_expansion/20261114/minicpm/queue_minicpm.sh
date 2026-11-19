#!/usr/bin/env bash
# MiniCPM-V-4.5 on the Vast.ai RTX 4090, run as the LAST item of the queue:
#   wait for /workspace/QUEUE_DONE (any content: "OK" or "STOPPED rc=..") -> GPU idle -> network check
#   -> checks -> setup /workspace/mpr-minicpm (own venv) -> pinned model
#   -> smoke (REL, Exp A) -> REL O/R/C/F (18,432) -> Exp A O/T/R (5,688) -> package -> delete model cache
#   -> /workspace/MINICPM_DONE
# Usage (inside tmux, in /workspace, with minicpm_addon_20261114.tar.gz next to this script):
#   bash queue_minicpm.sh            # waits for QUEUE_DONE first
#   bash queue_minicpm.sh --plan     # only print what would run; changes nothing
# Rerunning resumes finished work (repo, venv, data, predictions are reused).
# Network steps (git clone, pip install, image fetch, hf download x3) are retried. FORCE_SLOW_NET=1: continue although the network is below 20 MB/s.
# Cache rule at the end: a successful run deletes the model cache. After a failure or timeout the model cache is KEPT (a rerun resumes
# without downloading) while the free disk is at least NEXT_NEEDS_GB (default 52, the UI-TARS download of the 6x6 queue); otherwise it is
# deleted. pip/torch caches are always deleted. KEEP_MODEL_CACHE=1 never deletes the model cache.
# Progress: tail /workspace/QUEUE_STATUS ; per-step logs: /workspace/queue_results/minicpm/logs/
# The instance is NOT stopped or destroyed. Existing /workspace/mpr-gui-crosslocale and /workspace/mpr-ivl are not touched.
set -uo pipefail

W="${W:-/workspace}"
REPO_URL=https://github.com/dltmdwoOS/mpr-gui-crosslocale.git
BRANCH=followup-rel-4lang-inference
COMMIT=666440011f538cd9336794c9965429de640843c3
MODEL_ID=openbmb/MiniCPM-V-4_5
REVISION=14a2004908140699317e2109e90573c1e6323c36
MODEL_BYTES=17391890184            # r1: sum of the four safetensors FILE sizes (HF API tree @ revision); index total_size (17391790560) omits headers
ADDON=minicpm_addon_20261115.tar.gz   # r2: EXPA_SHA256SUMS + ADDON_SHA256SUMS updated for the generic check_T_gate.py and run_expA_generic.sh (20261114 failed its own sha check on the server)
ADDON_SHA=e638bb79e9795f15ef3a2759f2efe12c1a6a313a40018e4405fe8221620fb150
MIN_FREE_GB="${MIN_FREE_GB:-40}"
# Give up waiting for QUEUE_DONE after this many hours. The sizes queue now always writes QUEUE_DONE when it ends (OK or STOPPED), so this is
# only a backstop: overnight job up to about 12 h plus the sizes queue 19-26 h (sum of its step timeouts is about 48 h).
MAX_WAIT_H="${MAX_WAIT_H:-48}"
POLL_S="${POLL_S:-60}"
# per-step time limits (seconds); on timeout the step counts as failed
NET_MIN_MBPS="${NET_MIN_MBPS:-20}"; NET_SECS="${NET_SECS:-15}"; NEXT_NEEDS_GB="${NEXT_NEEDS_GB:-52}"
export RETRY_SLEEP="${RETRY_SLEEP:-30}"
T_SETUP=$((90*60)); T_MODEL=$((120*60)); T_SMOKE=$((40*60)); T_REL=$((6*3600)); T_EXPA=$((3*3600))

R="$W/mpr-minicpm"
RES="$W/queue_results/minicpm"
LOG="$RES/logs"
DAY=$(date -u +%Y%m%d)
FAILED=()
FINISHED=0

PLAN_ONLY=0; [ "${1:-}" = "--plan" ] && PLAN_ONLY=1

say() { local l; l="$(date -u +%FT%TZ) minicpm: $*"; echo "$l"; [ "$PLAN_ONLY" = 1 ] && return 0; echo "$l" >> "$W/QUEUE_STATUS"; echo "$l" >> "$RES/STATUS"; }

if [ "$PLAN_ONLY" = 1 ]; then
  cat <<EOF
Plan (nothing is executed):
 0. wait for $W/QUEUE_DONE (any content; poll ${POLL_S}s, give up after ${MAX_WAIT_H}h; keeps waiting while run_all_overnight.sh runs),
    then wait until GPU memory used < 1.5 GB (max 30 min), then network check (curl ${NET_SECS}s ranged download from huggingface.co and
    download.pytorch.org; STOP below ${NET_MIN_MBPS} MB/s unless FORCE_SLOW_NET=1; MB/s go to QUEUE_STATUS)
 1. checks: $ADDON present and sha256 = $ADDON_SHA; python 3.12; GPU is RTX 4090; free disk >= ${MIN_FREE_GB} GB
 2. setup (limit $((T_SETUP/60)) min): clone $REPO_URL ($BRANCH) to $R at $COMMIT; unpack addon; git am adapter patch;
    venv + pip (same pins as requirements-qwen3-4090.txt: torch 2.9.1+cu130, transformers 4.57.6); data_bundle unpack / fetch-images / verify
 3. model (limit $((T_MODEL/60)) min): hf download $MODEL_ID @ $REVISION (~17.4 GB, 3 tries, partial files resume), size check, tokenizer/processor load check
 4. smoke REL 80 inferences, smoke Exp A 108 inferences (each limit $((T_SMOKE/60)) min); a failed gate (incl. the sanity rules) stops before the full runs;
    each full step is skipped if the smoke timing predicts it cannot finish within its limit (IGNORE_ETA=1 overrides)
 5. REL O/R/C/F full 18,432 inferences (limit $((T_REL/3600)) h); Exp A O/T/R full 5,688 inferences (limit $((T_EXPA/3600)) h)
 6. package to $RES/minicpm_results_<date>.tar.gz; model cache: delete on success, keep after a failure while free disk >= ${NEXT_NEEDS_GB} GB;
    pip/torch caches deleted; append to $W/QUEUE_STATUS; write $W/MINICPM_DONE (OK or FAILED steps: ...)
EOF
  exit 0
fi

mkdir -p "$LOG"
exec 9> "$W/.minicpm_queue.lock"
flock -n 9 || { echo "another queue_minicpm.sh is already running"; exit 1; }

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
  [ $rc -eq 124 ] && say "step $name TIMEOUT" || say "step $name FAILED (exit $rc), see $LOG/$name.log"
  FAILED+=("$name"); return $rc
}

package() {
  local out="$RES/minicpm_results_${DAY}.tar.gz" stage="$RES/_stage"
  rm -rf "$stage"; mkdir -p "$stage"
  if [ -d "$R" ]; then
    cp -r "$R/followup_rel_4lang/results/minicpm_rel" "$stage/" 2>/dev/null || true
    cp -r "$R/followup_rel_4lang/results/expA_minicpm" "$stage/" 2>/dev/null || true
    (cd "$R" && git rev-parse HEAD > "$stage/git_commit.txt"; git log --oneline -3 > "$stage/git_log.txt"; git status --short > "$stage/git_status.txt") 2>/dev/null || true
    [ -x "$R/.venv/bin/python" ] && "$R/.venv/bin/python" -m pip freeze > "$stage/requirements.freeze.txt" 2>/dev/null || true
    cp "$R/01_code/configs/models/minicpm_v4_5_8b_4090.yaml" "$stage/" 2>/dev/null || true
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
  if [ -d "$R/.cache" ]; then
    rm -rf "$R/.cache/pip" "$R/.cache/torch"
    local f; f=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
    if [ "${KEEP_MODEL_CACHE:-0}" = 1 ]; then say "model cache kept (KEEP_MODEL_CACHE=1)"
    elif [ "$verdict" = OK ]; then rm -rf "$R/.cache/huggingface"; say "model cache deleted (run finished); free disk: $(df -h --output=avail "$W" | tail -1 | tr -d ' ')"
    elif [ "$f" -ge "$NEXT_NEEDS_GB" ]; then say "model cache KEPT (not finished; free ${f} GB >= ${NEXT_NEEDS_GB} GB). A rerun resumes without downloading."
    else rm -rf "$R/.cache/huggingface"; say "model cache deleted (not finished; free ${f} GB < ${NEXT_NEEDS_GB} GB needed by the next queue)"; fi
  fi
  say "FINISHED $verdict. Results: $RES/minicpm_results_${DAY}.tar.gz"
  echo "$verdict" > "$W/MINICPM_DONE"
}
trap 'finish' EXIT
trap 'say "interrupted by signal; no MINICPM_DONE written, rerun the script to resume"; FINISHED=1; exit 130' INT TERM

abort() { say "STOP: $*"; FAILED+=("$1"); exit 1; }

# ---- 0. wait for the earlier queue ------------------------------------------------------------
say "queue_minicpm started; waiting for $W/QUEUE_DONE"
waited=0
while [ ! -e "$W/QUEUE_DONE" ]; do
  sleep "$POLL_S"; waited=$((waited + POLL_S))
  [ $((waited % 1800)) -lt "$POLL_S" ] && say "still waiting for QUEUE_DONE (${waited}s)"
  [ "$waited" -ge "${MAX_WAIT_S:-$((MAX_WAIT_H*3600))}" ] && abort "QUEUE_DONE did not appear within ${MAX_WAIT_H}h"   # MAX_WAIT_S: local test hook
done
say "QUEUE_DONE found: $(head -c 200 "$W/QUEUE_DONE" | tr '\n' ' ')"
# A queue that STOPped early writes QUEUE_DONE although the overnight job may still be on the GPU: wait for that job too.
pgrep -f 'run_all_overnight\.sh' >/dev/null 2>&1 && say "run_all_overnight.sh is still running; waiting for it to end"
while pgrep -f 'run_all_overnight\.sh' >/dev/null 2>&1; do
  sleep "$POLL_S"; waited=$((waited + POLL_S))
  [ "$waited" -ge $((MAX_WAIT_H*3600)) ] && abort "run_all_overnight.sh still running after ${MAX_WAIT_H}h"
done
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
  [ -z "$used" ] && abort "nvidia-smi not available"
  [ "$used" -lt 1500 ] && break
  sleep 60
done
[ "${used:-99999}" -lt 1500 ] || abort "GPU still busy after QUEUE_DONE (${used} MiB used)"

# ---- 0b. network check (same method as queue_6x6.sh): 15 s ranged download -------------------------
net_speed() {  # $1 url -> prints "MB/s" or "ERR:<http code>"
  local o sz t code
  o=$(curl -sL -r 0-419430399 --max-time "$NET_SECS" -o /dev/null -w '%{size_download} %{time_total} %{http_code}' "$1" 2>/dev/null) || true
  sz=${o%% *}; code=${o##* }; t=$(echo "$o" | awk '{print $2}')
  case "$code" in 200|206) ;; *) echo "ERR:${code:-none}"; return 0 ;; esac
  awk -v s="${sz:-0}" -v t="${t:-0}" 'BEGIN{ if (t < 0.5) t = 0.5; printf "%.1f", s / t / 1e6 }'
}
net_check() {  # download.pytorch.org only decides when torch still has to be installed (no venv with torch yet)
  local hf pt slow=0 need_pt=0
  grep -q "2.9.1+cu130" "$R"/.venv/lib/python3*/site-packages/torch/version.py 2>/dev/null || need_pt=1
  hf=$(net_speed "${NET_HF_URL:-https://huggingface.co/$MODEL_ID/resolve/$REVISION/model-00001-of-00004.safetensors}")
  pt=$(net_speed "${NET_PT_URL:-https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl}")
  say "network check (${NET_SECS}s range request): huggingface.co ${hf} MB/s, download.pytorch.org ${pt} MB/s (torch download needed: $([ $need_pt = 1 ] && echo yes || echo no)); minimum $NET_MIN_MBPS MB/s"
  printf 'huggingface.co %s MB/s\ndownload.pytorch.org %s MB/s\n' "$hf" "$pt" > "$RES/net_check.txt"
  case "$hf" in
    ERR:4*) say "WARNING: the huggingface test URL answered HTTP ${hf#ERR:}; speed unknown, continuing" ;;
    ERR*) slow=1 ;;
    *) awk -v v="$hf" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;;
  esac
  if [ "$need_pt" = 1 ]; then
    case "$pt" in
      ERR:4*) say "WARNING: the pytorch test URL answered HTTP ${pt#ERR:}; speed unknown, continuing" ;;
      ERR*) slow=1 ;;
      *) awk -v v="$pt" -v m="$NET_MIN_MBPS" 'BEGIN{exit !(v < m)}' && slow=1 ;;
    esac
  fi
  if [ "$slow" = 1 ]; then
    if [ "${FORCE_SLOW_NET:-0}" = 1 ]; then say "network below $NET_MIN_MBPS MB/s but FORCE_SLOW_NET=1: continuing"; return 0; fi
    abort "network too slow or unreachable (need >= $NET_MIN_MBPS MB/s). Nothing was downloaded. Use another instance, or FORCE_SLOW_NET=1"
  fi
}
net_check

# ---- 1. checks --------------------------------------------------------------------------------
[ -s "$W/$ADDON" ] || abort "$ADDON missing in $W"
echo "$ADDON_SHA  $W/$ADDON" | sha256sum -c - >> "$LOG/addon_sha.log" 2>&1 || abort "$ADDON sha256 differs"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 2>> "$LOG/python.log" || abort "python3 is not 3.12 ($(python3 --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$LOG/gpu.log" 2>&1
grep -q 4090 "$LOG/gpu.log" || abort "GPU is not an RTX 4090 ($(cat "$LOG/gpu.log"))"
FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
df -h "$W" > "$LOG/disk_start.log"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || abort "only ${FREE_GB} GB free in $W (need ${MIN_FREE_GB}+)"
say "checks ok: $(cat "$LOG/gpu.log"); ${FREE_GB} GB free"

# ---- 2. setup -------------------------------------------------------------------------------
setup() {
  cd "$W"
  clone_repo() { rm -rf mpr-minicpm; git clone --branch "$BRANCH" "$REPO_URL" mpr-minicpm; }
  [ -d mpr-minicpm/.git ] || retry 5 clone_repo
  cd mpr-minicpm
  if [ ! -f .git/minicpm_patched ]; then
    git am --abort 2>/dev/null || true
    git checkout -f "$COMMIT"
    tar -xzf "$W/$ADDON"
    sha256sum -c followup_rel_4lang/minicpm/ADDON_SHA256SUMS
    git -c user.name=local -c user.email=local@local am followup_rel_4lang/minicpm_patches/0001-*.patch
    touch .git/minicpm_patched
  else
    tar -xzf "$W/$ADDON"
    sha256sum -c followup_rel_4lang/minicpm/ADDON_SHA256SUMS
  fi
  [ "$(git rev-list --count "$COMMIT"..HEAD)" = 1 ]
  git log --oneline -3
  source followup_rel_4lang/minicpm/minicpm_env.sh
  export PIP_NO_CACHE_DIR=1
  if [ ! -f .venv/bin/activate ]; then
    rm -rf .venv
    python3 -m venv .venv || { apt-get update && apt-get install -y python3.12-venv && python3 -m venv .venv; }
  fi
  source .venv/bin/activate
  if [ ! -f .venv/.minicpm_installed ]; then
    retry 3 python -m pip install -r 01_code/requirements-qwen3-4090.txt
    retry 3 python -m pip install -e './01_code[models]'
    python -m pip check
    touch .venv/.minicpm_installed
  fi
  v=$(python -c 'import torch,transformers;print(torch.__version__,transformers.__version__)'); echo "torch/transformers: $v"
  [ "$v" = "2.9.1+cu130 4.57.6" ]
  python followup_rel_4lang/data_bundle.py unpack
  retry 6 python followup_rel_4lang/data_bundle.py fetch-images --workers 2
  python followup_rel_4lang/data_bundle.py verify
}

model() {
  cd "$R"; source followup_rel_4lang/minicpm/minicpm_env.sh; source .venv/bin/activate
  # 3 tries; partial files stay in the cache and hf resumes them (also after a timeout of this step)
  retry 3 hf download "$MINICPM_MODEL_ID" --revision "$MINICPM_REVISION"
  python - <<'PY'
import glob, os
from huggingface_hub import snapshot_download
snap = snapshot_download(os.environ["MINICPM_MODEL_ID"], revision=os.environ["MINICPM_REVISION"], local_files_only=True)
size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(snap, "model-*.safetensors")))
print("snapshot", snap, "safetensors bytes", size)
assert len(glob.glob(os.path.join(snap, "model-*.safetensors"))) == 4 and size == 17391890184, size  # r1 file sizes incl. headers
open(os.path.join(os.environ["R"], "model_size.txt"), "w").write(f"{snap}\nsafetensors_bytes={size}\nparams=8695895280 (bf16)\n")
# fail fast on a transformers / remote-code mismatch before the GPU is used
from transformers import AutoProcessor, AutoTokenizer
tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
AutoProcessor.from_pretrained(snap, trust_remote_code=True)
ids = {l: tok.encode(l, add_special_tokens=False) for l in "ABCD"}
print("label ids", ids)
assert all(len(v) == 1 for v in ids.values())
PY
  du -sh "$HF_HOME"
  cp "$R/model_size.txt" "$RES/" 2>/dev/null || true
}

smoke_rel()  { cd "$R"; bash followup_rel_4lang/minicpm/run_rel_minicpm.sh smoke; }
smoke_expa() { cd "$R"; bash followup_rel_4lang/expA/run_expA_generic.sh minicpm smoke; }
full_rel()   { cd "$R"; bash followup_rel_4lang/minicpm/run_rel_minicpm.sh full; }
full_expa()  { cd "$R"; bash followup_rel_4lang/expA/run_expA_generic.sh minicpm full; }
export -f retry setup model smoke_rel smoke_expa full_rel full_expa
export W R RES REPO_URL BRANCH COMMIT ADDON

smoke_report() {  # accuracy, speed, ETA and warnings from the two smoke gates
  python3 - "$R/followup_rel_4lang/results/minicpm_rel/smoke/gate_smoke.json" "$R/followup_rel_4lang/results/expA_minicpm/smoke/gate_smoke.json" <<'PY' 2>/dev/null
import json, sys
for name, p in zip(("rel", "expa"), sys.argv[1:]):
    try:
        g = json.load(open(p))
    except Exception:
        continue
    s = g.get("sanity", {})
    eta = g.get("eta_full_seconds")
    line = f"{name}: acc {s.get('accuracy')} ETA full {round(eta / 3600, 2) if eta else None} h"
    if g.get("warnings"):
        line += " WARN " + "; ".join(g["warnings"])
    notes = [f"{k}: {v}" for k, v in s.items() if isinstance(v, str) and ("not evaluated" in v or "not recorded" in v)]
    if notes:
        line += " NOT CHECKED (field missing) " + "; ".join(notes)
    print(line, end=" | ")
PY
}

eta_guard() {  # eta_guard <name> <smoke gate json> <limit_s> <full predictions file> <total>; returns 3 if the step cannot finish in its limit
  [ "${IGNORE_ETA:-0}" = 1 ] && return 0
  local msg rc=0
  msg=$(python3 - "$2" "$3" "$4" "$5" <<'PY' 2>&1
import json, sys
gate, limit, preds, total = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
try:
    eta = json.load(open(gate))["eta_full_seconds"]
except Exception:
    eta = None
if eta is None:
    print("no ETA in the smoke gate; step starts without ETA check"); sys.exit(0)
done = 0
try:
    done = sum(1 for line in open(preds) if '"status": "success"' in line)
except OSError:
    pass
remaining = eta * max(0, total - done) / total
print(f"ETA {remaining / 3600:.1f} h (smoke median x {total - done} remaining inferences) vs step limit {limit / 3600:.1f} h")
sys.exit(3 if remaining > limit else 0)
PY
  ) || rc=$?
  if [ "$rc" = 3 ]; then say "STOP step $1: $msg. The step is not started. IGNORE_ETA=1 overrides."; FAILED+=("$1_eta"); return 3; fi
  say "step $1 ETA check: $msg"; return 0
}

RR="$R/followup_rel_4lang/results"
if step setup "$T_SETUP" setup && step model "$T_MODEL" model; then
  if step smoke_rel "$T_SMOKE" smoke_rel && step smoke_expa "$T_SMOKE" smoke_expa; then
    say "smoke gates: $(smoke_report)"
    eta_guard full_rel "$RR/minicpm_rel/smoke/gate_smoke.json" "$T_REL" "$RR/minicpm_rel/full/predictions.jsonl" 18432 && { step full_rel "$T_REL" full_rel || true; }
    eta_guard full_expa "$RR/expA_minicpm/smoke/gate_smoke.json" "$T_EXPA" "$RR/expA_minicpm/full/predictions.jsonl" 5688 && { step full_expa "$T_EXPA" full_expa || true; }
  else
    say "smoke failed: $(smoke_report)see $LOG/smoke_rel.log and $LOG/smoke_expa.log"
  fi
fi
exit $([ ${#FAILED[@]} -eq 0 ] && echo 0 || echo 1)
