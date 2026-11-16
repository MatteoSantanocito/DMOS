#!/usr/bin/env bash
# =============================================================================
# setup-kubephos-fleet.sh — DMOS on a multicluster built by Kubephos, N clusters
# =============================================================================
# From the kubeconfig Kubephos gives (one context per cluster) to a fleet ready
# for DMOS, checked, calibrated, with an optional first test (DMOS off vs on).
#
#   bash scripts/setup-kubephos-fleet.sh --kubeconfig ~/Downloads/kubeconfig.yaml \
#        --name myfleet [--regions DE,FR,PL,US-VA,SG,NO] [--no-runs]
#
# Steps (each prints what it does; the script stops at the first failure):
#   1  discover     one kubeconfig per cluster in --out-dir, app-pool CPU/RAM,
#                   Prometheus, GatewayClass, Cilium, ClusterMesh (read only)
#   2  hubble       adds the Hubble httpV2 labels DMOS needs where missing
#                   (helm upgrade of the installed Cilium chart + agent restart;
#                   --no-hubble-fix: stop instead)
#   3  workload     scripts/setup-dmos-on-clusters.ps1 -IngressMode gateway:
#                   Online Boutique on the app pool, L7 policies, global
#                   services, Hubble ServiceMonitor, Gateway + HTTPRoute
#   4  local.yaml   config/local.yaml for the fleet (previous one saved in
#                   --out-dir) + scripts/render-runtime-files.sh
#   5  prepare      apply_resource_hardening.sh, apply_pod_lifecycle.sh,
#                   clean_reset.sh
#   6  check        every cluster: Hubble labels, Deployments, pods on the app
#                   pool, Gateway 200, Hubble scraped once, ClusterMesh; probe:
#                   traffic to the first Gateway, demand measured client-side,
#                   frontends of the other clusters serving part of it
#   7  calibrate    k = app-node CPU per request, at a steady load with fixed
#                   replicas (~8 min; only when --out-dir has no
#                   calibration.json, or with --calibrate); then
#                   config/local.yaml again with the replica cap of every
#                   cluster (docs/KUBEPHOS.md, "Replica cap")
#   8  runs         flash crowd with DMOS off, reset, DMOS (nca) on, reset,
#                   comparison table (about 40 min; --no-runs skips it)
#
# Options:
#   --kubeconfig PATH   kubeconfig from Kubephos (required)
#   --name NAME         fleet name (default: the kubeconfig file name)
#   --out-dir DIR       kubeconfigs, fleet.json, calibration.json,
#                       local.<name>.yaml, backups, run logs
#                       (default: ~/Desktop/dmos-iac/<name>)
#   --contexts a,b,c    subset/order of the contexts (default: all, kubeconfig
#                       order = order of the k6 weights; the first is the hot spot)
#   --regions R1,R2,..  region of each cluster (default DE,FR,PL,US-VA,SG,NO
#                       truncated to N; required beyond 6 clusters)
#   --baseline R=G      carbon baseline gCO2/kWh of a region not in the
#                       reference list (repeatable)
#   --peak-rps N        peak of the test runs (default of run_flash_crowd.sh: 250)
#   --from STEP         start from step 1..8 (e.g. --from 6 to re-check)
#   --no-hubble-fix     do not touch Cilium: stop if the labels are missing
#   --calibrate         measure k again even if calibration.json exists
#   --no-calibrate      skip the measurement (caps from an existing
#                       calibration.json, otherwise from services.yaml)
#   --k MCORE           mcore per request instead of the measured one
#   --u-cpu U           target CPU utilisation of the app pool (default 0.8)
#   --no-runs           stop before the runs
#
# Run it from Git Bash in the repository root. Kubernetes objects are changed
# by steps 2, 3, 5, 7 and 8, only on the clusters of the kubeconfig.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2

KUBECONFIG_IN=""; NAME=""; OUT_DIR=""; CONTEXTS=""; REGIONS=""; BASELINES=()
PEAK_RPS=""; FROM=1; HUBBLE_FIX=1; RUNS=1; CALIBRATE=auto; CAP_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kubeconfig) KUBECONFIG_IN="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --contexts) CONTEXTS="$2"; shift 2 ;;
        --regions) REGIONS="$2"; shift 2 ;;
        --baseline) BASELINES+=("--baseline" "$2"); shift 2 ;;
        --peak-rps) PEAK_RPS="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        --no-hubble-fix) HUBBLE_FIX=0; shift ;;
        --calibrate) CALIBRATE=yes; shift ;;
        --no-calibrate) CALIBRATE=no; shift ;;
        --k) CAP_ARGS+=(--k "$2"); shift 2 ;;
        --u-cpu) CAP_ARGS+=(--u-cpu "$2"); shift 2 ;;
        --no-runs) RUNS=0; shift ;;
        -h|--help) sed -n '2,59p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

[[ -n "$KUBECONFIG_IN" ]] || { echo "--kubeconfig is required (see --help)" >&2; exit 2; }
[[ -f "$KUBECONFIG_IN" ]] || { echo "kubeconfig not found: $KUBECONFIG_IN" >&2; exit 2; }
if [[ -z "$NAME" ]]; then NAME="$(basename "$KUBECONFIG_IN")"; NAME="${NAME%.*}"; fi
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--name: letters, digits, . _ - only" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-$HOME/Desktop/dmos-iac/$NAME}"

PY="${DMOS_PYTHON:-}"
if [[ -z "$PY" ]]; then
    for c in "$REPO/venv/Scripts/python.exe" "$REPO/venv/bin/python" python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
fi
FLEET=("$PY" "$REPO/scripts/kubephos_fleet.py")

hdr()  { printf '\n\033[1;36m=== [%s] %s\033[0m\n' "$1" "$2"; }
die()  { printf '\033[31m  x  %s\033[0m\n' "$*" >&2; exit 1; }
step() { [[ "$FROM" -le "$1" ]]; }

for tool in kubectl helm jq curl pwsh k6 "$PY"; do
    command -v "$tool" >/dev/null 2>&1 || die "missing in PATH: $tool"
done
echo "  fleet '$NAME' -> $OUT_DIR   (python: $PY)"

# Steps 3, 5, 7 and 8 reset the workload, send load and stop DMOS: never during a run
if command -v tasklist >/dev/null 2>&1 && tasklist //FI "IMAGENAME eq k6.exe" 2>/dev/null | grep -qi "k6.exe"; then
    die "k6 is running on this machine (a run in progress?): wait for it to finish"
fi

# ── 1. discover ─────────────────────────────────────────────────────────────
if step 1; then
    hdr 1 "Discover the clusters of $(basename "$KUBECONFIG_IN")"
    args=(discover --out-dir "$OUT_DIR" --kubeconfig "$KUBECONFIG_IN" --name "$NAME")
    [[ -n "$CONTEXTS" ]] && args+=(--contexts "$CONTEXTS")
    "${FLEET[@]}" "${args[@]}" || die "blocking problems: fix them in Kubephos and re-run"
fi
[[ -f "$OUT_DIR/fleet.json" ]] || die "$OUT_DIR/fleet.json missing: start from step 1"
PROM_NS=$(jq -r '.clusters[0].prometheus.namespace' "$OUT_DIR/fleet.json")
PROM_RELEASE=$(jq -r '.clusters[0].prometheus.release' "$OUT_DIR/fleet.json")
GW_CLASS=$(jq -r '.clusters[0].gateway_class' "$OUT_DIR/fleet.json")
APP_POOL=$(jq -r '.app_pool' "$OUT_DIR/fleet.json")
N=$(jq '.clusters | length' "$OUT_DIR/fleet.json")

# ── 2. Hubble labels ────────────────────────────────────────────────────────
if step 2; then
    hdr 2 "Hubble httpV2 labels"
    missing=$(jq -r '[.clusters[] | select(.hubble_missing | length > 0) | .name] | join(" ")' "$OUT_DIR/fleet.json")
    if [[ -z "$missing" ]]; then
        echo "  already configured on all $N clusters"
    elif [[ "$HUBBLE_FIX" == "1" ]]; then
        echo "  missing on: $missing  -> helm upgrade (same Cilium version) + agent restart"
        "${FLEET[@]}" hubble-fix --out-dir "$OUT_DIR" || die "hubble-fix failed"
    else
        die "missing on: $missing (re-run without --no-hubble-fix, or have Kubephos set the httpV2 labelsContext)"
    fi
fi

# ── 3. workload, policies, global services, Gateway ─────────────────────────
if step 3; then
    hdr 3 "Online Boutique + DMOS manifests + Gateway ($GW_CLASS)"
    pwsh -NoProfile -File scripts/setup-dmos-on-clusters.ps1 \
        -KubeconfigDir "$OUT_DIR" -KubeconfigFilter "kubeconfig-*" \
        -PrometheusNamespace "$PROM_NS" -PrometheusReleaseLabel "$PROM_RELEASE" \
        -WorkloadNodeSelector "kubephos.dev/pool=$APP_POOL" \
        -IngressMode gateway -GatewayClass "$GW_CLASS" -SkipCiliumUpgrade \
        || die "setup-dmos-on-clusters.ps1 failed"
fi

# ── 4. local.yaml + runtime files ───────────────────────────────────────────
write_local() {
    local args=(write-local --out-dir "$OUT_DIR")
    [[ -n "$REGIONS" ]] && args+=(--regions "$REGIONS")
    "${FLEET[@]}" "${args[@]}" "${BASELINES[@]}" "${CAP_ARGS[@]}" || die "write-local failed"
    bash scripts/render-runtime-files.sh >/dev/null || die "render-runtime-files.sh failed"
    echo "  .cache/ rendered"
}
if step 4; then
    hdr 4 "config/local.yaml for the fleet"
    write_local
fi

# ── 5. hardening, lifecycle, reset ──────────────────────────────────────────
if step 5; then
    hdr 5 "Hardening, pod lifecycle, reset"
    bash experiments/apply_resource_hardening.sh || die "apply_resource_hardening.sh failed"
    bash experiments/apply_pod_lifecycle.sh || die "apply_pod_lifecycle.sh failed"
    sleep 60   # rolling updates of the patches before the reset
    bash experiments/clean_reset.sh || die "clean_reset.sh: fleet not healthy"
fi

# ── 6. check + probe ────────────────────────────────────────────────────────
if step 6; then
    hdr 6 "Check the fleet"
    "${FLEET[@]}" check --out-dir "$OUT_DIR" --probe || die "check failed: see the lines marked x"
fi

# ── 7. calibration + replica caps ───────────────────────────────────────────
if step 7; then
    hdr 7 "Calibration and replica caps"
    if [[ "$CALIBRATE" == "yes" || ( "$CALIBRATE" == "auto" && ! -f "$OUT_DIR/calibration.json" ) ]]; then
        "${FLEET[@]}" calibrate --out-dir "$OUT_DIR" || die "calibration failed"
    elif [[ -f "$OUT_DIR/calibration.json" ]]; then
        echo "  using $OUT_DIR/calibration.json (k = $(jq -r .k_mcore "$OUT_DIR/calibration.json") mcore/req; --calibrate measures again)"
    else
        echo "  no calibration (--no-calibrate): caps from services.yaml"
    fi
    write_local
fi

if [[ "$RUNS" == "0" ]]; then
    printf '\n  Fleet ready. Runs: bash scripts/setup-kubephos-fleet.sh --kubeconfig %q --name %q --from 8\n' \
        "$KUBECONFIG_IN" "$NAME"
    exit 0
fi

# ── 8. runs: DMOS off, DMOS on ──────────────────────────────────────────────
if step 8; then
    hdr 8 "Test runs (DMOS off, then DMOS nca) — about 40 min"
    [[ -n "$PEAK_RPS" ]] && export DMOS_PEAK_RPS="$PEAK_RPS"
    run_dir_of() { sed -n 's/^RUN_DIR=//p' "$1" | tail -1 | tr -d '\r'; }
    LOGS="$OUT_DIR/runs"; mkdir -p "$LOGS"; TS=$(date +%Y%m%d-%H%M%S)

    bash experiments/clean_reset.sh || die "reset before the DMOS-off run failed"
    bash experiments/run_flash_crowd.sh --label "${NAME}-off" --no-dmos 2>&1 | tee "$LOGS/$TS-off.log"
    OFF_DIR=$(run_dir_of "$LOGS/$TS-off.log")
    [[ -n "$OFF_DIR" ]] || die "DMOS-off run produced no RUN_DIR (see $LOGS/$TS-off.log)"

    bash experiments/clean_reset.sh || die "reset before the DMOS run failed"
    echo "  starting DMOS (nca), log: $LOGS/$TS-dmos.log"
    DMOS_SCORE_MODEL=nca "$PY" -m src.dmos_main >"$LOGS/$TS-dmos.log" 2>&1 &
    for i in $(seq 1 60); do
        curl -s -o /dev/null --max-time 2 http://localhost:9090/metrics && break
        sleep 2
    done
    curl -s -o /dev/null --max-time 2 http://localhost:9090/metrics || die "DMOS did not start (see $LOGS/$TS-dmos.log)"
    bash experiments/run_flash_crowd.sh --label "${NAME}-nca" 2>&1 | tee "$LOGS/$TS-nca.log"
    NCA_DIR=$(run_dir_of "$LOGS/$TS-nca.log")
    bash experiments/clean_reset.sh >/dev/null 2>&1   # also stops DMOS
    [[ -n "$NCA_DIR" ]] || die "DMOS run produced no RUN_DIR (see $LOGS/$TS-nca.log)"

    if grep -q "FALLBACK server-side" "$LOGS/$TS-dmos.log"; then
        echo "  ! DMOS used the server-side demand fallback at least once (grep FALLBACK $LOGS/$TS-dmos.log)"
    fi
    hdr 8 "Comparison"
    "${FLEET[@]}" compare "$OFF_DIR" "$NCA_DIR" --out-dir "$OUT_DIR"
    echo "  runs: $OFF_DIR"
    echo "        $NCA_DIR"
fi
