#!/usr/bin/env bash
# =============================================================================
# validate-setup.sh — DMOS post-install smoke test
# =============================================================================
#
# Verifies that the 3-cluster fleet is healthy enough for DMOS to run.
# Exits 0 only if every check passes. Any failure prints the offending
# command and a hint, then exits non-zero so it can be wired into CI.
#
# Designed to run from the repo root:
#     bash scripts/validate-setup.sh
#
# Reads config/local.yaml for the IPs (no flags / no prompts).
# =============================================================================

set -uo pipefail

# ── Locate repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Colors ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
    BOLD=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

PASS=0
FAIL=0
SKIP=0

ok()   { printf '  %s✓%s %s\n' "${GREEN}" "${RESET}" "$*"; PASS=$((PASS+1)); }
nok()  { printf '  %s✗%s %s\n' "${RED}"   "${RESET}" "$*"; FAIL=$((FAIL+1)); }
skip() { printf '  %s-%s %s (skipped: %s)\n' "${YELLOW}" "${RESET}" "$1" "$2"; SKIP=$((SKIP+1)); }
hdr()  { printf '\n%s== %s ==%s\n' "${BOLD}" "$*" "${RESET}"; }

# ── Pre-flight ──────────────────────────────────────────────────────────────
hdr "Pre-flight"

if [[ ! -f "config/local.yaml" ]]; then
    nok "config/local.yaml not found — run scripts/bootstrap-assisted.sh first"
    exit 2
fi
ok "config/local.yaml exists"

for tool in kubectl helm curl python3; do
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool available"
    else
        nok "$tool not on PATH"
    fi
done

# Parse cluster IPs out of clusters.yaml using python (no yq dependency)
IPS=$(python3 - <<'PY' 2>/dev/null
import sys, yaml
try:
    with open("config/local.yaml") as f:
        d = yaml.safe_load(f)
    cs = d.get("clusters", [])
    if isinstance(cs, dict):
        cs = list(cs.values())
    for c in cs:
        print(c.get("name", "?"), c.get("ip", "?"))
except Exception as e:
    print(f"ERROR {e}", file=sys.stderr)
    sys.exit(1)
PY
)

if [[ -z "$IPS" ]]; then
    nok "Could not parse cluster IPs from config/local.yaml"
    exit 2
fi
ok "Parsed cluster list:"
echo "$IPS" | sed 's/^/    /'

# ── Per-cluster checks ──────────────────────────────────────────────────────
declare -A IP_OF
while read -r name ip; do
    [[ -n "$name" && -n "$ip" ]] && IP_OF["$name"]="$ip"
done <<< "$IPS"

for name in "${!IP_OF[@]}"; do
    ip="${IP_OF[$name]}"
    kubeconfig=".kube/${name}.yaml"

    hdr "Checks for ${name} (${ip})"

    if [[ ! -f "$kubeconfig" ]]; then
        nok "$kubeconfig missing"
        continue
    fi
    ok "$kubeconfig present"

    # 1. API reachable
    if KUBECONFIG="$kubeconfig" kubectl get --raw=/healthz >/dev/null 2>&1; then
        ok "K8s API reachable"
    else
        nok "K8s API unreachable — check firewall on TCP 6443"
        continue
    fi

    # 2. Cilium pods Running
    not_ready=$(KUBECONFIG="$kubeconfig" kubectl -n kube-system get pods \
                -l k8s-app=cilium --no-headers 2>/dev/null \
                | awk '$3 != "Running" {n++} END{print n+0}')
    if [[ "$not_ready" == "0" ]]; then
        ok "Cilium pods Running"
    else
        nok "Cilium pods not Running (${not_ready} bad) — check 'kubectl logs -n kube-system -l k8s-app=cilium'"
    fi

    # 3. ClusterMesh status
    if command -v cilium >/dev/null 2>&1; then
        if KUBECONFIG="$kubeconfig" cilium clustermesh status --wait-duration=5s >/dev/null 2>&1; then
            ok "ClusterMesh connected"
        else
            nok "ClusterMesh not connected — check NodePort 32379 between nodes"
        fi
    else
        skip "ClusterMesh status" "cilium CLI not installed"
    fi

    # 4. Prometheus NodePort: 32090 (kube-prometheus-stack via proxmox-k3s).
    #    Legacy testbed usava 30090 (standalone prometheus chart). Auto-detect.
    prom_port=32090
    if ! curl -fsS --max-time 3 "http://${ip}:${prom_port}/-/ready" >/dev/null 2>&1; then
        # Fallback al port legacy 30090
        if curl -fsS --max-time 3 "http://${ip}:30090/-/ready" >/dev/null 2>&1; then
            prom_port=30090
        fi
    fi
    if curl -fsS --max-time 5 "http://${ip}:${prom_port}/-/ready" >/dev/null 2>&1; then
        ok "Prometheus reachable on ${ip}:${prom_port}"
    else
        nok "Prometheus NOT reachable on ${ip} (provato :32090 e :30090)"
    fi

    # 5. ping_exporter Service exists in observability namespace
    if KUBECONFIG="$kubeconfig" kubectl -n observability get svc ping-exporter >/dev/null 2>&1; then
        ok "ping_exporter Service exists"
    else
        nok "ping_exporter not deployed in 'observability' namespace"
    fi

    # 6. Online Boutique frontend Ready
    fe_ready=$(KUBECONFIG="$kubeconfig" kubectl -n online-boutique get deploy frontend \
               -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
    if [[ "${fe_ready:-0}" -ge 1 ]]; then
        ok "Online Boutique frontend Ready (${fe_ready} replicas)"
    else
        nok "Online Boutique frontend not Ready"
    fi

    # 7. Hubble L7 metric available
    metric=$(curl -fsS --max-time 5 "http://${ip}:${prom_port}/api/v1/query?query=hubble_http_requests_total" 2>/dev/null \
             | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d["data"]["result"]))' 2>/dev/null \
             || echo 0)
    if [[ "${metric:-0}" -gt 0 ]]; then
        ok "Hubble L7 metric 'hubble_http_requests_total' present (${metric} series)"
    else
        nok "Hubble L7 metric absent — apply deployments/cnp-*-l7.yaml and helm upgrade Cilium con hubble.metrics.enabled"
    fi

    # 8. ping RTT metric
    rtt=$(curl -fsS --max-time 5 "http://${ip}:${prom_port}/api/v1/query?query=ping_rtt_mean_seconds" 2>/dev/null \
          | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d["data"]["result"]))' 2>/dev/null \
          || echo 0)
    if [[ "${rtt:-0}" -gt 0 ]]; then
        ok "ping_rtt_mean_seconds present (${rtt} series)"
    else
        nok "ping_rtt_mean_seconds absent — check ping_exporter logs and ServiceMonitor labels"
    fi
done

# ── DMOS local prerequisites ────────────────────────────────────────────────
hdr "DMOS local environment"

if [[ -x "venv/bin/python" || -x "venv/Scripts/python.exe" ]]; then
    ok "Python virtualenv present at ./venv"
else
    nok "./venv missing — run: python3.12 -m venv venv && pip install -r requirements.txt"
fi

PYBIN="venv/bin/python"
[[ -x "venv/Scripts/python.exe" ]] && PYBIN="venv/Scripts/python.exe"
if [[ -x "$PYBIN" ]]; then
    if "$PYBIN" -c "from src.utils.config_loader import ConfigLoader; ConfigLoader('config')" 2>/dev/null; then
        ok "ConfigLoader parses config/ cleanly"
    else
        nok "ConfigLoader failed — check config/ files and weights.yaml"
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────────
hdr "Summary"
printf '  %sPASS%s: %d   %sFAIL%s: %d   %sSKIP%s: %d\n' \
    "${GREEN}" "${RESET}" "$PASS" \
    "${RED}"   "${RESET}" "$FAIL" \
    "${YELLOW}" "${RESET}" "$SKIP"

if [[ $FAIL -eq 0 ]]; then
    printf '\n%sAll checks passed — you can run: %spython -m src.dmos_main%s\n' \
        "${GREEN}" "${BOLD}" "${RESET}"
    exit 0
else
    printf '\n%s%d check(s) failed.%s See docs/setup-guide.md → Troubleshooting.\n' \
        "${RED}" "$FAIL" "${RESET}"
    exit 1
fi
