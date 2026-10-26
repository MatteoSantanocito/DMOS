#!/usr/bin/env bash
# =============================================================================
# bootstrap-assisted.sh — DMOS guided setup orchestrator
# =============================================================================
#
# Asks for cluster count and per-cluster details (IP, region, display name,
# carbon baseline, pricing, netem delay), probes each node via SSH for
# CPU/RAM, and writes a single `config/local.yaml` — the only
# environment-specific file DMOS reads. Helm values for Cilium and
# ping_exporter, env vars for k6 / shell scripts, and (optional) the
# alertmanager ConfigMap are rendered on-demand from local.yaml by
# `scripts/render-runtime-files.sh`, called automatically at the end.
#
# A complete command transcript is written to `bootstrap-commands.sh` for
# step-by-step execution (gitignored).
#
# Re-runnable: existing `config/local.yaml` is NOT overwritten unless you
# pass `--force`.
#
# Prerequisites on every node BEFORE running this script:
#   1. Ubuntu installed, SSH server enabled.
#   2. Operator's SSH public key in ~ubuntu/.ssh/authorized_keys.
#   3. Passwordless sudo for the `ubuntu` user
#      (echo "ubuntu ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/dmos).
# Without these the SSH probe step will fail with a clear message and the
# script will fall back to default CPU/RAM values that you can edit later.
# =============================================================================

set -euo pipefail

# ── CLI flags ────────────────────────────────────────────────────────────────
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '2,28p' "$0"; exit 0 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Colors ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; DIM=$'\e[2m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'
    YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
    BOLD=''; DIM=''; CYAN=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi
say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n%s%s── %s ───────────────────────────────────────────────────────────────%s\n' \
                "${BOLD}" "${CYAN}" "$*" "${RESET}"; }
ok()   { printf '%s✓ %s%s\n' "${GREEN}" "$*" "${RESET}"; }
warn() { printf '%s! %s%s\n' "${YELLOW}" "$*" "${RESET}"; }
fail() { printf '%s✗ %s%s\n' "${RED}"   "$*" "${RESET}"; exit 1; }

[[ -d "config" ]] || fail "Run this script from the DMOS repo root (config/ not found)."

if [[ -f "config/local.yaml" && $FORCE -eq 0 ]]; then
    warn "config/local.yaml already exists — will not overwrite."
    warn "Pass --force to regenerate, or delete the file first."
    exit 1
fi

# ── Defaults for the original 3-cluster DMOS thesis topology ────────────────
DEFAULT_REGIONS=("DE" "FR" "PL")
DEFAULT_DISPLAYS=("EU-DE (node1)" "EU-FR (node2)" "EU-PL (node3)")
DEFAULT_LOCATIONS=("Frankfurt" "Paris" "Warsaw")
DEFAULT_CARBON=("350" "80" "650")
DEFAULT_PRICING=("0.12" "0.15" "0.08")
DEFAULT_NETEM_DELAY=("0" "150" "350")

hdr "Cluster topology"
say "DMOS supports any N >= 2 K3s clusters connected via Cilium ClusterMesh."
say "The default of 3 reproduces the original DMOS thesis topology"
say "(DE/Frankfurt 0ms, FR/Paris 150ms, PL/Warsaw 350ms)."
say ""

read -r -p "  How many clusters? [3]: " CLUSTER_COUNT
CLUSTER_COUNT=${CLUSTER_COUNT:-3}
[[ "$CLUSTER_COUNT" =~ ^[0-9]+$ ]] && (( CLUSTER_COUNT >= 2 )) \
    || fail "Cluster count must be an integer >= 2 (got: ${CLUSTER_COUNT})."

ok "Configuring ${CLUSTER_COUNT} clusters"

declare -a IPS=() REGIONS=() DISPLAYS=() LOCATIONS=() CARBONS=() PRICINGS=() NETEM_DELAYS=()

prompt_ip() {
    local label="$1" v
    while true; do
        read -r -p "    ${label}: " v
        [[ "$v" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && { printf '%s\n' "$v"; return; }
        warn "    Not a valid IPv4 address — try again."
    done
}

for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1))
    say ""
    say "  ── cluster${n} ──"

    ip=$(prompt_ip "cluster${n} IP")
    IPS+=("$ip")

    default_region="${DEFAULT_REGIONS[$i]:-}"
    if [[ -n "$default_region" ]]; then
        read -r -p "    Region code (Electricity Maps zone) [${default_region}]: " v
        REGIONS+=("${v:-$default_region}")
    else
        read -r -p "    Region code (e.g. US-VA, IT-NO): " v
        [[ -z "$v" ]] && fail "Region cannot be empty."
        REGIONS+=("$v")
    fi

    default_display="${DEFAULT_DISPLAYS[$i]:-${REGIONS[$i]} (node${n})}"
    read -r -p "    Display name [${default_display}]: " v
    DISPLAYS+=("${v:-$default_display}")

    default_loc="${DEFAULT_LOCATIONS[$i]:-${REGIONS[$i]}}"
    read -r -p "    Location [${default_loc}]: " v
    LOCATIONS+=("${v:-$default_loc}")

    default_c="${DEFAULT_CARBON[$i]:-300}"
    read -r -p "    Carbon baseline gCO2/kWh [${default_c}]: " v
    [[ -n "$v" && ! "$v" =~ ^[0-9]+$ ]] && fail "Carbon baseline must be an integer."
    CARBONS+=("${v:-$default_c}")

    default_p="${DEFAULT_PRICING[$i]:-0.10}"
    read -r -p "    Cost €/replica/hour (informational) [${default_p}]: " v
    PRICINGS+=("${v:-$default_p}")

    default_netem="${DEFAULT_NETEM_DELAY[$i]:-0}"
    read -r -p "    netem delay ms (0 = no shaping) [${default_netem}]: " v
    [[ -n "$v" && ! "$v" =~ ^[0-9]+$ ]] && fail "netem delay must be an integer."
    NETEM_DELAYS+=("${v:-$default_netem}")
done

# IP uniqueness check
for ((i=0; i<CLUSTER_COUNT; i++)); do
    for ((j=i+1; j<CLUSTER_COUNT; j++)); do
        [[ "${IPS[$i]}" == "${IPS[$j]}" ]] && \
            fail "Duplicate IP: cluster$((i+1)) and cluster$((j+1)) both have ${IPS[$i]}."
    done
done

say ""
read -r -p "  SSH username on all ${CLUSTER_COUNT} nodes [ubuntu]: " SSH_USER
SSH_USER=${SSH_USER:-ubuntu}
read -r -p "  SSH key path [~/.ssh/dmos_nodes]: " SSH_KEY
SSH_KEY=${SSH_KEY:-~/.ssh/dmos_nodes}
SSH_KEY_EXPANDED="${SSH_KEY/#\~/$HOME}"
read -r -p "  Network interface for tc netem [ens18]: " NETEM_IFACE
NETEM_IFACE=${NETEM_IFACE:-ens18}
read -r -p "  Alertmanager webhook URL (optional, press Enter to skip): " WEBHOOK_URL

ok "Got it: ${SSH_USER}@{$(IFS=,; echo "${IPS[*]}")}"

# ── Auto-detect CPU / RAM via SSH ───────────────────────────────────────────
hdr "Detecting hardware on each node via SSH"
say "Probing each node for CPU cores and memory. Falls back to defaults"
say "(4 cores / 8 GB) if a node is unreachable."
say ""

declare -a CPUS=() MEMS=()
SSH_OPTS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new
          -o BatchMode=yes -o LogLevel=ERROR -i "$SSH_KEY_EXPANDED")

for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1)); ip="${IPS[$i]}"
    rcmd='echo "$(nproc):$(free -m | awk "/^Mem:/{printf \"%d\", (\$2 + 1023) / 1024}")"'
    if probe=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "$rcmd" 2>/dev/null) && \
       [[ "$probe" =~ ^[0-9]+:[0-9]+$ ]]; then
        cpu="${probe%:*}"; mem="${probe#*:}"
        if (( cpu < 1 || mem < 1 )); then
            warn "  cluster${n} (${ip}): implausible probe (${cpu}c/${mem}GB) — using defaults"
            cpu=4; mem=8
        else
            ok "  cluster${n} (${ip}): ${cpu} cores, ${mem} GB RAM"
        fi
    else
        warn "  cluster${n} (${ip}): SSH probe failed — using defaults (4c/8GB)"
        cpu=4; mem=8
    fi
    CPUS+=("$cpu"); MEMS+=("$mem")
done

# ── Generate config/local.yaml ──────────────────────────────────────────────
hdr "Generating config/local.yaml"

{
    cat <<EOF
# Auto-generated by scripts/bootstrap-assisted.sh on $(date -u +'%Y-%m-%dT%H:%M:%SZ')
# Single source of truth for environment-specific data.
# Re-run the script (with --force) to regenerate.
# This file is gitignored — do not commit.

ssh:
  user: ${SSH_USER}
  key: ${SSH_KEY}

clusters:
EOF
    for ((i=0; i<CLUSTER_COUNT; i++)); do
        n=$((i+1))
        netem_jitter=0
        (( NETEM_DELAYS[i] > 0 )) && netem_jitter=20
        cat <<EOF
  - name: cluster${n}
    display_name: "${DISPLAYS[$i]}"
    region: ${REGIONS[$i]}
    location: ${LOCATIONS[$i]}
    ip: ${IPS[$i]}
    kubeconfig_path: ".kube/cluster${n}.yaml"
    resources:
      cpu_cores: ${CPUS[$i]}
      memory_gb: ${MEMS[$i]}
    pricing:
      cost_per_replica_hour: ${PRICINGS[$i]}
    carbon:
      region_code: "${REGIONS[$i]}"
      baseline_gco2_kwh: ${CARBONS[$i]}
    latency:
      baseline_ms: 2.0
    netem:
      delay_ms: ${NETEM_DELAYS[$i]}
      jitter_ms: ${netem_jitter}
      interface: "${NETEM_IFACE}"

EOF
    done
    cat <<EOF
global:
  scheduling_interval_seconds: 30
  score_request_timeout_seconds: 5
EOF

    if [[ -n "$WEBHOOK_URL" ]]; then
        cat <<EOF

alertmanager:
  webhook_url: "${WEBHOOK_URL}"
EOF
    fi
} > config/local.yaml
ok "Wrote config/local.yaml"

# ── Render runtime files (Helm values, env vars, alertmanager config) ───────
hdr "Rendering runtime files from local.yaml"
bash scripts/render-runtime-files.sh

# ── Generate bootstrap-commands.sh transcript ───────────────────────────────
TRANSCRIPT="bootstrap-commands.sh"
exec 3>"$TRANSCRIPT"
emit() { printf '%s\n' "$*" >&3; printf '%s%s%s\n' "${DIM}" "$*" "${RESET}"; }

emit "#!/usr/bin/env bash"
emit "# Auto-generated by scripts/bootstrap-assisted.sh on $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
emit "# Cluster count: ${CLUSTER_COUNT}"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1))
    emit "#   cluster${n} = ${IPS[$i]} (${REGIONS[$i]} / ${LOCATIONS[$i]})"
done
emit "set -euo pipefail"
emit ""
emit "# Source env vars rendered from config/local.yaml"
emit 'source .cache/test-env.sh'
emit ""

hdr "Step A — Install K3s on the ${CLUSTER_COUNT} nodes"
emit "# ── A. Install K3s on each node ──"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1)); ip="${IPS[$i]}"
    cidr="10.$((41+n)).0.0/16"
    svc_cidr="10.$((141+n)).0.0/16"
    emit ""
    emit "ssh -i ${SSH_KEY} ${SSH_USER}@${ip} '"
    emit "  curl -sfL https://get.k3s.io | sh -s - server \\"
    emit "    --flannel-backend=none --disable-network-policy \\"
    emit "    --disable=traefik --disable=servicelb \\"
    emit "    --cluster-cidr=${cidr} --service-cidr=${svc_cidr} \\"
    emit "    --node-ip=${ip} --tls-san=${ip}"
    emit "'"
done

hdr "Step B — Pull kubeconfigs"
emit ""
emit "# ── B. Pull kubeconfigs ──"
emit "mkdir -p .kube"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1)); ip="${IPS[$i]}"
    emit "scp -i ${SSH_KEY} ${SSH_USER}@${ip}:/etc/rancher/k3s/k3s.yaml .kube/cluster${n}.yaml"
    emit "sudo chown \$(id -u):\$(id -g) .kube/cluster${n}.yaml"
    emit "sed -i 's/127\\.0\\.0\\.1/${ip}/g' .kube/cluster${n}.yaml"
done

hdr "Step C — Install Cilium"
emit ""
emit "# ── C. Cilium (uses .cache/cilium-clusterN-values.yaml rendered from local.yaml) ──"
emit "helm repo add cilium https://helm.cilium.io || true"
emit "helm repo update"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1))
    emit "helm upgrade --install cilium cilium/cilium \\"
    emit "    --kubeconfig .kube/cluster${n}.yaml \\"
    emit "    -n kube-system --version 1.16.5 \\"
    emit "    -f .cache/cilium-cluster${n}-values.yaml"
done

hdr "Step D — Connect ClusterMesh"
emit ""
emit "# ── D. ClusterMesh (every unordered pair i<j) ──"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1))
    emit "cilium clustermesh enable --kubeconfig .kube/cluster${n}.yaml"
done
for ((i=0; i<CLUSTER_COUNT; i++)); do
    for ((j=i+1; j<CLUSTER_COUNT; j++)); do
        emit "cilium clustermesh connect --kubeconfig .kube/cluster$((i+1)).yaml --destination-context cluster$((j+1))"
    done
done
emit "cilium clustermesh status --kubeconfig .kube/cluster1.yaml --wait"

hdr "Step E — Install kube-prometheus-stack"
emit ""
emit "# ── E. Prometheus stack ──"
emit "helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true"
emit "helm repo update"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1))
    emit "helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \\"
    emit "    --kubeconfig .kube/cluster${n}.yaml \\"
    emit "    -n monitoring --create-namespace \\"
    emit "    -f config/prometheus-helm-values.yaml"
done

hdr "Step F — Install ping_exporter"
emit ""
emit "# ── F. ping_exporter (uses .cache/ping-exporter-clusterN-values.yaml) ──"
emit "helm repo add czerwonk https://charts.0l.de || true"
emit "helm repo update"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1))
    emit "helm upgrade --install ping-exporter czerwonk/ping-exporter \\"
    emit "    --kubeconfig .kube/cluster${n}.yaml \\"
    emit "    -n observability --create-namespace \\"
    emit "    -f .cache/ping-exporter-cluster${n}-values.yaml"
done

hdr "Step G — Deploy Online Boutique + DMOS L7 policies"
emit ""
emit "# ── G. Workload + L7 policies + Cilium Ingress + Global Services ──"
emit "OB_URL='https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml'"
for ((i=0; i<CLUSTER_COUNT; i++)); do
    n=$((i+1))
    emit "KUBECONFIG=.kube/cluster${n}.yaml kubectl create namespace online-boutique --dry-run=client -o yaml | kubectl --kubeconfig .kube/cluster${n}.yaml apply -f -"
    emit "KUBECONFIG=.kube/cluster${n}.yaml kubectl apply -n online-boutique -f \"\$OB_URL\""
    for f in 00-namespace.yaml cnp-frontend-l7.yaml cnp-backends-l7.yaml \
             ingress-frontend-cilium.yaml global-service-frontend.yaml \
             global-service-backends.yaml hubble-servicemonitor-monitoring.yaml; do
        emit "KUBECONFIG=.kube/cluster${n}.yaml kubectl apply -f deployments/${f}"
    done
done

hdr "Step H — (Optional) Apply tc netem from local.yaml"
emit ""
emit "# ── H. tc netem (delays read from config/local.yaml via .cache/test-env.json) ──"
emit "bash scripts/setup-netem-and-ping.sh netem-only"

hdr "Step I — Validate"
emit ""
emit "# ── I. Smoke test ──"
emit "bash scripts/validate-setup.sh"

hdr "Step J — Run DMOS"
emit ""
emit "# ── J. Start DMOS ──"
emit "source venv/bin/activate"
emit "python -m src.dmos_main"

exec 3>&-

# ── Final hand-off ──────────────────────────────────────────────────────────
hdr "Done"
ok  "Generated: ${BOLD}config/local.yaml${RESET} (gitignored)"
ok  "Rendered:  ${BOLD}.cache/${RESET} (Helm values + test env, gitignored)"
ok  "Transcript: ${BOLD}${TRANSCRIPT}${RESET} (gitignored)"
say ""
say "Next steps:"
say "  1. Inspect ${BOLD}${TRANSCRIPT}${RESET} top-to-bottom."
say "  2. Run each ── section ── in order, verifying each step before moving on."
say "  3. After Step G, ${BOLD}bash scripts/validate-setup.sh${RESET} should exit 0."
say "  4. ${BOLD}python -m src.dmos_main${RESET}"
say ""
warn "This script DOES NOT execute the cluster commands — by design."
warn "Run them yourself and watch the output of each."
