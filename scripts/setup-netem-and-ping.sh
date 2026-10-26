#!/usr/bin/env bash
# =============================================================================
# setup-netem-and-ping.sh
# =============================================================================
#
# Script unico per Fase 5+6 del setup DMOS multi-cluster:
#   1. tc netem sui worker (delay inter-cluster geografico simulato)
#   2. ping_exporter via Helm su ogni cluster (per metrica ping_rtt_mean_seconds)
#
# Idempotente: i comandi `tc qdisc replace` e `helm upgrade --install`
# possono essere rilanciati senza side-effect.
#
# Prerequisiti:
#   - config/local.yaml gia' configurato col fleet di cluster
#   - .cache/ generato da scripts/render-runtime-files.sh (lo rigenera automaticamente
#     all'inizio se non e' fresh)
#   - VPN al lab attiva (per SSH ai worker e per kubectl/helm ai cluster)
#   - SSH key in ~/.proxmox-k3s/id_ed25519 (default usato da render-runtime-files.sh)
#   - helm, kubectl, jq, ssh nel PATH
#
# Usage:
#   bash scripts/setup-netem-and-ping.sh             # apply tutto (default)
#   bash scripts/setup-netem-and-ping.sh apply
#   bash scripts/setup-netem-and-ping.sh netem-only
#   bash scripts/setup-netem-and-ping.sh ping-only
#   bash scripts/setup-netem-and-ping.sh verify       # solo verifica, no apply
#   bash scripts/setup-netem-and-ping.sh remove-netem # rimuovi netem (torna a LAN pura)
#
# Exit codes:
#   0 = tutto OK
#   1 = errore in qualche step
#   2 = mancanza di prerequisiti
# =============================================================================

set -euo pipefail

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache"
ENV_JSON="${CACHE_DIR}/test-env.json"

# ── Colori ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'
    YELLOW=$'\e[33m'; RED=$'\e[31m'; DIM=$'\e[2m'; RESET=$'\e[0m'
else
    BOLD=''; CYAN=''; GREEN=''; YELLOW=''; RED=''; DIM=''; RESET=''
fi

hdr()  { printf '\n%s%s%s %s%s\n' "${BOLD}" "${CYAN}" "===" "$*" "${RESET}"; }
step() { printf '  %s→%s %s\n' "${CYAN}" "${RESET}" "$*"; }
ok()   { printf '  %s✓%s %s\n' "${GREEN}" "${RESET}" "$*"; }
warn() { printf '  %s!%s %s\n' "${YELLOW}" "${RESET}" "$*"; }
fail() { printf '  %s✗%s %s\n' "${RED}" "${RESET}" "$*"; exit 1; }
info() { printf '  %s%s%s\n' "${DIM}" "$*" "${RESET}"; }

# ── Pre-flight check ────────────────────────────────────────────────────────
preflight() {
    hdr "Pre-flight"

    local missing=()
    for tool in helm kubectl jq ssh; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing+=("$tool")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        fail "Tool mancanti nel PATH: ${missing[*]}"
    fi
    ok "Tool richiesti (helm, kubectl, jq, ssh) presenti"

    if [[ ! -f "${REPO_ROOT}/config/local.yaml" ]]; then
        fail "config/local.yaml non trovato — crea il file prima di lanciare lo script"
    fi
    ok "config/local.yaml trovato"

    # Rigenera .cache se manca (idempotente)
    if [[ ! -f "$ENV_JSON" ]]; then
        step "Rigenero .cache/ da config/local.yaml..."
        bash "${SCRIPT_DIR}/render-runtime-files.sh"
    fi

    if [[ ! -f "$ENV_JSON" ]]; then
        fail ".cache/test-env.json non e' stato creato — controlla render-runtime-files.sh"
    fi
    ok "$ENV_JSON pronto"

    # SSH key
    local ssh_key
    ssh_key=$(jq -r '.ssh.key' "$ENV_JSON" | tr -d '\r')
    # Espande ~ a $HOME
    ssh_key="${ssh_key/#\~/$HOME}"
    if [[ ! -f "$ssh_key" ]]; then
        fail "SSH key non trovata: $ssh_key (configurata in local.yaml > ssh.key)"
    fi
    ok "SSH key: $ssh_key"

    # Numero cluster
    local n_clusters
    n_clusters=$(jq '.clusters | length' "$ENV_JSON" | tr -d '\r')
    ok "Cluster nel fleet: $n_clusters"
    if [[ "$n_clusters" -lt 2 ]]; then
        fail "Servono almeno 2 cluster per ping_exporter (peer-to-peer)"
    fi
}

# ── Phase A — tc netem ──────────────────────────────────────────────────────
apply_netem() {
    hdr "Fase A — tc netem"

    local ssh_key ssh_user
    ssh_key=$(jq -r '.ssh.key' "$ENV_JSON" | tr -d '\r')
    ssh_key="${ssh_key/#\~/$HOME}"
    ssh_user=$(jq -r '.ssh.user' "$ENV_JSON" | tr -d '\r')

    local applied=0
    local skipped=0

    while IFS=$'\t' read -r name ip delay jitter iface; do
        if [[ -z "$delay" || "$delay" == "0" || "$delay" == "null" ]]; then
            info "$name (${ip}): delay=0 — skip (baseline)"
            skipped=$((skipped+1))
            continue
        fi
        if [[ -z "$iface" || "$iface" == "null" ]]; then
            warn "$name: interface non specificata — skip"
            skipped=$((skipped+1))
            continue
        fi

        step "$name (${ip}): applico delay=${delay}ms jitter=${jitter}ms su ${iface}"

        # Usa `qdisc replace` per idempotenza (add fallisce se gia' presente).
        # Se non c'e' nessuna regola, replace ne crea una nuova.
        local cmd="sudo tc qdisc replace dev ${iface} root netem delay ${delay}ms ${jitter}ms distribution normal"
        # -n: no stdin (altrimenti ssh consuma lo stdin del while-read e il loop si ferma)
        if ssh -n -i "$ssh_key" \
               -o BatchMode=yes \
               -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=5 \
               "${ssh_user}@${ip}" "$cmd" 2>&1; then
            ok "netem applicato su $name"
            applied=$((applied+1))
        else
            warn "netem fallito su $name — controlla SSH/sudo"
        fi
    done < <(
        jq -r '.clusters[] |
               [.name, .ip,
                (.netem.delay_ms    // 0),
                (.netem.jitter_ms   // 0),
                (.netem.interface   // "")]
               | @tsv' "$ENV_JSON" | tr -d '\r'
    )

    info "Applicati: $applied, skip: $skipped"
}

# ── Phase B — ping_exporter ──────────────────────────────────────────────────
install_ping_exporter() {
    hdr "Fase B — ping_exporter"

    step "Aggiungo repo Helm czerwonk (idempotente)..."
    helm repo add czerwonk https://czerwonk.github.io/ping_exporter 2>&1 | grep -v "already exists" || true
    helm repo update czerwonk >/dev/null
    ok "Repo czerwonk aggiornato"

    local installed=0
    local failed=0

    while IFS=$'\t' read -r name kubeconfig; do
        # Espande il path del kubeconfig se contiene ~
        kubeconfig="${kubeconfig/#\~/$HOME}"
        # Per path relativi (es. ./kubeconfig-...), li risolviamo da REPO_ROOT
        if [[ "$kubeconfig" != /* ]] && [[ "$kubeconfig" != [A-Za-z]:* ]]; then
            kubeconfig="${REPO_ROOT}/${kubeconfig}"
        fi
        # Su Windows il path puo' essere C:/... con slash, helm/kubectl lo gestiscono OK.

        if [[ ! -f "$kubeconfig" ]]; then
            warn "$name: kubeconfig mancante in $kubeconfig — skip"
            failed=$((failed+1))
            continue
        fi

        # I values file sono gia' renderizzati da render-runtime-files.sh
        local values="${CACHE_DIR}/ping-exporter-${name}-values.yaml"
        if [[ ! -f "$values" ]]; then
            warn "$name: values mancanti ($values) — rigenera con render-runtime-files.sh"
            failed=$((failed+1))
            continue
        fi

        step "$name: helm upgrade --install ping-exporter..."

        if helm upgrade --install ping-exporter czerwonk/ping-exporter \
               --kubeconfig "$kubeconfig" \
               -n observability \
               --create-namespace \
               -f "$values" \
               --wait \
               --timeout 3m 2>&1 | tail -5; then
            ok "$name: ping_exporter installato"
            # Il chart czerwonk/ping-exporter NON supporta serviceMonitor.labels:
            # crea il ServiceMonitor SENZA la label `release` che il Prometheus
            # Operator di kube-prometheus-stack usa per lo scoping. Senza quella
            # label il ServiceMonitor non viene scoperto -> niente scrape ->
            # ping_rtt_mean_seconds resta assente. La applichiamo a mano qui
            # (idempotente con --overwrite), cosi' non serve un fix manuale
            # dopo ogni re-provisioning.
            if kubectl --kubeconfig "$kubeconfig" -n observability label servicemonitor \
                   ping-exporter release=prometheus-stack --overwrite >/dev/null 2>&1; then
                ok "$name: ServiceMonitor etichettato (release=prometheus-stack)"
            else
                warn "$name: label ServiceMonitor non applicata (verifica a mano)"
            fi
            installed=$((installed+1))
        else
            warn "$name: helm install fallito"
            failed=$((failed+1))
        fi
    done < <(
        jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r'
    )

    info "Installati: $installed, falliti/skipped: $failed"
}

# ── Verify ──────────────────────────────────────────────────────────────────
verify() {
    hdr "Verifica finale"

    local ssh_key ssh_user
    ssh_key=$(jq -r '.ssh.key' "$ENV_JSON" | tr -d '\r')
    ssh_key="${ssh_key/#\~/$HOME}"
    ssh_user=$(jq -r '.ssh.user' "$ENV_JSON" | tr -d '\r')

    # A) Verifica netem
    hdr "  [A] tc qdisc show su ogni worker"
    while IFS=$'\t' read -r name ip iface; do
        printf '\n  %s%s%s\n' "${BOLD}" "$name ($ip)" "${RESET}"
        if [[ -z "$iface" || "$iface" == "null" ]]; then
            info "    interfaccia non specificata — skip"
            continue
        fi
        # -n: no stdin (altrimenti ssh consuma lo stdin del while-read e il loop si ferma)
        ssh -n -i "$ssh_key" \
            -o BatchMode=yes \
            -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ConnectTimeout=3 \
            "${ssh_user}@${ip}" "tc qdisc show dev ${iface} 2>/dev/null" \
            | sed 's/^/    /' || warn "SSH a $ip fallito"
    done < <(
        jq -r '.clusters[] | [.name, .ip, (.netem.interface // "")] | @tsv' "$ENV_JSON" | tr -d '\r'
    )

    # B) Verifica ping_exporter pod
    hdr "  [B] ping_exporter pod status"
    while IFS=$'\t' read -r name kubeconfig prom_url; do
        kubeconfig="${kubeconfig/#\~/$HOME}"
        if [[ "$kubeconfig" != /* ]] && [[ "$kubeconfig" != [A-Za-z]:* ]]; then
            kubeconfig="${REPO_ROOT}/${kubeconfig}"
        fi
        printf '\n  %s%s%s\n' "${BOLD}" "$name" "${RESET}"
        if [[ ! -f "$kubeconfig" ]]; then
            warn "    kubeconfig mancante"
            continue
        fi
        # Pod
        if kubectl --kubeconfig "$kubeconfig" -n observability get pods \
               -l app.kubernetes.io/name=ping-exporter -o wide --no-headers 2>/dev/null \
               | head -5 | sed 's/^/    /'; then
            :
        else
            warn "    nessun pod ping-exporter trovato"
        fi
        # Metric
        local metric_count
        metric_count=$(
            curl -s --max-time 5 \
                "${prom_url}/api/v1/query?query=ping_rtt_mean_seconds" 2>/dev/null \
            | jq -r '.data.result | length' 2>/dev/null | tr -d '\r' || echo 0
        )
        if [[ "${metric_count:-0}" -gt 0 ]]; then
            ok "    ping_rtt_mean_seconds = $metric_count serie su Prometheus"
        else
            warn "    ping_rtt_mean_seconds ANCORA assente (puo' richiedere 30-60s di scrape)"
        fi
    done < <(
        jq -r '.clusters[] | [.name, .kubeconfig_path, .prometheus_url] | @tsv' "$ENV_JSON" | tr -d '\r'
    )
}

# ── Remove netem ────────────────────────────────────────────────────────────
remove_netem() {
    hdr "Rimuovo netem da tutti i worker"

    local ssh_key ssh_user
    ssh_key=$(jq -r '.ssh.key' "$ENV_JSON" | tr -d '\r')
    ssh_key="${ssh_key/#\~/$HOME}"
    ssh_user=$(jq -r '.ssh.user' "$ENV_JSON" | tr -d '\r')

    while IFS=$'\t' read -r name ip iface; do
        if [[ -z "$iface" || "$iface" == "null" ]]; then
            info "$name: skip (no interface)"
            continue
        fi
        step "$name (${ip}): tc qdisc del dev ${iface} root"
        # -n: no stdin (altrimenti ssh consuma lo stdin del while-read e il loop si ferma)
        ssh -n -i "$ssh_key" \
            -o BatchMode=yes \
            -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "${ssh_user}@${ip}" \
            "sudo tc qdisc del dev ${iface} root 2>/dev/null || true"
        ok "$name: netem rimosso (o gia' assente)"
    done < <(
        jq -r '.clusters[] | [.name, .ip, (.netem.interface // "")] | @tsv' "$ENV_JSON" | tr -d '\r'
    )
}

# ── Main ────────────────────────────────────────────────────────────────────
cmd="${1:-apply}"

case "$cmd" in
    apply)
        preflight
        apply_netem
        install_ping_exporter
        verify
        hdr "Tutto fatto"
        info "Lancia ora: python scripts/dmos-ready-check.py"
        ;;
    netem-only)
        preflight
        apply_netem
        ;;
    ping-only)
        preflight
        install_ping_exporter
        ;;
    verify)
        preflight
        verify
        ;;
    remove-netem)
        preflight
        remove_netem
        ;;
    *)
        echo "Usage: $0 {apply|netem-only|ping-only|verify|remove-netem}"
        echo ""
        echo "  apply         (default) tutto: netem + ping_exporter + verifica"
        echo "  netem-only    solo tc netem sui worker"
        echo "  ping-only     solo helm install ping_exporter"
        echo "  verify        no apply, solo verifica stato attuale"
        echo "  remove-netem  rimuove tutte le regole netem (torna a LAN pura)"
        exit 1
        ;;
esac
