#!/bin/bash
# =============================================================================
# setup_netem.sh — Simulazione geografica con tc netem
# =============================================================================
#
# SCOPO:
#   Inietta artificialmente delay di rete sulle interfacce dei nodi dei cluster
#   per simulare distanze geografiche realistiche tra le regioni europee.
#   Questo è necessario perché i 3 cluster sono fisicamente sulla stessa LAN
#   (192.168.1.x), ma simulano deployment in Frankfurt (DE), Paris (FR) e
#   Warsaw (PL) per la tesi.
#
# APPROCCIO (come Romano 2025):
#   tc netem agisce sull'interfaccia di OUTPUT del nodo.
#   Il delay viene applicato ai pacchetti in uscita → impatta la RTT percepita
#   dagli altri cluster quando il nodo risponde.
#
#   cluster2 (Paris/FR, 192.168.1.246): 150ms ±20ms
#     → RTT Frankfurt↔Paris reale: ~100-200ms (valore medio: 150ms)
#   cluster3 (Warsaw/PL, 192.168.1.247): 350ms ±20ms
#     → RTT Frankfurt↔Warsaw reale: ~300-400ms (valore medio: 350ms)
#   cluster1 (Frankfurt/DE, 192.168.1.245): nessun delay
#     → cluster1 è il cluster di riferimento (gateway utenti DE)
#
# RTT ATTESE con ping_exporter dopo setup:
#   cluster1 → cluster2 ping: ≈ 150ms  (delay outgoing di cluster2)
#   cluster1 → cluster3 ping: ≈ 350ms  (delay outgoing di cluster3)
#   cluster2 → cluster3 ping: ≈ 350+150 = 500ms (entrambi i delay)
#
# RTT MEDIE per Φ_net (avg verso tutti i peer):
#   cluster1: avg(150, 350) = 250ms → Φ_net = exp(-2×250/500) ≈ 0.368
#   cluster2: avg(150, 500) = 325ms → Φ_net = exp(-2×325/500) ≈ 0.272
#   cluster3: avg(350, 500) = 425ms → Φ_net = exp(-2×425/500) ≈ 0.183
#
# PREREQUISITI:
#   - SSH passwordless verso i nodi (o eseguire i comandi manualmente)
#   - iproute2 installato sui nodi (tc è incluso di default in k3s/Ubuntu)
#   - L'interfaccia si chiama 'ens18' (verificare con: ip link show)
#
# USO:
#   bash scripts/setup_netem.sh apply    # Applica i delay
#   bash scripts/setup_netem.sh remove   # Rimuove i delay (torna a LAN pura)
#   bash scripts/setup_netem.sh status   # Mostra configurazione attuale
#   bash scripts/setup_netem.sh verify   # Verifica con ping da ogni nodo
#
# ESECUZIONE REMOTA (se SSH configurato):
#   bash scripts/setup_netem.sh apply-remote
# =============================================================================

set -euo pipefail

# ── Configurazione ────────────────────────────────────────────────────────────
CLUSTER1_IP="192.168.1.245"   # Frankfurt (DE) — nessun delay, cluster di riferimento
CLUSTER2_IP="192.168.1.246"   # Paris (FR)     — delay 150ms ±20ms
CLUSTER3_IP="192.168.1.247"   # Warsaw (PL)    — delay 350ms ±20ms

CLUSTER2_DELAY="150ms"
CLUSTER2_JITTER="20ms"
CLUSTER3_DELAY="350ms"
CLUSTER3_JITTER="20ms"

INTERFACE="ens18"              # Interfaccia di rete principale dei nodi
                               # Verificare con: ip link show

SSH_USER="ubuntu"              # Utente SSH per i nodi

# ── Colori per output ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[ERR]${NC}   $*"; }

# ── Funzioni per tc netem ─────────────────────────────────────────────────────

apply_netem() {
    local iface="$1"
    local delay="$2"
    local jitter="$3"

    # Rimuove configurazione esistente (ignora errore se non esiste)
    tc qdisc del dev "$iface" root 2>/dev/null || true

    # Applica delay con distribuzione normale (jitter) per simulare variabilità reale
    # distribution normal: i delay seguono una distribuzione gaussiana centrata su $delay
    tc qdisc add dev "$iface" root netem \
        delay "$delay" "$jitter" distribution normal

    echo "  tc netem: $iface → delay $delay ±$jitter (distribution normal)"
}

remove_netem() {
    local iface="$1"
    tc qdisc del dev "$iface" root 2>/dev/null || true
    echo "  tc netem rimosso da $iface"
}

show_netem() {
    local iface="$1"
    echo "  $iface: $(tc qdisc show dev "$iface" | grep -v 'qdisc noqueue' || echo 'nessuna configurazione')"
}

# ── Comandi principali ────────────────────────────────────────────────────────

cmd_apply_local() {
    # Da eseguire DENTRO il nodo target (cluster2 o cluster3)
    local node="$1"

    case "$node" in
        cluster2)
            log_info "Applicando delay $CLUSTER2_DELAY ±$CLUSTER2_JITTER su $INTERFACE (cluster2 / Paris)"
            apply_netem "$INTERFACE" "$CLUSTER2_DELAY" "$CLUSTER2_JITTER"
            log_ok "cluster2: delay $CLUSTER2_DELAY applicato"
            ;;
        cluster3)
            log_info "Applicando delay $CLUSTER3_DELAY ±$CLUSTER3_JITTER su $INTERFACE (cluster3 / Warsaw)"
            apply_netem "$INTERFACE" "$CLUSTER3_DELAY" "$CLUSTER3_JITTER"
            log_ok "cluster3: delay $CLUSTER3_DELAY applicato"
            ;;
        cluster1)
            log_warn "cluster1 (Frankfurt) è il riferimento — nessun delay applicato"
            ;;
        *)
            log_err "Nodo sconosciuto: $node. Usa: cluster1, cluster2, cluster3"
            exit 1
            ;;
    esac
}

cmd_remove_local() {
    log_info "Rimuovendo delay da $INTERFACE"
    remove_netem "$INTERFACE"
    log_ok "Delay rimosso — tornato a LAN pura"
}

cmd_status_local() {
    log_info "Configurazione tc su $INTERFACE:"
    show_netem "$INTERFACE"
}

cmd_apply_remote() {
    log_info "Applicando delay geografico via SSH..."
    echo ""

    # cluster1: nessun delay (Frankfurt = riferimento)
    log_info "cluster1 ($CLUSTER1_IP, Frankfurt/DE): nessun delay"

    # cluster2: 150ms ±20ms (Paris)
    log_info "cluster2 ($CLUSTER2_IP, Paris/FR): delay $CLUSTER2_DELAY ±$CLUSTER2_JITTER"
    ssh "$SSH_USER@$CLUSTER2_IP" "
        tc qdisc del dev $INTERFACE root 2>/dev/null || true
        tc qdisc add dev $INTERFACE root netem delay $CLUSTER2_DELAY $CLUSTER2_JITTER distribution normal
        echo OK
    "
    log_ok "cluster2: delay $CLUSTER2_DELAY applicato"

    # cluster3: 350ms ±20ms (Warsaw)
    log_info "cluster3 ($CLUSTER3_IP, Warsaw/PL): delay $CLUSTER3_DELAY ±$CLUSTER3_JITTER"
    ssh "$SSH_USER@$CLUSTER3_IP" "
        tc qdisc del dev $INTERFACE root 2>/dev/null || true
        tc qdisc add dev $INTERFACE root netem delay $CLUSTER3_DELAY $CLUSTER3_JITTER distribution normal
        echo OK
    "
    log_ok "cluster3: delay $CLUSTER3_DELAY applicato"

    echo ""
    log_ok "Setup completo. Verifica con: bash scripts/setup_netem.sh verify-remote"
}

cmd_remove_remote() {
    log_info "Rimuovendo delay da tutti i nodi via SSH..."

    for ip in "$CLUSTER2_IP" "$CLUSTER3_IP"; do
        log_info "Rimuovendo da $ip..."
        ssh "$SSH_USER@$ip" "tc qdisc del dev $INTERFACE root 2>/dev/null || true; echo OK"
        log_ok "$ip: delay rimosso"
    done

    log_ok "Tutti i delay rimossi — LAN pura"
}

cmd_status_remote() {
    log_info "Stato tc netem su tutti i nodi:"
    echo ""
    for name_ip in "cluster1:$CLUSTER1_IP" "cluster2:$CLUSTER2_IP" "cluster3:$CLUSTER3_IP"; do
        name="${name_ip%%:*}"
        ip="${name_ip##*:}"
        echo -n "  $name ($ip): "
        ssh "$SSH_USER@$ip" "tc qdisc show dev $INTERFACE | head -2" 2>/dev/null || echo "SSH non disponibile"
    done
}

cmd_verify_remote() {
    log_info "Verifica RTT con ping (10 pacchetti per target):"
    echo ""

    echo -e "  ${YELLOW}Da cluster1 (Frankfurt):${NC}"
    echo -n "    → cluster2 ($CLUSTER2_IP, Paris):  "
    ping -c 5 -q "$CLUSTER2_IP" 2>/dev/null | grep "rtt" || echo "ping fallito"
    echo -n "    → cluster3 ($CLUSTER3_IP, Warsaw): "
    ping -c 5 -q "$CLUSTER3_IP" 2>/dev/null | grep "rtt" || echo "ping fallito"

    echo ""
    log_info "RTT attese:"
    echo "    cluster1→cluster2: ~$CLUSTER2_DELAY (Paris)"
    echo "    cluster1→cluster3: ~$CLUSTER3_DELAY (Warsaw)"
    echo ""
    log_info "Se le RTT non corrispondono, verificare:"
    echo "    1. tc netem applicato sul nodo corretto (cluster2/cluster3)"
    echo "    2. Interfaccia corretta: ip link show (deve essere $INTERFACE)"
    echo "    3. Firewall non blocca ICMP"
}

# ── Comandi kubectl per applicare netem tramite pod privilegiato ──────────────
#
# ALTERNATIVA SSH: se SSH non è configurato, usa un pod privilegiato
# con nsenter per accedere al network namespace dell'host.
#
# Applicare netem su cluster2 (senza SSH):
#   kubectl run netem-cluster2 --rm -it --image=ubuntu \
#     --overrides='{"spec":{"hostNetwork":true,"hostPID":true,"nodeName":"ms02"}}' \
#     --restart=Never --context=cluster2 -- bash -c \
#     "tc qdisc del dev ens18 root 2>/dev/null || true; \
#      tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal; \
#      echo 'Done'"
#
# Applicare netem su cluster3 (senza SSH):
#   kubectl run netem-cluster3 --rm -it --image=ubuntu \
#     --overrides='{"spec":{"hostNetwork":true,"hostPID":true,"nodeName":"ms03"}}' \
#     --restart=Never --context=cluster3 -- bash -c \
#     "tc qdisc del dev ens18 root 2>/dev/null || true; \
#      tc qdisc add dev ens18 root netem delay 350ms 20ms distribution normal; \
#      echo 'Done'"

# ── Comandi PowerShell equivalenti (per Windows) ──────────────────────────────
#
# Poiché l'ambiente di sviluppo è Windows/PowerShell, i comandi SSH
# equivalenti (da eseguire in PowerShell sul laptop):
#
# # Applica delay su cluster2 (Paris, 150ms):
# ssh ubuntu@192.168.1.246 "tc qdisc del dev ens18 root 2>/dev/null; tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal && echo OK"
#
# # Applica delay su cluster3 (Warsaw, 350ms):
# ssh ubuntu@192.168.1.247 "tc qdisc del dev ens18 root 2>/dev/null; tc qdisc add dev ens18 root netem delay 350ms 20ms distribution normal && echo OK"
#
# # Verifica su cluster2:
# ssh ubuntu@192.168.1.246 "tc qdisc show dev ens18"
#
# # Verifica ping da laptop (richiede rotta verso i nodi):
# ping -n 5 192.168.1.246

# ── Main ──────────────────────────────────────────────────────────────────────

ACTION="${1:-help}"

case "$ACTION" in
    apply)
        # Da eseguire dentro il nodo: bash setup_netem.sh apply cluster2
        NODE="${2:-}"
        if [[ -z "$NODE" ]]; then
            log_err "Specifica il nodo: apply cluster1|cluster2|cluster3"
            exit 1
        fi
        cmd_apply_local "$NODE"
        ;;
    remove)
        cmd_remove_local
        ;;
    status)
        cmd_status_local
        ;;
    apply-remote)
        cmd_apply_remote
        ;;
    remove-remote)
        cmd_remove_remote
        ;;
    status-remote)
        cmd_status_remote
        ;;
    verify-remote|verify)
        cmd_verify_remote
        ;;
    help|*)
        echo ""
        echo "  setup_netem.sh — Simulazione geografica con tc netem"
        echo ""
        echo "  USO:"
        echo "    bash scripts/setup_netem.sh <command> [args]"
        echo ""
        echo "  COMANDI (locale — da eseguire dentro il nodo):"
        echo "    apply <cluster2|cluster3>  Applica delay sul nodo corrente"
        echo "    remove                     Rimuove delay dal nodo corrente"
        echo "    status                     Mostra tc qdisc attuale"
        echo ""
        echo "  COMANDI (remoto — richiede SSH configurato):"
        echo "    apply-remote               Applica delay su cluster2 e cluster3 via SSH"
        echo "    remove-remote              Rimuove delay da tutti i nodi via SSH"
        echo "    status-remote              Mostra stato tc su tutti i nodi"
        echo "    verify-remote              Verifica RTT con ping"
        echo ""
        echo "  CONFIGURAZIONE ATTUALE:"
        echo "    cluster1 ($CLUSTER1_IP) Frankfurt/DE: nessun delay (riferimento)"
        echo "    cluster2 ($CLUSTER2_IP) Paris/FR:     $CLUSTER2_DELAY ±$CLUSTER2_JITTER"
        echo "    cluster3 ($CLUSTER3_IP) Warsaw/PL:    $CLUSTER3_DELAY ±$CLUSTER3_JITTER"
        echo ""
        echo "  COMANDI POWERSHELL (da Windows, senza SSH key):"
        echo "    ssh ubuntu@$CLUSTER2_IP \"tc qdisc del dev $INTERFACE root 2>/dev/null; tc qdisc add dev $INTERFACE root netem delay $CLUSTER2_DELAY $CLUSTER2_JITTER distribution normal\""
        echo "    ssh ubuntu@$CLUSTER3_IP \"tc qdisc del dev $INTERFACE root 2>/dev/null; tc qdisc add dev $INTERFACE root netem delay $CLUSTER3_DELAY $CLUSTER3_JITTER distribution normal\""
        echo ""
        ;;
esac
