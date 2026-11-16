#!/usr/bin/env bash
# =============================================================================
# fleet.sh — elenco dei cluster per gli script del banco (source, non eseguire)
# =============================================================================
# fleet_load riempie tre array paralleli:
#   FLEET_NAMES  nome del cluster (es. ms-cluster-01, test-cluster-1)
#   FLEET_KCS    kubeconfig del cluster
#   FLEET_IPS    IP dei NodePort (vuoto nel caso legacy)
#
# Fonte, in ordine:
#   1. DMOS_KUBECONFIG_DIR impostata -> layout storico: ms-cluster-01..06 con
#      <dir>/kubeconfig-ms-cluster-NN (comportamento di prima di questo file);
#   2. .cache/test-env.json (scritto da scripts/render-runtime-files.sh da
#      config/local.yaml): nome, kubeconfig_path e ip di ogni cluster;
#   3. altrimenti il layout storico in $HOME/Desktop/dmos-iac.
# =============================================================================

fleet_load() {
  local repo_root env_json dir n name kc ip
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  env_json="${repo_root}/.cache/test-env.json"
  FLEET_NAMES=(); FLEET_KCS=(); FLEET_IPS=()

  if [[ -z "${DMOS_KUBECONFIG_DIR:-}" && -f "$env_json" ]] && command -v jq >/dev/null 2>&1; then
    while IFS=$'\t' read -r name kc ip; do
      [[ -n "$name" ]] || continue
      kc="${kc/#\~/$HOME}"
      FLEET_NAMES+=("$name"); FLEET_KCS+=("$kc"); FLEET_IPS+=("$ip")
    done < <(jq -r '.clusters[] | [.name, .kubeconfig_path, (.ip // "")] | @tsv' "$env_json" | tr -d '\r')
  fi

  if [[ ${#FLEET_NAMES[@]} -eq 0 ]]; then
    dir="${DMOS_KUBECONFIG_DIR:-$HOME/Desktop/dmos-iac}"
    for n in 01 02 03 04 05 06; do
      FLEET_NAMES+=("ms-cluster-${n}"); FLEET_KCS+=("${dir}/kubeconfig-ms-cluster-${n}"); FLEET_IPS+=("")
    done
  fi
}
