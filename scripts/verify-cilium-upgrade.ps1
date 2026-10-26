<#
.SYNOPSIS
    Verifica che l'helm upgrade di Cilium (Fase 3 di setup-dmos-on-clusters.ps1)
    sia andato a buon fine su tutti i cluster, PRIMA di procedere con
    setup-netem-and-ping.sh.

.DESCRIPTION
    Per ciascun cluster auto-rilevato in KubeconfigDir, controlla:
      1. Release Helm 'cilium' in stato deployed (+ revision)
      2. DaemonSet cilium (agent) tutti i pod Ready
      3. Deployment cilium-operator Ready
      4. DaemonSet cilium-envoy Ready
      5. cilium-dbg status --brief == OK
      6. CRD Envoy registrate (ciliumenvoyconfigs.cilium.io)
      7. Service hubble-metrics in kube-system
      8. ServiceMonitor hubble-metrics in observability
      9. IngressClass 'cilium' presente
     10. ClusterMesh: N-1 cluster remoti connessi

    Lo script e' SOLO read-only: non modifica nulla. Exit code 0 se tutto OK
    su tutti i cluster, 1 se almeno un check fallisce.

    File ASCII puro (niente caratteri Unicode) -> nessun problema di encoding
    su PowerShell 5.1 / 7, nessun BOM richiesto.

.PARAMETER KubeconfigDir
    Directory con i file kubeconfig-ms-cluster-*. Default $env:USERPROFILE\dmos-iac

.EXAMPLE
    .\scripts\verify-cilium-upgrade.ps1
#>

param(
    [string] $KubeconfigDir = (Join-Path $env:USERPROFILE "dmos-iac")
)

# NON usare Stop: vogliamo controllare TUTTI i cluster anche se uno fallisce.
$ErrorActionPreference = "Continue"

# ---- Helpers (ASCII-only) --------------------------------------------------
function Hdr($t)  { Write-Host ""; Write-Host ("=== " + $t + " ===") -ForegroundColor Cyan }
function Ok($m)   { Write-Host ("  [OK]   " + $m) -ForegroundColor Green }
function Warn($m) { Write-Host ("  [WARN] " + $m) -ForegroundColor Yellow }
function Bad($m)  { Write-Host ("  [FAIL] " + $m) -ForegroundColor Red }
function Info($m) { Write-Host ("  " + $m) -ForegroundColor DarkGray }

# ---- Pre-flight ------------------------------------------------------------
if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    Bad "kubectl non trovato nel PATH"; exit 2
}
if (-not (Get-Command helm -ErrorAction SilentlyContinue)) {
    Bad "helm non trovato nel PATH"; exit 2
}
if (-not (Test-Path $KubeconfigDir)) {
    Bad ("KubeconfigDir non esiste: " + $KubeconfigDir); exit 2
}

$clusters = Get-ChildItem -Path $KubeconfigDir -Filter "kubeconfig-ms-cluster-*" -File -Force |
    Where-Object { $_.Name -notmatch '\.(OLD|bak|backup|disabled)$' } |
    Sort-Object Name |
    ForEach-Object {
        [PSCustomObject]@{
            Name       = ($_.Name -replace '^kubeconfig-', '')
            Kubeconfig = $_.FullName
        }
    }

if ($clusters.Count -eq 0) {
    Bad ("Nessun kubeconfig-ms-cluster-* in " + $KubeconfigDir); exit 2
}

Hdr ("Verifica Cilium upgrade - " + $clusters.Count + " cluster")
$expectedPeers = $clusters.Count - 1

$totalFail = 0

foreach ($c in $clusters) {
    Hdr $c.Name
    $kc = $c.Kubeconfig

    # --- 1. Helm release cilium ---
    $relJson = helm --kubeconfig $kc -n kube-system list --filter '^cilium$' -o json 2>$null
    $rel = $null
    if ($relJson) { try { $rel = $relJson | ConvertFrom-Json } catch { $rel = $null } }
    if ($rel -and $rel.Count -ge 1 -and $rel[0].status -eq 'deployed') {
        Ok ("Helm release cilium: deployed (revision " + $rel[0].revision + ", chart " + $rel[0].chart + ")")
    } else {
        Bad "Helm release cilium NON in stato 'deployed'"
        $totalFail++
    }

    # --- 2. DaemonSet cilium (agent) ---
    $dsCilium = kubectl --kubeconfig $kc -n kube-system get ds cilium `
        -o jsonpath='{.status.numberReady}/{.status.desiredNumberScheduled}' 2>$null
    if ($dsCilium -match '^(\d+)/(\d+)$' -and $Matches[1] -eq $Matches[2] -and [int]$Matches[1] -gt 0) {
        Ok ("DaemonSet cilium (agent): " + $dsCilium + " Ready")
    } else {
        Bad ("DaemonSet cilium (agent): " + $dsCilium + " - NON tutti Ready")
        $totalFail++
    }

    # --- 3. Deployment cilium-operator ---
    $depOp = kubectl --kubeconfig $kc -n kube-system get deploy cilium-operator `
        -o jsonpath='{.status.readyReplicas}/{.spec.replicas}' 2>$null
    if ($depOp -match '^(\d+)/(\d+)$' -and $Matches[1] -eq $Matches[2] -and [int]$Matches[1] -gt 0) {
        Ok ("Deployment cilium-operator: " + $depOp + " Ready")
    } else {
        Bad ("Deployment cilium-operator: " + $depOp + " - NON Ready")
        $totalFail++
    }

    # --- 4. DaemonSet cilium-envoy ---
    $dsEnvoy = kubectl --kubeconfig $kc -n kube-system get ds cilium-envoy `
        -o jsonpath='{.status.numberReady}/{.status.desiredNumberScheduled}' 2>$null
    if ($dsEnvoy -match '^(\d+)/(\d+)$' -and $Matches[1] -eq $Matches[2] -and [int]$Matches[1] -gt 0) {
        Ok ("DaemonSet cilium-envoy: " + $dsEnvoy + " Ready")
    } else {
        Bad ("DaemonSet cilium-envoy: " + $dsEnvoy + " - NON Ready (serve per l'Ingress)")
        $totalFail++
    }

    # --- 5. cilium-dbg status --brief ---
    $st = kubectl --kubeconfig $kc -n kube-system exec ds/cilium -c cilium-agent -- `
        cilium-dbg status --brief 2>$null
    if ($st -and ($st -join '') -match 'OK') {
        Ok "cilium-dbg status: OK"
    } else {
        Bad ("cilium-dbg status: '" + ($st -join ' ') + "' - non OK")
        $totalFail++
    }

    # --- 6. CRD Envoy (servono per ingressController) ---
    $crd = kubectl --kubeconfig $kc get crd ciliumenvoyconfigs.cilium.io `
        -o name 2>$null
    if ($crd -match 'ciliumenvoyconfigs') {
        Ok "CRD ciliumenvoyconfigs.cilium.io registrata"
    } else {
        Bad "CRD ciliumenvoyconfigs.cilium.io MANCANTE (operator non ha registrato le CRD)"
        $totalFail++
    }

    # --- 7. Service hubble-metrics ---
    $hm = kubectl --kubeconfig $kc -n kube-system get svc hubble-metrics `
        -o name 2>$null
    if ($hm -match 'hubble-metrics') {
        Ok "Service hubble-metrics presente (kube-system)"
    } else {
        Bad "Service hubble-metrics MANCANTE - Hubble L7 metrics non esposte"
        $totalFail++
    }

    # --- 8. ServiceMonitor hubble-metrics in observability ---
    $sm = kubectl --kubeconfig $kc -n observability get servicemonitor hubble-metrics `
        -o name 2>$null
    if ($sm -match 'hubble-metrics') {
        Ok "ServiceMonitor hubble-metrics presente (observability)"
    } else {
        Warn "ServiceMonitor hubble-metrics non trovato in observability (lo applica Fase 4b)"
    }

    # --- 9. IngressClass cilium ---
    $ic = kubectl --kubeconfig $kc get ingressclass cilium -o name 2>$null
    if ($ic -match 'cilium') {
        Ok "IngressClass cilium presente"
    } else {
        Bad "IngressClass cilium MANCANTE - ingressController non abilitato"
        $totalFail++
    }

    # --- 10. ClusterMesh ---
    $cmLine = kubectl --kubeconfig $kc -n kube-system exec ds/cilium -c cilium-agent -- `
        cilium-dbg status 2>$null | Select-String -Pattern 'ClusterMesh'
    if ($cmLine) {
        $cmText = ($cmLine | Select-Object -First 1).ToString().Trim()
        # Tipico: "ClusterMesh:   5/5 clusters ready, 30 global-services"
        if ($cmText -match '(\d+)/(\d+)\s+clusters ready') {
            $ready = [int]$Matches[1]; $tot = [int]$Matches[2]
            if ($ready -ge $expectedPeers -and $ready -eq $tot) {
                Ok ("ClusterMesh: " + $ready + "/" + $tot + " cluster remoti connessi")
            } else {
                Bad ("ClusterMesh: " + $ready + "/" + $tot + " - attesi " + $expectedPeers + " peer")
                $totalFail++
            }
        } else {
            Info ("ClusterMesh: " + $cmText)
        }
    } else {
        Warn "ClusterMesh status non leggibile (riprova tra qualche secondo)"
    }
}

# ---- Riepilogo -------------------------------------------------------------
Hdr "Riepilogo"
if ($totalFail -eq 0) {
    Ok ("Tutti i " + $clusters.Count + " cluster hanno Cilium upgrade OK.")
    Write-Host ""
    Write-Host "  Puoi procedere con:  bash scripts/setup-netem-and-ping.sh" -ForegroundColor Green
    exit 0
} else {
    Bad ($totalFail.ToString() + " check falliti. NON procedere con netem finche' non risolvi.")
    Write-Host ""
    Write-Host "  Troubleshooting: dmos-iac/DMOS-DEPLOYMENT-PIPELINE.md sezione 10" -ForegroundColor Yellow
    Write-Host "  - CRD Envoy mancanti / cilium-envoy non Ready: 10.1 (restart operator)" -ForegroundColor DarkGray
    Write-Host "  - IngressClass mancante: 10.1 / 10.12" -ForegroundColor DarkGray
    Write-Host "  - Per ri-applicare solo Fase 3: .\scripts\setup-dmos-on-clusters.ps1 -SkipOnlineBoutique -SkipDmosManifests" -ForegroundColor DarkGray
    exit 1
}
