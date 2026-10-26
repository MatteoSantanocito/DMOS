# Upgrade Prometheus su tutti i cluster definiti in config/local.yaml
# Applica: scrape_interval=15s per Hubble -> rate()[1m] stabile in DMOS
#
# Esegui dalla root del repo DMOS (la directory con README.md):
#   PS> .\scripts\upgrade-prometheus.ps1
#
# Cluster IPs e nomi vengono letti da .cache/test-env.json (generato da
# `bash scripts/render-runtime-files.sh` partendo da config/local.yaml).

$ErrorActionPreference = "Stop"
$ValuesFile = "config\prometheus-helm-values.yaml"
$EnvPath    = ".cache\test-env.json"

if (-not (Test-Path $EnvPath)) {
    Write-Host "ERROR: $EnvPath not found." -ForegroundColor Red
    Write-Host "Run: bash scripts/render-runtime-files.sh" -ForegroundColor Yellow
    exit 1
}

$Config = Get-Content $EnvPath -Raw | ConvertFrom-Json
$Clusters = $Config.clusters | ForEach-Object { $_.name }
$IPs = @{}
foreach ($c in $Config.clusters) { $IPs[$c.name] = $c.ip }

foreach ($c in $Clusters) {
    Write-Host "`n══════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Upgrading Prometheus on $c" -ForegroundColor Cyan
    Write-Host "══════════════════════════════════════" -ForegroundColor Cyan

    # NOTA: nel setup nuovo (proxmox-k3s) il chart è kube-prometheus-stack
    # con release name prometheus-stack in namespace observability.
    # Per setup legacy testbed thesis: chart prometheus-community/prometheus,
    # release prometheus, ns monitoring.
    helm upgrade prometheus-stack prometheus-community/kube-prometheus-stack `
        -n observability `
        -f $ValuesFile `
        --force `
        --kubeconfig ".kube\$c.yaml"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Upgrade failed on $c" -ForegroundColor Red
        exit 1
    }
    Write-Host "$c done" -ForegroundColor Green
}

Write-Host "`n══════════════════════════════════════" -ForegroundColor Green
Write-Host "  Attendi ~30s poi verifica campioni" -ForegroundColor Green
Write-Host "══════════════════════════════════════" -ForegroundColor Green

Start-Sleep -Seconds 35

Write-Host "`n-- Verifica campioni Hubble (attesi ~8 in 2min a 15s) --" -ForegroundColor Yellow

$End   = [int][double]::Parse((Get-Date -UFormat %s))
$Start = $End - 120

foreach ($c in $Clusters) {
    $ip = $IPs[$c]
    $url = "http://${ip}:32090/api/v1/query_range?query=hubble_http_requests_total&start=${Start}&end=${End}&step=15"
    try {
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 5
        $samples = ($resp.data.result[0].values).Count
        if ($samples -ge 6) {
            Write-Host "  $c -- $samples campioni in 2min (scrape 15s OK)" -ForegroundColor Green
        } else {
            Write-Host "  $c -- solo $samples campioni (attesi ~8, potrebbe servire altro tempo)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  $c -- nessun campione Hubble ancora (normale se non c'e' traffico attivo)" -ForegroundColor Gray
    }
}
