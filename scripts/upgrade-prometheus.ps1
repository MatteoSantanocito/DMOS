# Upgrade Prometheus su tutti e 3 i cluster
# Applica: scrape_interval=15s per Hubble → rate()[1m] stabile in DMOS
#
# Esegui da: C:\Users\matte\Desktop\Voda
# PS> .\scripts\upgrade-prometheus.ps1

$ErrorActionPreference = "Stop"
$ValuesFile = "config\prometheus-helm-values.yaml"
$Clusters = @("cluster1", "cluster2", "cluster3")

foreach ($c in $Clusters) {
    Write-Host "`n══════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Upgrading Prometheus on $c" -ForegroundColor Cyan
    Write-Host "══════════════════════════════════════" -ForegroundColor Cyan

    helm upgrade prometheus prometheus-community/prometheus `
        -n monitoring `
        -f $ValuesFile `
        --force `
        --kubeconfig ".kube\$c.yaml"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Upgrade failed on $c" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ $c done" -ForegroundColor Green
}

Write-Host "`n══════════════════════════════════════" -ForegroundColor Green
Write-Host "  Attendi ~30s poi verifica campioni" -ForegroundColor Green
Write-Host "══════════════════════════════════════" -ForegroundColor Green

Start-Sleep -Seconds 35

Write-Host "`n── Verifica campioni Hubble (attesi ~8 in 2min a 15s) ──" -ForegroundColor Yellow

$IPs = @{ "cluster1" = "192.168.1.245"; "cluster2" = "192.168.1.246"; "cluster3" = "192.168.1.247" }
$End   = [int][double]::Parse((Get-Date -UFormat %s))
$Start = $End - 120

foreach ($c in $Clusters) {
    $ip = $IPs[$c]
    $url = "http://${ip}:30090/api/v1/query_range?query=hubble_http_requests_total&start=${Start}&end=${End}&step=15"
    try {
        $resp = Invoke-RestMethod -Uri $url -TimeoutSec 5
        $samples = ($resp.data.result[0].values).Count
        if ($samples -ge 6) {
            Write-Host "  ✅ $c — $samples campioni in 2min (scrape 15s OK)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  $c — solo $samples campioni (attesi ~8, potrebbe servire altro tempo)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ℹ️  $c — nessun campione Hubble ancora (normale se non c'è traffico attivo)" -ForegroundColor Gray
    }
}
