# scripts/reset_test.ps1 — reset ALL online-boutique deployments to replicas=1
# (except the Online Boutique `loadgenerator`, kept at 0)
#
# Azzera lo stato di scaling su ogni cluster fra un run e l'altro: senza questo,
# le repliche scalate da DMOS (frontend E backend co-locati) resterebbero alte
# e falserebbero il run successivo (specie il baseline dmos-off).
#
# Usa i kubeconfig LOCALI del laptop (come preflight_runs.sh e DMOS stesso),
# NON via SSH: robusto alla ri-provisione delle VM (host key nuove + kubeconfig
# di k3s non leggibile sul nodo). Legge i path da .cache/test-env.json.

$envPath = Join-Path $PSScriptRoot "..\.cache\test-env.json"
if (-not (Test-Path $envPath)) {
    Write-Host "ERROR: .cache/test-env.json not found." -ForegroundColor Red
    Write-Host "Run: bash scripts/render-runtime-files.sh" -ForegroundColor Yellow
    exit 1
}

$config = Get-Content $envPath -Raw | ConvertFrom-Json
$ns = "online-boutique"
$failed = 0

Write-Host "Resetting all deployments in '$ns' to 1 replica..." -ForegroundColor Cyan

foreach ($c in $config.clusters) {
    $kc = $c.kubeconfig_path
    if ($kc -like "~*") { $kc = $kc -replace '^~', $HOME }
    Write-Host ("{0} ..." -f $c.name) -ForegroundColor Yellow
    if (-not (Test-Path $kc)) {
        Write-Host ("  [SKIP] kubeconfig non trovato: {0}" -f $kc) -ForegroundColor Red
        $failed++
        continue
    }
    & kubectl --kubeconfig $kc -n $ns scale deployment --all --replicas=1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("  [FAIL] {0}: kubectl exit {1}" -f $c.name, $LASTEXITCODE) -ForegroundColor Red
        $failed++
    }
    # The Online Boutique load generator sends internal traffic to the frontend
    # that is not part of the experiment: keep it at 0 (if it is deployed).
    & kubectl --kubeconfig $kc -n $ns get deployment loadgenerator *> $null
    if ($LASTEXITCODE -eq 0) {
        & kubectl --kubeconfig $kc -n $ns scale deployment loadgenerator --replicas=0 | Out-Null
        Write-Host "  loadgenerator -> 0" -ForegroundColor DarkGray
    }
}

# Attesa esplicita: lo `scale` non blocca, i pod extra (es. currency 5→1,
# frontend 6→1) terminano in background. Aspettiamo che lo scale-down sia
# COMPLETO su ogni cluster, così il run successivo parte da un baseline pulito
# (nessun pod del run precedente ancora vivo). Timeout 120s per cluster.
Write-Host "Attendo il completamento dello scale-down (pod extra terminati)..." -ForegroundColor Cyan
foreach ($c in $config.clusters) {
    $kc = $c.kubeconfig_path
    if ($kc -like "~*") { $kc = $kc -replace '^~', $HOME }
    if (-not (Test-Path $kc)) { continue }
    $deadline = (Get-Date).AddSeconds(120)
    while ($true) {
        $over = & kubectl --kubeconfig $kc -n $ns get deploy `
                    -o jsonpath='{range .items[*]}{.status.replicas}{"\n"}{end}' 2>$null |
                Where-Object { ($_ -as [int]) -gt 1 }
        if (-not $over) { Write-Host ("  [ok] {0} a 1 replica" -f $c.name) -ForegroundColor Green; break }
        if ((Get-Date) -gt $deadline) {
            Write-Host ("  [warn] {0}: timeout 120s, alcuni deploy ancora >1" -f $c.name) -ForegroundColor Yellow
            break
        }
        Start-Sleep -Seconds 3
    }
}

if ($failed -eq 0) {
    Write-Host "Done! Tutti i cluster resettati a 1 replica." -ForegroundColor Green
} else {
    Write-Host ("Completato con {0} errori - controlla sopra." -f $failed) -ForegroundColor Red
    exit 1
}
