<#
.SYNOPSIS
    Setup degli "addon DMOS" su un parco di cluster K3s già esistenti.

.DESCRIPTION
    Script di automazione per la Fase 3 + Fase 4 del setup DMOS dopo che
    i cluster K3s sono stati provisionati con un tool esterno (es.
    unict-cclab/proxmox-k3s). Lo script applica su CIASCUN cluster:

      Fase 3 — helm upgrade Cilium:
        * Hubble L7 httpV2 metrics (per Prometheus discovery)
        * ServiceMonitor con label release=prometheus-stack
        * Cilium Ingress Controller (NodePort 30080)
        * Rollout restart del DaemonSet cilium

      Fase 4 — Deploy workload e manifest DMOS:
        * Online Boutique (Google microservices-demo)
        * I 7 manifest deployments/ DMOS-required:
            00-namespace.yaml
            cnp-frontend-l7.yaml
            cnp-backends-l7.yaml
            ingress-frontend-cilium.yaml
            global-service-frontend.yaml
            global-service-backends.yaml
            hubble-servicemonitor-monitoring.yaml

    Lo script è IDEMPOTENTE: helm upgrade non duplica releases, kubectl
    apply riconcilia lo stato. Puoi rilanciarlo dopo modifiche al config.

    Lo script auto-rileva i kubeconfig nella cartella indicata (default:
    ../../dmos-iac/kubeconfig-ms-cluster-*) per non dover hard-codare N.

.PARAMETER KubeconfigDir
    Directory contenente i file `kubeconfig-ms-cluster-*`.
    Default: $env:USERPROFILE\dmos-iac

.PARAMETER DeploymentsDir
    Directory con i manifest DMOS-required.
    Default: <DMOS_REPO>\deployments

.PARAMETER IngressNodePort
    NodePort per il Cilium Ingress Controller. Default 30080.

.PARAMETER PrometheusReleaseLabel
    Valore della label `release` che il Prometheus Operator del chart
    kube-prometheus-stack usa per scoprire i ServiceMonitor.
    Default: prometheus-stack (= helm release name usato dal tool
    proxmox-k3s). Se hai installato kube-prometheus-stack con un nome
    diverso, sovrascrivi qui.

.PARAMETER PrometheusNamespace
    Namespace di kube-prometheus-stack, dove va il ServiceMonitor di Hubble.
    Default: observability (proxmox-k3s). Kubephos: kubephos-observability.
.PARAMETER KubeconfigFilter
    Pattern dei file kubeconfig in KubeconfigDir. Default:
    kubeconfig-ms-cluster-*. Il nome del cluster e' il nome del file senza
    il prefisso `kubeconfig-`.
.PARAMETER WorkloadNodeSelector
    `chiave=valore`: se impostato, tutte le Deployment di online-boutique
    ricevono questo nodeSelector (per tenere il workload sui nodi app quando il
    cluster ha nodi di servizio senza taint, es. Kubephos:
    kubephos.dev/pool=app). Default: vuoto (nessun nodeSelector).
.PARAMETER IngressMode
    `ingress` (default): Cilium Ingress controller su NodePort IngressNodePort
    (deployments/ingress-frontend-cilium.yaml), come il testbed di riferimento.
    `gateway`: Gateway API gia' attiva nel Cilium del cluster
    (deployments/gateway-frontend.yaml); l'upgrade di Cilium non accende il
    controller Ingress e lo script stampa la NodePort del Gateway di ogni
    cluster, da mettere in config/local.yaml come `ingress_port`.
.PARAMETER GatewayClass
    GatewayClass del Gateway con -IngressMode gateway. Default:
    kubephos-cilium (Kubephos: Service di tipo NodePort).
.PARAMETER SkipCiliumUpgrade
    Salta la Fase 3 (helm upgrade Cilium). Utile se hai già fatto
    l'upgrade manualmente o vuoi solo applicare i manifest.

.PARAMETER SkipOnlineBoutique
    Salta il deploy di Online Boutique. Utile se vuoi solo
    aggiornare i manifest DMOS senza ricaricare il workload.

.PARAMETER SkipDmosManifests
    Salta l'apply dei 7 manifest DMOS. Utile in fase di debug.

.PARAMETER CiliumChartVersion
    Versione del chart Helm di Cilium da usare per l'upgrade.
    Lasciare vuoto = la versione gia' installata su ciascun cluster, letta da
    `helm list` e passata a `--version` (senza `--version` helm prenderebbe
    l'ultimo chart del repo, cioe' un upgrade di Cilium). Default: vuoto.

.EXAMPLE
    .\setup-dmos-on-clusters.ps1
    # Run con tutti i default sui kubeconfig in dmos-iac/

.EXAMPLE
    .\setup-dmos-on-clusters.ps1 -SkipCiliumUpgrade
    # Solo Fase 4 (workload + manifest DMOS), salta Cilium

.EXAMPLE
    .\setup-dmos-on-clusters.ps1 -KubeconfigDir "D:\altrolab\kubeconfigs"
    # Punta a una directory diversa

.NOTES
    Author: Matteo Santanocito + Claude
    Repository: github.com/MatteoSantanocito/dmos
    Documentato in: dmos-iac/SETUP-LOG.md (sezioni 9.1 e 9.2)
#>

param(
    [string] $KubeconfigDir          = (Join-Path $env:USERPROFILE "dmos-iac"),
    [string] $DeploymentsDir         = (Join-Path $PSScriptRoot "..\deployments"),
    [int]    $IngressNodePort        = 30080,
    [string] $PrometheusReleaseLabel = "prometheus-stack",
    [string] $PrometheusNamespace    = "observability",
    [string] $KubeconfigFilter       = "kubeconfig-ms-cluster-*",
    [string] $WorkloadNodeSelector   = "",
    [ValidateSet('ingress', 'gateway')]
    [string] $IngressMode            = "ingress",
    [string] $GatewayClass           = "kubephos-cilium",
    [string] $CiliumChartVersion     = "",
    [switch] $SkipCiliumUpgrade,
    [switch] $SkipOnlineBoutique,
    [switch] $SkipDmosManifests
)

$ErrorActionPreference = "Stop"

# ─── Helpers ─────────────────────────────────────────────────────────────────

function Write-Section($title) {
    Write-Host ""
    Write-Host ("─" * 75)             -ForegroundColor DarkGray
    Write-Host ("  $title")            -ForegroundColor Cyan
    Write-Host ("─" * 75)             -ForegroundColor DarkGray
}

function Write-Step($msg)    { Write-Host "  → $msg"            -ForegroundColor White }
function Write-Ok($msg)      { Write-Host "  ✓ $msg"            -ForegroundColor Green }
function Write-Warn($msg)    { Write-Host "  ! $msg"            -ForegroundColor Yellow }
function Write-Fail($msg)    { Write-Host "  ✗ $msg"            -ForegroundColor Red }

function Assert-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Fail "Tool richiesto non trovato in PATH: $name"
        Write-Host "Installa $name e riprova." -ForegroundColor Yellow
        exit 1
    }
}

# ─── Pre-flight ──────────────────────────────────────────────────────────────

Write-Section "Pre-flight check"

Assert-Tool "helm"
Assert-Tool "kubectl"
Write-Ok "helm e kubectl trovati nel PATH"

if (-not (Test-Path $KubeconfigDir)) {
    Write-Fail "KubeconfigDir non esiste: $KubeconfigDir"
    exit 1
}
Write-Ok "KubeconfigDir trovata: $KubeconfigDir"

if (-not (Test-Path $DeploymentsDir)) {
    Write-Fail "DeploymentsDir non esiste: $DeploymentsDir"
    exit 1
}
$DeploymentsDir = (Resolve-Path $DeploymentsDir).Path
Write-Ok "DeploymentsDir trovata: $DeploymentsDir"

# Auto-discover clusters dai kubeconfig.
# `-Force` include anche i file con attributo Hidden (su Windows il tool
# proxmox-k3s a volte salva i file con flag Hidden — quirk noto di Go su NTFS
# che capita quando si sovrascrivono file appena cancellati).
$clusters = Get-ChildItem -Path $KubeconfigDir -Filter $KubeconfigFilter -File -Force |
    Where-Object { $_.Name -notmatch '\.(OLD|bak|backup)$' } |
    Sort-Object Name |
    ForEach-Object {
        $name = $_.Name -replace '^kubeconfig-', ''
        [PSCustomObject]@{
            Name       = $name
            Kubeconfig = $_.FullName
        }
    }

if ($clusters.Count -eq 0) {
    Write-Fail "Nessun file '$KubeconfigFilter' trovato in $KubeconfigDir"
    exit 1
}

Write-Ok "Trovati $($clusters.Count) cluster:"
foreach ($c in $clusters) {
    Write-Host "    • $($c.Name)  →  $($c.Kubeconfig)" -ForegroundColor DarkGray
}

# Verifica che ogni cluster sia raggiungibile
Write-Step "Test connettività ai cluster..."
$failed = @()
foreach ($c in $clusters) {
    try {
        $null = kubectl --kubeconfig $c.Kubeconfig get nodes -o name 2>$null
        if ($LASTEXITCODE -ne 0) { throw "kubectl ha ritornato exit code $LASTEXITCODE" }
        Write-Host "    ✓ $($c.Name)" -ForegroundColor Green
    } catch {
        Write-Host "    ✗ $($c.Name) — $_" -ForegroundColor Red
        $failed += $c.Name
    }
}
if ($failed.Count -gt 0) {
    Write-Fail "Non riesco a contattare $($failed.Count) cluster: $($failed -join ', ')"
    Write-Warn "Verifica VPN UniCT e che i cluster siano up. Esco."
    exit 1
}

# ─── Fase 3: Helm upgrade Cilium ─────────────────────────────────────────────

if (-not $SkipCiliumUpgrade) {
    Write-Section "Fase 3 — Helm upgrade Cilium su tutti i cluster"

    Write-Step "Aggiungo repo Helm cilium (idempotente)..."
    # --force-update rende `helm repo add` idempotente: se il repo `cilium`
    # esiste gia' (run precedente) lo aggiorna invece di fallire con
    # "repository name (cilium) already exists" — errore che, con
    # $ErrorActionPreference='Stop', interromperebbe l'intero script.
    helm repo add cilium https://helm.cilium.io --force-update 2>&1 | Out-Null
    helm repo update cilium 2>&1 | Out-Null
    Write-Ok "Repo cilium up-to-date"

    # IMPORTANTE: scriviamo i values in un file YAML temporaneo invece di
    # passarli via --set inline. PowerShell + helm --set + virgole con escape
    # (`\,`) hanno un bug noto in cui le `\,` vengono perse e i valori complessi
    # come `hubble.metrics.enabled={httpV2:...labelsContext=A\,B,drop,tcp}`
    # vengono mal-interpretati. Risultato: Cilium parte ma il config Hubble è
    # corrotto e l'agente si pianta (startup probe fallisce su :9879).
    # Il values file YAML evita ogni escape — helm legge il file come-è.
    $valuesYaml = @"
hubble:
  metrics:
    enableOpenMetrics: true
    enabled:
      - 'httpV2:exemplars=true;labelsContext=source_namespace,source_workload,destination_namespace,destination_workload,traffic_direction'
      - dns
      - drop
      - tcp
      - flow
      - icmp
    serviceMonitor:
      enabled: true
      labels:
        release: $PrometheusReleaseLabel
"@
    if ($IngressMode -eq 'ingress') {
        # Con -IngressMode gateway l'ingresso e' un Gateway (Gateway API gia'
        # attiva nel Cilium del cluster): il controller Ingress resta spento.
        $valuesYaml += @"

ingressController:
  enabled: true
  loadbalancerMode: shared
  service:
    type: NodePort
    insecureNodePort: $IngressNodePort
"@
    }

    # Scrivi in un file temporaneo (cleanup automatico in finally)
    $tmpValues = [System.IO.Path]::GetTempFileName()
    # Cambia estensione in .yaml (cosmesi, helm non lo richiede)
    $tmpValuesYaml = [System.IO.Path]::ChangeExtension($tmpValues, '.yaml')
    Move-Item $tmpValues $tmpValuesYaml -Force
    Set-Content -Path $tmpValuesYaml -Value $valuesYaml -Encoding UTF8
    Write-Host ("  ${DIM}values temporanei: $tmpValuesYaml${RESET}") -ForegroundColor DarkGray

    try {
        foreach ($c in $clusters) {
            Write-Section "[$($c.Name)] Cilium upgrade"

            $args = @(
                'upgrade', 'cilium', 'cilium/cilium',
                '-n', 'kube-system',
                '--reuse-values',
                '--kubeconfig', $c.Kubeconfig,
                '-f', $tmpValuesYaml
            )
            $chartVersion = $CiliumChartVersion
            if (-not $chartVersion) {
                # Versione installata: senza --version helm userebbe l'ultimo
                # chart del repo e aggiornerebbe Cilium.
                $rel = helm list -n kube-system --kubeconfig $c.Kubeconfig -f '^cilium$' -o json | ConvertFrom-Json
                if ($LASTEXITCODE -ne 0 -or -not $rel -or $rel[0].chart -notmatch '^cilium-(.+)$') {
                    Write-Fail "Non trovo la release Helm 'cilium' in kube-system su $($c.Name): passa -CiliumChartVersion"
                    exit 1
                }
                $chartVersion = $Matches[1]
            }
            Write-Step "Chart cilium $chartVersion (versione mantenuta)"
            $args += @('--version', $chartVersion)

            Write-Step "helm upgrade (values from $($tmpValuesYaml | Split-Path -Leaf))..."
            & helm @args
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "Helm upgrade fallito su $($c.Name)"
                exit 1
            }
            Write-Ok "Helm upgrade completato"

            Write-Step "Rollout restart cilium DaemonSet (agent)..."
            kubectl --kubeconfig $c.Kubeconfig -n kube-system rollout restart ds cilium
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "Rollout restart DS cilium fallito su $($c.Name)"
                exit 1
            }

            # IMPORTANTE: helm upgrade aggiorna il ConfigMap `cilium-config` ma NON
            # restarta automaticamente il Deployment cilium-operator. Senza questo
            # restart, l'operator continua con la config vecchia → non registra le
            # CRD `cilium*envoyconfigs` (che servono per ingressController.enabled) →
            # cilium-agent (con config nuova) si blocca in "Still waiting for Cilium
            # Operator to register CRDs" → rollout status va in timeout.
            # Forziamo manualmente il restart dell'operator.
            Write-Step "Rollout restart cilium-operator Deployment..."
            kubectl --kubeconfig $c.Kubeconfig -n kube-system rollout restart deployment cilium-operator
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "Rollout restart Deployment cilium-operator fallito su $($c.Name)"
                exit 1
            }

            Write-Step "Attendo che cilium-operator sia pronto..."
            kubectl --kubeconfig $c.Kubeconfig -n kube-system rollout status deployment cilium-operator --timeout=2m
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "Operator non torna ready entro 2 min su $($c.Name)"
                exit 1
            }
            Write-Ok "Cilium operator ready (CRD Envoy registrate)"

            Write-Step "Attendo che cilium DaemonSet sia pronto..."
            kubectl --kubeconfig $c.Kubeconfig -n kube-system rollout status ds cilium --timeout=5m
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "Cilium agent non torna ready entro 5 min su $($c.Name)"
                exit 1
            }
            Write-Ok "Cilium ready su $($c.Name)"
        }
    }
    finally {
        # Cleanup del values file temporaneo anche se lo script esce con errore
        if (Test-Path $tmpValuesYaml) {
            Remove-Item $tmpValuesYaml -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Section "Fase 3 — Skipped (flag -SkipCiliumUpgrade)"
}

# ─── Fase 4a: Deploy Online Boutique ──────────────────────────────────────────

if (-not $SkipOnlineBoutique) {
    Write-Section "Fase 4a — Deploy Online Boutique"

    $obUrl = 'https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml'

    foreach ($c in $clusters) {
        Write-Step "[$($c.Name)] Creazione namespace online-boutique..."
        # `kubectl create ns --dry-run | apply` rende il create idempotente
        kubectl --kubeconfig $c.Kubeconfig create namespace online-boutique --dry-run=client -o yaml |
            kubectl --kubeconfig $c.Kubeconfig apply -f -
        if ($LASTEXITCODE -ne 0) { Write-Fail "create namespace fallito"; exit 1 }

        Write-Step "[$($c.Name)] Apply Online Boutique manifest..."
        kubectl --kubeconfig $c.Kubeconfig -n online-boutique apply -f $obUrl
        if ($LASTEXITCODE -ne 0) { Write-Fail "Apply Online Boutique fallito"; exit 1 }

        if ($WorkloadNodeSelector) {
            if ($WorkloadNodeSelector -notmatch '^([^=\s]+)=([^=\s]*)$') {
                Write-Fail "-WorkloadNodeSelector deve essere chiave=valore, non '$WorkloadNodeSelector'"
                exit 1
            }
            $patch = @{ spec = @{ template = @{ spec = @{ nodeSelector = @{ $Matches[1] = $Matches[2] } } } } } |
                ConvertTo-Json -Depth 6 -Compress
            $patchFile = [System.IO.Path]::GetTempFileName()
            try {
                Set-Content -Path $patchFile -Value $patch -Encoding ASCII
                $deps = kubectl --kubeconfig $c.Kubeconfig -n online-boutique get deployments -o name
                foreach ($d in $deps) {
                    kubectl --kubeconfig $c.Kubeconfig -n online-boutique patch $d --type merge --patch-file $patchFile | Out-Null
                    if ($LASTEXITCODE -ne 0) { Write-Fail "nodeSelector su $d fallito"; exit 1 }
                }
            } finally {
                Remove-Item $patchFile -Force -ErrorAction SilentlyContinue
            }
            Write-Ok "nodeSelector $WorkloadNodeSelector su $(@($deps).Count) Deployment"
        }

        Write-Ok "Online Boutique applicato su $($c.Name)"
    }
} else {
    Write-Section "Fase 4a — Skipped (flag -SkipOnlineBoutique)"
}

# ─── Fase 4b: Apply manifest DMOS-required ───────────────────────────────────

if (-not $SkipDmosManifests) {
    Write-Section "Fase 4b — Apply i 7 manifest DMOS-required"

    $manifests = @(
        '00-namespace.yaml',
        'cnp-frontend-l7.yaml',
        'cnp-backends-l7.yaml',
        $(if ($IngressMode -eq 'gateway') { 'gateway-frontend.yaml' } else { 'ingress-frontend-cilium.yaml' }),
        'global-service-frontend.yaml',
        'global-service-backends.yaml',
        'hubble-servicemonitor-monitoring.yaml'
    )

    # Verifica che tutti i manifest esistano prima di iniziare
    foreach ($m in $manifests) {
        $p = Join-Path $DeploymentsDir $m
        if (-not (Test-Path $p)) {
            Write-Fail "Manifest mancante: $p"
            exit 1
        }
    }
    Write-Ok "Tutti i 7 manifest trovati in $DeploymentsDir"

    foreach ($c in $clusters) {
        Write-Step "[$($c.Name)] Apply dei manifest..."
        foreach ($m in $manifests) {
            $p = Join-Path $DeploymentsDir $m
            if ($m -eq 'hubble-servicemonitor-monitoring.yaml') {
                # Namespace e label del Prometheus di questo setup
                $sm = (Get-Content -Raw $p) `
                    -replace '(?m)^(\s*namespace:\s*)observability\s*$', "`${1}$PrometheusNamespace" `
                    -replace '(?m)^(\s*release:\s*)prometheus-stack\s*$', "`${1}$PrometheusReleaseLabel"
                $sm | kubectl --kubeconfig $c.Kubeconfig apply -f -
            } elseif ($m -eq 'gateway-frontend.yaml') {
                # GatewayClass di questo setup
                (Get-Content -Raw $p) `
                    -replace '(?m)^(\s*gatewayClassName:\s*)\S+\s*$', "`${1}$GatewayClass" |
                    kubectl --kubeconfig $c.Kubeconfig apply -f -
            } else {
                kubectl --kubeconfig $c.Kubeconfig apply -f $p
            }
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "Apply $m fallito su $($c.Name)"
                exit 1
            }
        }
        Write-Ok "Tutti i manifest applicati su $($c.Name)"
    }

    if ($IngressMode -eq 'gateway') {
        # La NodePort del Service del Gateway la assegna Kubernetes: va scritta
        # in config/local.yaml (ingress_port) prima di render-runtime-files.sh.
        Write-Section "Gateway: NodePort da mettere in config/local.yaml"
        foreach ($c in $clusters) {
            kubectl --kubeconfig $c.Kubeconfig -n online-boutique wait --for=condition=Programmed `
                gateway/frontend-gateway --timeout=180s | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "[$($c.Name)] Gateway frontend-gateway non Programmed entro 180 s"
                kubectl --kubeconfig $c.Kubeconfig -n online-boutique describe gateway frontend-gateway
                exit 1
            }
            $np = kubectl --kubeconfig $c.Kubeconfig -n online-boutique get svc cilium-gateway-frontend-gateway `
                -o jsonpath='{.spec.ports[0].nodePort}'
            if ($LASTEXITCODE -ne 0 -or -not $np) {
                Write-Fail "[$($c.Name)] Service cilium-gateway-frontend-gateway senza NodePort"
                exit 1
            }
            Write-Host ("    {0,-24} ingress_port: {1}" -f $c.Name, $np) -ForegroundColor Green
        }
        Write-Warn "Copia ogni ingress_port nel cluster corrispondente di config/local.yaml, poi: bash scripts/render-runtime-files.sh"
    }
} else {
    Write-Section "Fase 4b — Skipped (flag -SkipDmosManifests)"
}

# ─── Verifica finale ─────────────────────────────────────────────────────────

Write-Section "Verifica finale"

foreach ($c in $clusters) {
    Write-Host ""
    Write-Host "  [$($c.Name)]" -ForegroundColor Cyan

    # Hubble metrics service
    $hubbleSvc = kubectl --kubeconfig $c.Kubeconfig -n kube-system get svc hubble-metrics -o name 2>$null
    if ($hubbleSvc) {
        Write-Host "    ✓ Service hubble-metrics presente"             -ForegroundColor Green
    } else {
        Write-Host "    ⚠ Service hubble-metrics MANCA"                -ForegroundColor Yellow
    }

    # ServiceMonitor Hubble
    $sm = kubectl --kubeconfig $c.Kubeconfig -n $PrometheusNamespace get servicemonitor hubble-metrics -o name 2>$null
    if ($sm) {
        Write-Host "    ✓ ServiceMonitor hubble-metrics in $PrometheusNamespace" -ForegroundColor Green
    } else {
        Write-Host "    ⚠ ServiceMonitor mancante in $PrometheusNamespace"    -ForegroundColor Yellow
    }

    if ($IngressMode -eq 'gateway') {
        # Gateway API
        $gwOk = kubectl --kubeconfig $c.Kubeconfig -n online-boutique get gateway frontend-gateway `
            -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}' 2>$null
        if ($gwOk -eq 'True') {
            Write-Host "    ✓ Gateway frontend-gateway Programmed"       -ForegroundColor Green
        } else {
            Write-Host "    ⚠ Gateway frontend-gateway non Programmed"   -ForegroundColor Yellow
        }
    } else {
        # Cilium Ingress class
        $ingClass = kubectl --kubeconfig $c.Kubeconfig get ingressclass cilium -o name 2>$null
        if ($ingClass) {
            Write-Host "    ✓ IngressClass cilium presente"                -ForegroundColor Green
        } else {
            Write-Host "    ⚠ IngressClass cilium mancante"                -ForegroundColor Yellow
        }
    }

    # Online Boutique pods
    $obPods = (kubectl --kubeconfig $c.Kubeconfig -n online-boutique get pods --no-headers 2>$null |
        Measure-Object).Count
    Write-Host "    ℹ Pod in online-boutique: $obPods (attesi: 11-12)" -ForegroundColor DarkGray

    # CNP L7
    $cnpCount = (kubectl --kubeconfig $c.Kubeconfig -n online-boutique get ciliumnetworkpolicy --no-headers 2>$null |
        Measure-Object).Count
    if ($cnpCount -ge 2) {
        Write-Host "    ✓ CiliumNetworkPolicy L7 applicate ($cnpCount)" -ForegroundColor Green
    } else {
        Write-Host "    ⚠ CiliumNetworkPolicy attese ≥2, trovate $cnpCount" -ForegroundColor Yellow
    }
}

Write-Section "Tutto fatto"
Write-Host ""
Write-Host "  Prossimi step manuali (NON automatizzati da questo script):" -ForegroundColor Cyan
Write-Host "    1. (opzionale) Install ping_exporter su ogni cluster"        -ForegroundColor White
Write-Host "    2. tc netem via SSH ai worker (delay inter-cluster simulato)" -ForegroundColor White
Write-Host "    3. Aggiornare config/local.yaml di DMOS con i 6 cluster"     -ForegroundColor White
Write-Host "    4. python -m src.dmos_main"                                  -ForegroundColor White
Write-Host ""
Write-Host "  Documentazione completa: dmos-iac\SETUP-LOG.md"               -ForegroundColor DarkGray
Write-Host ""
