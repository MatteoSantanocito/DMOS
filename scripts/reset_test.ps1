# scripts/reset_simple.ps1
Write-Host "Resetting clusters..." -ForegroundColor Cyan

# Cluster 1
Write-Host "Cluster1..." -ForegroundColor Yellow
ssh ubuntu@192.168.1.245 "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml; kubectl scale deployment frontend --replicas=1 -n online-boutique"

# Cluster 2
Write-Host "Cluster2..." -ForegroundColor Yellow
ssh ubuntu@192.168.1.246 "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml; kubectl scale deployment frontend --replicas=1 -n online-boutique"

# Cluster 3
Write-Host "Cluster3..." -ForegroundColor Yellow
ssh ubuntu@192.168.1.247 "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml; kubectl scale deployment frontend --replicas=1 -n online-boutique"

Write-Host "Done!" -ForegroundColor Green