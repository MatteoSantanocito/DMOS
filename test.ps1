# File: scripts/uninstall_k3s_all.ps1

$clusters = @(
    @{IP="192.168.1.245"; Name="Cluster1"},
    @{IP="192.168.1.246"; Name="Cluster2"},
    @{IP="192.168.1.247"; Name="Cluster3"}
)

foreach ($cluster in $clusters) {
    Write-Host "🔧 Uninstalling K3s on $($cluster.Name) ($($cluster.IP))..."
    
    ssh ubuntu@$($cluster.IP) @'
        # Stop DMOS se running
        pkill -f dmos_main
        
        # Uninstall K3s
        /usr/local/bin/k3s-uninstall.sh
        
        # Cleanup residual
        sudo rm -rf /var/lib/rancher
        sudo rm -rf /etc/rancher
        sudo rm -rf ~/.kube
        
        # Reboot per cleanup networking
        echo "Rebooting for clean network state..."
        sudo reboot
'@
    
    Write-Host "✅ $($cluster.Name) uninstall initiated (rebooting...)"
}

Write-Host "`n⏳ Waiting 2 minutes for all nodes to reboot..."
Start-Sleep -Seconds 120

# Verify uninstall
foreach ($cluster in $clusters) {
    Write-Host "`nVerifying $($cluster.Name)..."
    ssh ubuntu@$($cluster.IP) "command -v k3s || echo '✅ K3s removed'"
}