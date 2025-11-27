$ErrorActionPreference = 'Stop'

# Change to script directory
Set-Location -Path $PSScriptRoot

while ($true) {
    try {
        & "$PSScriptRoot\run_obstacle_pipeline.bat"
    }
    catch {
        Write-Host "run_obstacle_pipeline.bat failed: $_" -ForegroundColor Red
    }
    Start-Sleep -Seconds 10
}
