$ErrorActionPreference = 'Stop'

# Change to script directory
Set-Location -Path $PSScriptRoot

# Start detector in watch mode (runs until this script stops)
$watcher = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$PSScriptRoot\run_obstacle_pipeline.bat`" --watch" -PassThru -WindowStyle Minimized

try {
    while ($true) {
        try {
            # Only sync; new images will be processed immediately by watcher
            & "$PSScriptRoot\sync_vive_connect.bat"
        }
        catch {
            Write-Host "sync_vive_connect.bat failed: $_" -ForegroundColor Red
        }
        Start-Sleep -Seconds 5
    }
}
finally {
    if ($watcher -and -not $watcher.HasExited) {
        $watcher.CloseMainWindow() | Out-Null
        Start-Sleep -Seconds 2
        if (-not $watcher.HasExited) {
            $watcher.Kill()
        }
    }
}
