$archive = Join-Path $PSScriptRoot "..\data\full_2.1.0.7z"
$totalBytes = 116GB

if (Test-Path $archive) {
    $file = Get-Item $archive
    $percent = [math]::Round(($file.Length / $totalBytes) * 100, 2)
    "Archive: $($file.FullName)"
    "Size:    {0:N2} GB / ~116 GB ({1}%)" -f ($file.Length / 1GB), $percent
    "Updated: $($file.LastWriteTime)"
} else {
    "Archive not found yet: $archive"
}

$downloaders = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -like "*aria2*" -or
        $_.CommandLine -like "*full_2.1.0.7z*" -or
        $_.CommandLine -like "*sensor_vlm.download_alfred*"
    } |
    Select-Object ProcessId, Name, CommandLine

if ($downloaders) {
    "Downloader process:"
    $downloaders | Format-Table -AutoSize
} else {
    "Downloader process: not running"
}

