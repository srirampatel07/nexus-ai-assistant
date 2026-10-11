
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("text", "vision")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"

# Locate the models
$hf = "$env:USERPROFILE\.cache\huggingface\hub"

$textRepo = Join-Path $hf "models--Qwen--Qwen3-8B-GGUF"
$visionRepo = Join-Path $hf "models--ggml-org--Qwen2.5-VL-3B-Instruct-GGUF"

$textSnapshot = (Get-ChildItem "$textRepo\snapshots" -Directory |
    Select-Object -First 1).FullName

$visionSnapshot = (Get-ChildItem "$visionRepo\snapshots" -Directory |
    Select-Object -First 1).FullName

$textModel = Join-Path $textSnapshot "Qwen3-8B-Q4_K_M.gguf"
$visionModel = Join-Path $visionSnapshot "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
$mmproj = Join-Path $visionSnapshot "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"

$server = (Get-Command llama-server.exe -ErrorAction Stop).Source

# Verify selected model files before stopping anything
if ($Mode -eq "text" -and !(Test-Path $textModel)) {
    throw "Text model file not found: $textModel"
}
if ($Mode -eq "vision" -and (!(Test-Path $visionModel) -or !(Test-Path $mmproj))) {
    throw "Vision model or matching mmproj file not found."
}

# Stop only NEXUS model servers using ports 8080 and 8081
$oldServers = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
    Where-Object { $_.CommandLine -match '--port\s+(8080|8081)(\s|$)' }

foreach ($proc in $oldServers) {
    Write-Host "Stopping NEXUS model server PID $($proc.ProcessId)..."
    Stop-Process -Id $proc.ProcessId -Force
}

# Wait for those processes to exit
Start-Sleep -Seconds 3

$remaining = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" |
    Where-Object { $_.CommandLine -match '--port\s+(8080|8081)(\s|$)' }

if ($remaining) {
    throw "An old NEXUS model server is still running. Stop it manually before continuing."
}

# Check available physical RAM after stopping the previous server
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)

Write-Host "Available RAM after stopping previous server: $freeGB GB"

if ($freeGB -lt 3.5) {
    throw "Less than 3.5 GB RAM is available. Closing other applications may be necessary. No model was started."
}

if ($Mode -eq "text") {
    Write-Host "Starting Qwen3-8B text server on port 8080..."
    & $server -m $textModel --host 127.0.0.1 --port 8080 -c 4096 -t 6 -b 256
}
else {
    Write-Host "Starting Qwen2.5-VL-3B vision server on port 8081..."
    & $server -m $visionModel --mmproj $mmproj --host 127.0.0.1 --port 8081 -c 4096 -t 6 -b 128
}
