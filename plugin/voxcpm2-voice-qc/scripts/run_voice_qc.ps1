param(
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$entry = Join-Path $projectRoot 'voice_qc_flow.py'
$config = Join-Path $projectRoot 'config.json'

if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
    throw "Workflow entrypoint was not found: $entry"
}
if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "Local config was not found: $config"
}
$localConfig = Get-Content -Raw -LiteralPath $config | ConvertFrom-Json
$python = [Environment]::ExpandEnvironmentVariables([string]$localConfig.paths.voxcpm_python)
if (-not [IO.Path]::IsPathRooted($python)) {
    $python = Join-Path $projectRoot $python
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Configured VoxCPM2 Python was not found: $python"
}

$arguments = @($entry, '--config', $config)
if ($CheckOnly) {
    $arguments += '--check-only'
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "VoxCPM2 workflow failed with exit code $LASTEXITCODE"
}
