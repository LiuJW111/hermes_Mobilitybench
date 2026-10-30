param(
    [ValidateSet(1, 2, 4, 8)]
    [int]$Workers = 4,
    [string]$OutputDir = '',
    [switch]$Resume,
    [string]$ContinueWithModel = '',
    [switch]$RetryProviderErrors
)
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
if (-not $OutputDir) { $OutputDir = 'data/results/sample10_parallel{0}_{1}' -f $Workers, (Get-Date -Format 'yyyyMMdd') }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$Host.UI.RawUI.WindowTitle = "MobilityBench - Hermes - $Workers parallel - sample10 (115)"
$runArguments = @('-X', 'utf8', 'scripts/run_hermes_batch.py', '--workers', "$Workers", '--limit', '115', '--output-dir', $OutputDir)
if ($Resume) { $runArguments += '--resume' }
if ($ContinueWithModel) { $runArguments += @('--continue-with-model', $ContinueWithModel) }
if ($RetryProviderErrors) { $runArguments += '--retry-provider-errors' }
& (Join-Path $projectRoot '.venv/Scripts/python.exe') @runArguments
Write-Host "Run exited with code $LASTEXITCODE. Results: $OutputDir"
