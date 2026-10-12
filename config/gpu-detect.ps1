param([string]$Catalog = (Join-Path $PSScriptRoot 'gpu_runtimes.json'), [string]$IniPath = '')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$result = @{ compatible=$false; provider=$null; reason='No compatible GPU was detected. Using CPU.'; adapters=@(); eligible_indices=@() }
try {
    $catalogData = Get-Content -Raw -LiteralPath $Catalog | ConvertFrom-Json
    $adapters = @(Get-CimInstance Win32_VideoController -OperationTimeoutSec 5 | ForEach-Object {
        $vendor = if ($_.PNPDeviceID -match 'VEN_10DE') {'nvidia'} elseif ($_.PNPDeviceID -match 'VEN_8086') {'intel'} elseif ($_.PNPDeviceID -match 'VEN_1002') {'amd'} else {'unknown'}
        @{ name=$_.Name; vendor=$vendor }
    })
    $result.adapters = $adapters
    foreach ($adapter in $adapters) {
        foreach ($property in $catalogData.providers.PSObject.Properties) {
            $provider = $property.Value
            if ($adapter.vendor -in $provider.vendors -and -not $provider.enabled) { $result.reason=$provider.reason }
        }
    }
    if (@($adapters | Where-Object { $_.vendor -eq 'nvidia' }).Count) {
        $result.reason = 'The GPU driver could not be checked. Update the graphics driver and check again. Using CPU.'
        $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($smi) {
            $rows = & $smi.Source '--query-gpu=index,driver_version,compute_cap' '--format=csv,noheader,nounits' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $policy=$catalogData.providers.cuda
                $result.reason='The GPU or its driver is not compatible with the available runtime. Update the graphics driver if supported. Using CPU.'
                foreach ($row in $rows) {
                    $fields=$row.Split(',').Trim()
                    if ($fields.Length -eq 3 -and [version]$fields[1] -lt [version]$policy.min_driver) {
                        $result.reason='The graphics driver is too old for this GPU runtime. Driver ' + $policy.min_driver + ' or newer is required. Using CPU.'
                    }
                    elseif ($fields.Length -eq 3 -and ([double]::Parse($fields[2],[cultureinfo]::InvariantCulture) -lt $policy.min_compute -or [double]::Parse($fields[2],[cultureinfo]::InvariantCulture) -gt $policy.max_compute)) {
                        $result.reason='This GPU architecture is not supported by the available runtime. Using CPU.'
                    }
                    if ($fields.Length -eq 3 -and [version]$fields[1] -ge [version]$policy.min_driver -and
                        [double]::Parse($fields[2],[cultureinfo]::InvariantCulture) -ge $policy.min_compute -and
                        [double]::Parse($fields[2],[cultureinfo]::InvariantCulture) -le $policy.max_compute) {
                        $result.eligible_indices += [int]$fields[0]
                    }
                }
                if ($result.eligible_indices.Count -gt 0 -and $policy.enabled) {
                    $result.compatible=$true; $result.provider='cuda'; $result.reason='A compatible GPU is available.'
                }
            }
        }
    }
} catch { $result.reason='GPU compatibility could not be checked. Using CPU; check again after updating the graphics driver.' }
if ($IniPath) {
    $downloadBytes=0
    if ($result.compatible) { $downloadBytes=($catalogData.providers.($result.provider).files | Measure-Object size -Sum).Sum }
    $text="[GPU]`r`ncompatible=$([int]$result.compatible)`r`nprovider=$($result.provider)`r`nreason=$($result.reason)`r`ndownload_bytes=$downloadBytes`r`n"
    [IO.File]::WriteAllText($IniPath,$text,[Text.Encoding]::Unicode)
}
$result | ConvertTo-Json -Depth 5 -Compress
