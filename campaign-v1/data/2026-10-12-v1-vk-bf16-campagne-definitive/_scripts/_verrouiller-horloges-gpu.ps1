# Lock the RTX 3060 Ti clocks for the V1 campaign (#002), as on the 4080 of beast.
# Double-click verrouiller-horloges-gpu.cmd on the Desktop (it asks for administrator rights).
# Undo afterwards (or reboot), in an administrator PowerShell: nvidia-smi -rgc ; nvidia-smi -rmc

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
  # Not elevated: relaunch itself as administrator (UAC prompt), then stop here.
  Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
  exit
}

$log = Join-Path ([Environment]::GetFolderPath("Desktop")) "verrouiller-horloges-gpu.log"
Start-Transcript -Path $log -Force | Out-Null
try {
  # Highest memory clock, then the highest graphics clock supported at that memory clock.
  $pairs = nvidia-smi --query-supported-clocks=memory,graphics --format=csv,noheader,nounits |
    ForEach-Object { $m, $g = $_ -split ',\s*'; [pscustomobject]@{ Mem = [int]$m; Gfx = [int]$g } }
  $mem = ($pairs | Measure-Object Mem -Maximum).Maximum
  $gfx = ($pairs | Where-Object { $_.Mem -eq $mem } | Measure-Object Gfx -Maximum).Maximum
  Write-Host "Locking graphics at $gfx MHz and memory at $mem MHz"

  # nvidia-smi refuses both locks in one command: two commands.
  nvidia-smi -lgc "$gfx,$gfx" | Out-Host
  nvidia-smi -lmc "$mem,$mem" | Out-Host

  Write-Host "`nState now (idle clocks may read low until a kernel runs):"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv | Out-Host
  Write-Host "`nDone."
} finally {
  Stop-Transcript | Out-Null
  Read-Host "Press Enter to close"
}
