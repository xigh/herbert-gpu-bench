$since = Get-Date '2026-10-12 18:00'
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$since} -ErrorAction SilentlyContinue |
  Where-Object { $_.ProviderName -match 'nvlddmkm|BugCheck|Kernel-Power|WHEA|Display|EventLog' -or $_.Level -le 2 } |
  Select-Object TimeCreated, ProviderName, Id, LevelDisplayName, @{n='Msg';e={($_.Message -split "`n")[0]}} |
  Format-Table -AutoSize -Wrap | Out-String -Width 220
