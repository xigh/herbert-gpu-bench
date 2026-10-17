Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; StartTime=(Get-Date '2026-10-10')} -ErrorAction SilentlyContinue |
  ForEach-Object { "{0}  id={1}  {2}" -f $_.TimeCreated, $_.Id, (($_.Properties | ForEach-Object { $_.Value }) -join ' | ') }
"--- BugCheck / WER since 10/12"
Get-WinEvent -FilterHashtable @{LogName='System'; Id=1001; StartTime=(Get-Date '2026-10-12')} -ErrorAction SilentlyContinue | ForEach-Object { "{0}  {1}  {2}" -f $_.TimeCreated, $_.ProviderName, ($_.Message -replace "`r?`n",' ') }
"--- minidumps"
Get-ChildItem C:\Windows\Minidump -ErrorAction SilentlyContinue | Select-Object Name, LastWriteTime, Length | Format-Table -AutoSize | Out-String -Width 200
