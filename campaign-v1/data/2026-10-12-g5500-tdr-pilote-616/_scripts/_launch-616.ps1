# g5500, driver 616.92 (12 October 2026, night): stop the scheduled task llama-task for the tests, then
# campaign-616/_g5500-616.sh by WMI, outside the ssh session. The script restarts the task at the end.
$p = $env:USERPROFILE
Stop-ScheduledTask -TaskName llama-task -ErrorAction SilentlyContinue
Start-Sleep 5
"llama-task: " + (Get-ScheduledTask -TaskName llama-task).State + "; llama-server running: " + [bool](Get-Process llama-server -ErrorAction SilentlyContinue)
$inner = 'bash ~/campaign-616/_g5500-616.sh > ~/g5500-616.out 2>&1'
$cmd = '"C:\Program Files\Git\bin\bash.exe" -l -c "' + $inner + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $p }
"pid=$($r.ProcessId) rc=$($r.ReturnValue)"
