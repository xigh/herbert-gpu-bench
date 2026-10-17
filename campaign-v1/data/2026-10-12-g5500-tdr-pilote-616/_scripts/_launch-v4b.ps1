# g5500, second monitored run (12 October 2026, evening): stop the scheduled task llama-task,
# set the earlier outputs aside, then V4 under its temperature watchdog and, only if it finished, the
# herbert-vk-f16 probe and f16/bf16 A/B (bench/_after.sh). Launched by WMI, outside the ssh session.
$p = $env:USERPROFILE
Stop-ScheduledTask -TaskName llama-task -ErrorAction SilentlyContinue
Start-Sleep 3
"llama-task: " + (Get-ScheduledTask -TaskName llama-task).State + "; llama-server running: " + [bool](Get-Process llama-server -ErrorAction SilentlyContinue)
foreach ($d in 'v4-g5500', 'v4-g5500.out') { if (Test-Path "$p\$d") { Rename-Item "$p\$d" "$d.plantage-16h16" } }
$inner = 'bash ~/_v4-f16-win.sh > ~/v4-g5500.out 2>&1; if grep -q ALLDONE-V4-G5500 ~/v4-g5500.out; then cd ~/herbert-vk-f16 && DONE_FILE=$HOME/v4-g5500.out DONE_PAT=ALLDONE-V4-G5500 DEVS=nvidia SIZES=\"0.8B 2B\" MODELS=/c/Users/zexig/models ORACLE=/c/Users/zexig/c1/pub/target/release/herbert-bf16-chat.exe EXE=.exe bash bench/_after.sh > after.out 2>&1; fi'
$cmd = '"C:\Program Files\Git\bin\bash.exe" -l -c "' + $inner + '"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $p }
"pid=$($r.ProcessId) rc=$($r.ReturnValue)"
