#!/usr/bin/env bash
# g5500 on driver 616.92 (12 October 2026, night, after the BSOD of 15:36 and the TDR of 21:48 on 591.86): the
# whole campaign again on the new driver, clocks not locked, under _watchdog-win.sh with a cooldown to 70 °C
# before each measurement. V1, V2, herbert-vk-f16 (probe, f16/bf16 A/B), then V4, the load that brought the
# card down twice; each step only if the previous one was not stopped. Then the scheduled task again.
set -u
U0=/c/Users/zexig; S=$(cd "$(dirname "$0")" && pwd); F=$U0/herbert-vk-f16-cc51363
. $S/_watchdog-win.sh
step() { local name=$1 dir=$2 script=$3; echo "== $name · $(date -u +%T)"; O=$dir bash $S/$script > $U0/$name.out 2>&1
  tail -n 2 $U0/$name.out; [ ! -f $dir/ABORT ]; }
f16() { local O=$U0/f16-g5500-616; mkdir -p $O; echo "== f16-616 · $(date -u +%T)"
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,fan.speed --format=csv -lms 500 > $O/clocks.csv 2>&1 &
  local SMI=$!; cooldown $O
  ( cd $F && exec env DONE_FILE=$U0/v2-616.out DONE_PAT=ALLDONE-V2-G5500 DEVS=nvidia SIZES="0.8B 2B" MODELS=$U0/models ORACLE=$U0/c1/pub/target/release/herbert-bf16-chat.exe EXE=.exe bash bench/_after.sh > $U0/f16-616.out 2>&1 ) &
  local P=$!; watchdog $O $P & local W=$!
  wait $P; kill $SMI 2>/dev/null; touch $O/DONE; wait $W; tail -n 2 $U0/f16-616.out; [ ! -f $O/ABORT ]; }
nvidia-smi --query-gpu=name,driver_version --format=csv
step v1-616 $U0/v1-g5500-616 _v1-win.sh && step v2-616 $U0/v2-g5500-616 _v2-win.sh && f16 && step v4-616 $U0/v4-g5500-616 _v4-f16-win.sh
echo "llama-task: $(schtasks //run //tn llama-task 2>&1)"
echo ALLDONE-G5500-616
