# Watchdog for the g5500 runs (Windows, Git Bash), sourced: `watchdog O [PID] &`. It reads the last lines of
# $O/clocks.csv, the run's own nvidia-smi log (no second nvidia-smi: on 12 October 2026 the second one only
# said "No devices were found" once the card had dropped). Stops at 90 °C (the card's max operating temperature
# is 93 °C, reached during V2 that morning), or when the log freezes under load:
# after the TDR of 21:48:52 the logger kept printing the same row (1920 MHz, 70.03 W, 65 °C) for 23 minutes,
# while a loaded GPU never gives 20 identical rows in a row (10 s at 0.5 s). Then it touches $O/ABORT, writes
# $O/watchdog.txt, stops PID (the step's bash) and every engine, then every engine started after, until
# $O/DONE exists.
watchdog() {
  local O=$1 S=${2:-} rows last t why
  while [ ! -f $O/DONE ]; do
    sleep 1
    # Once stopped, every engine a loop still starts is stopped too, until the step ends.
    [ -f $O/ABORT ] && { for x in llama-bench llama-server herbert-vk-chat herbert-vk-serve herbert-vk-probe herbert-bf16-chat v2-client; do taskkill //F //IM $x.exe > /dev/null 2>&1; done; continue; }
    rows=$(tail -n 20 $O/clocks.csv 2>/dev/null | tr -d '\r') || continue
    [ $(printf '%s\n' "$rows" | wc -l) -lt 20 ] && continue
    last=$(printf '%s\n' "$rows" | tail -n 1); why=
    t=$(printf '%s\n' "$last" | awk -F', *' '{print $6}')
    if [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -ge 90 ]; then why="$t C"
    elif ! printf '%s\n' "$last" | grep -q ', P8,' && [ $(printf '%s\n' "$rows" | cut -d, -f2- | sort -u | wc -l) -eq 1 ]; then
      why="clock log frozen under load (GPU lost?): $last"
    fi
    if [ -n "$why" ] && [ ! -f $O/ABORT ]; then
      echo "$(date -u +%T) watchdog: $why, stopped" >> $O/watchdog.txt; touch $O/ABORT
      [ -n "$S" ] && kill $S 2>/dev/null
    fi
  done
}

# Before each measurement: wait until the GPU is back at 70 °C or below (at most 300 s), so that no engine is
# measured on a card already heated by the previous one (V2 that morning: 93 °C, engines one after the other).
cooldown() {
  local O=$1 t i
  for i in $(seq 1 300); do
    t=$(tail -n 1 $O/clocks.csv 2>/dev/null | tr -d '\r' | awk -F', *' '{print $6}')
    [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -le 70 ] && return 0
    sleep 1
  done
  echo "$(date -u +%T) cooldown: still $t C after 300 s" >> $O/cooldown.txt
}
