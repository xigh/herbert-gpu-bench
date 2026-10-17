#!/usr/bin/env bash
# Run after the machine's current campaign, without the Mac: wait for DONE_PAT in DONE_FILE, then build this
# tree (f16 and bf16), run the probe of both on every device of DEVS (the 16-bit conversions selftest,
# bandwidth), then the f16/bf16 A/B (_f16-ab.sh, 3 rounds, tokens against the CPU oracle) on each.
# Settings: DONE_FILE, DONE_PAT, DEVS ("nvidia" "amd"...), SIZES, MODELS, ORACLE, EXE (".exe" on Windows), AB=0 to skip the A/B.
set -u
# Outside a login shell the system cargo may come first: rustup's, as the builds by hand.
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
cd "$(dirname "$0")/.."
until grep -q "$DONE_PAT" "$DONE_FILE" 2>/dev/null; do sleep 60; done
echo "start · $(date -u +%T)"
mkdir -p runs
cargo build --release --locked > runs/build-f16.log 2>&1; echo "build f16 exit $?"
cargo build --release --locked --no-default-features --features bf16 --target-dir target-bf16 > runs/build-bf16.log 2>&1; echo "build bf16 exit $?"
for dev in $DEVS; do
  for v in f16:target bf16:target-bf16; do
    n=${v%%:*}; d=${v#*:}
    $d/release/herbert-vk-probe${EXE:-} --device $dev > runs/probe-$dev-$n.log 2>&1
    echo "probe $dev $n exit $? · $(grep -h 'conversions' runs/probe-$dev-$n.log)"
  done
done
if [ "${AB:-1}" = 1 ]; then
  for dev in $DEVS; do
    bash bench/_f16-ab.sh runs/ab-$dev $dev 3 "${SIZES:-0.8B 4B}"
  done
fi
echo ALLDONE-AFTER
