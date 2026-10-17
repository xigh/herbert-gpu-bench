#!/usr/bin/env bash
# V8: an ablation of Herbert on the 4080, the same binary (herbert-vk-bf16 8440202) with one of its defaults turned off
# at a time, then all of them, through --tune (no code change). What each default does (src/vk/tuning.rs):
#   nofuse   fuse.norm=0 fuse.swiglu=0  the two decode fusions (RMSNorm in the projection that follows, SwiGLU in down)
#   split1   prefill.split=1            one submission per layer instead of the whole prefill block in one
#   nopipe   prefill.pipe=0             the next block recorded after the previous one returns, not queued ahead
#   block256 prefill.block=256          blocks of 256 rows instead of 1 024 (4B) or 2 048 (0.8B): the weights read more often
#   alloff   all of the above
# Herbert's own counter, as in V1: one process per measure, --warmup 1, the measured turn after it. Prompts of V1
# (p1-short, pp512, pp2048, pp4000: 1 token generated; p3-long: 256 tokens, for the decode). N rounds, the variants
# interleaved inside each round. Settings: H (tree with target/release/herbert-vk-chat), M (models), P (prompts), O, N.
set -u
: "${H:?}" "${M:?}" "${P:?}" "${O:?}"
N=${N:-5}; SIZES=${SIZES:-0.8B 4B}
HV=$H/target/release/herbert-vk-chat
mkdir -p $O/logs
{ date -u; uname -r; cat $H/COMMIT
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/env-start.txt 2>&1
declare -A T=(
  [base]=""
  [nofuse]="fuse.norm=0 fuse.swiglu=0"
  [split1]="prefill.split=1"
  [nopipe]="prefill.pipe=0"
  [block256]="prefill.block=256"
  [alloff]="fuse.norm=0 fuse.swiglu=0 prefill.split=1 prefill.pipe=0 prefill.block=256"
)
V="base nofuse split1 nopipe block256 alloff"
for r in $(seq 1 $N); do
  for m in $SIZES; do
    for p in p1-short pp512 pp2048 pp4000 p3-long; do
      max=1; [ $p = p3-long ] && max=256
      for v in $V; do
        args=(); for kv in ${T[$v]}; do args+=(--tune "$kv"); done
        $HV $M/Qwen3.5-$m --device nvidia --no-system --no-think --ctx 4608 --max-tokens $max --warmup 1 "${args[@]}" \
          --ask "$(cat $P/$p.txt)" > $O/logs/r$r-$m-$p-$v.log 2>&1
        echo "r$r $m $p $v exit $?" >> $O/exits.txt
      done
    done
  done
  echo "round $r · $(date -u +%T)"
done
{ date -u; nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv; } > $O/env-end.txt 2>&1
echo ALLDONE-V8
