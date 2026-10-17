#!/usr/bin/env bash
# The new loader (mapped shards, f16 weight cache, conversion and upload on every core) against e4901c4:
# (1) the same weights: the 5 best logits and the text of p1-p4, old and new, f16 and bf16, must be identical;
# (2) the load time: f16 first start (builds the cache), cached starts, no cache, bf16; old for reference.
set -u
O=$HOME/load-test; M=/opt3/models; OLD=$HOME/herbert-vk-f16-e4901c4; NEW=$HOME/herbert-vk-f16-load; P=$NEW/bench/prompts
mkdir -p $O/same $O/time
for m in 0.8B 4B; do cat $M/Qwen3.5-$m/*.safetensors > /dev/null; done
for m in 0.8B 4B; do
  for p in $P/p*.txt; do
    name=$(basename $p .txt); think="--no-think"; max=200; case $name in p3-*) think=""; max=256;; esac
    for v in old:$OLD new:$NEW; do for f in f16:target bf16:target-bf16; do
      t=${v%%:*}-${f%%:*}; d=${v#*:}/${f#*:}/release
      HERBERT_VK_TOP=5 $d/herbert-vk-chat $M/Qwen3.5-$m --device nvidia --no-system $think --max-tokens $max --ask "$(cat $p)" > $O/same/$m-$name-$t.log 2>&1
    done; done
    for f in f16 bf16; do
      if diff <(grep -v '^\[' $O/same/$m-$name-old-$f.log) <(grep -v '^\[' $O/same/$m-$name-new-$f.log) > /dev/null; then r=identical; else r=DIFFERENT; fi
      echo "same $m $name $f: $r"
    done
  done
done
one() { "$@" --device nvidia --no-system --max-tokens 1 --ask "Bonjour" 2>&1 | grep -E "^\[loader\] f16|loaded in" | sed -E 's/.*(loaded in [0-9.]+ s).*/\1/'; }
for m in 0.8B 4B; do
  C=$HOME/.cache/herbert/Qwen3.5-$m/vk-f16_v1.herbcache; rm -f -- "$C"
  echo "== $m"
  echo "new f16, first start (builds the cache):"; one $NEW/target/release/herbert-vk-chat $M/Qwen3.5-$m
  for i in 1 2 3; do echo "new f16, cached $i:"; one $NEW/target/release/herbert-vk-chat $M/Qwen3.5-$m; done
  for i in 1 2; do echo "new f16, HERBERT_CACHE=0 $i:"; HERBERT_CACHE=0 one $NEW/target/release/herbert-vk-chat $M/Qwen3.5-$m; done
  for i in 1 2; do echo "old f16 $i:"; one $OLD/target/release/herbert-vk-chat $M/Qwen3.5-$m; done
  for i in 1 2 3; do echo "new bf16 $i:"; one $NEW/target-bf16/release/herbert-vk-chat $M/Qwen3.5-$m; done
  for i in 1 2; do echo "old bf16 $i:"; one $OLD/target-bf16/release/herbert-vk-chat $M/Qwen3.5-$m; done
  ls -la "$C"
done
echo ALLDONE-LOAD
