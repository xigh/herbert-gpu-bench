set -u
cd ~/herbert-vk-bf16-load && tar xzf hvkbf16-load.tgz && rm hvkbf16-load.tgz && . ~/.cargo/env && cargo build --release --locked --features serve 2>&1 | tail -1
O=$HOME/load-test-bf16; M=/opt3/models; OLD=$HOME/herbert-vk-bf16-v1/target/release; NEW=$HOME/herbert-vk-bf16-load/target/release; P=$HOME/herbert-vk-bf16-load/bench/prompts
mkdir -p $O; cat $OLD/../../COMMIT
for m in 0.8B 4B; do cat $M/Qwen3.5-$m/*.safetensors > /dev/null; done
for m in 0.8B 4B; do for p in $P/p*.txt; do
  name=$(basename $p .txt); think="--no-think"; max=200; case $name in p3-*) think=""; max=256;; esac
  for v in old:$OLD new:$NEW; do HERBERT_VK_TOP=5 ${v#*:}/herbert-vk-chat $M/Qwen3.5-$m --device nvidia --no-system $think --max-tokens $max --ask "$(cat $p)" > $O/$m-$name-${v%%:*}.log 2>&1; done
  if diff <(grep -v '^\[' $O/$m-$name-old.log) <(grep -v '^\[' $O/$m-$name-new.log) > /dev/null; then echo "same $m $name: identical"; else echo "same $m $name: DIFFERENT"; fi
done; done
for m in 0.8B 4B; do for v in old new old new old new; do d=$OLD; [ $v = new ] && d=$NEW
  echo "$m $v: $($d/herbert-vk-chat $M/Qwen3.5-$m --device nvidia --no-system --max-tokens 1 --ask Bonjour 2>&1 | grep -o 'loaded in [0-9.]* s')"; done; done
echo ALLDONE
