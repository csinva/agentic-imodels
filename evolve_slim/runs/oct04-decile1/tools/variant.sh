#!/bin/bash
# usage: tools/variant.sh src.py out.py KNOB=VAL ...  (changes the env defaults of knobs in a copy)
src=$1; out=$2; shift 2; cp $src $out
for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}; grep -q "os.environ.get(\"$k\", " $out || { echo "no knob $k"; exit 1; }
  sed -i "s/os.environ.get(\"$k\", \"[^\"]*\")/os.environ.get(\"$k\", \"$v\")/" $out; done
