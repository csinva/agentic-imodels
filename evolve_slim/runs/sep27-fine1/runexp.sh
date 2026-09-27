#!/bin/sh
# usage: ./runexp.sh NAME "description"  -- sets MODEL_NAME/DESCRIPTION, runs the suite, snapshots
cd "$(dirname "$0")"
python3 - "$1" "$2" <<'P'
import re, sys
src = open('slim.py').read()
src = re.sub(r'MODEL_NAME = ".*?"', 'MODEL_NAME = "%s"' % sys.argv[1], src, count=1)
src = re.sub(r'DESCRIPTION = \(.*?\)\n', 'DESCRIPTION = (%r)\n' % sys.argv[2], src, count=1, flags=re.S)
open('slim.py', 'w').write(src)
P
uv run slim.py > run.log 2>&1
cp slim.py slim_lib/$1.py
tail -n 9 run.log | head -5
