"""Write a copy of a solver file with new defaults for env knobs. usage: uv run tools/variant.py src.py dst.py NAME=val ..."""
import re, sys
s = open(sys.argv[1]).read()
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    s, n = re.subn(r'os\.environ\.get\("%s", "[^"]*"\)' % k, 'os.environ.get("%s", "%s")' % (k, v), s)
    assert n == 1, k
open(sys.argv[2], "w").write(s)
