"""Write a copy of a solver file with some env-var defaults replaced: variant.py SRC DST NAME=VAL ..."""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    s, n = re.subn(r'(^%s = \w+\(os\.environ\.get\("%s", )"[^"]*"' % (k, k), r'\1"%s"' % v, s, flags=re.M)
    assert n == 1, k
open(dst, "w").write(s)
