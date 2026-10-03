"""Set MODEL_NAME / DESCRIPTION in slim.py, run the recorded full suite, snapshot to slim_lib/, print summary.
usage: uv run tools/attempt.py NAME "DESCRIPTION" """
import os, re, subprocess, sys, shutil
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name, desc = sys.argv[1], sys.argv[2]
p = os.path.join(here, "slim.py"); s = open(p).read()
s = re.sub(r'^MODEL_NAME = .*$', f'MODEL_NAME = "{name}"', s, count=1, flags=re.M)
s = re.sub(r'^DESCRIPTION = \(.*?\)\n', 'DESCRIPTION = (' + repr(desc) + ')\n', s, count=1, flags=re.M | re.S)
open(p, "w").write(s)
shutil.copy(p, os.path.join(here, "slim_lib", name + ".py"))
with open(os.path.join(here, "run.log"), "w") as f:
    subprocess.run([sys.executable, "slim.py"], cwd=here, stdout=f, stderr=subprocess.STDOUT)
print("".join(open(os.path.join(here, "run.log")).readlines()[-11:]))
print("load", os.getloadavg()[0])
