"""Record a solver file as an attempt without touching slim.py: copy it to the run folder root with MODEL_NAME /
DESCRIPTION set, run the harness (recorded), snapshot it to slim_lib/NAME.py, set the status, print the summary.
usage: uv run tools/rec2.py file.py NAME "DESCRIPTION" [status=discard]"""
import os, re, subprocess, sys, shutil
import pandas as pd
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src, name, desc = sys.argv[1], sys.argv[2], sys.argv[3]
status = sys.argv[4] if len(sys.argv) > 4 else "discard"
s = open(src).read()
s = re.sub(r'^MODEL_NAME = .*$', f'MODEL_NAME = "{name}"', s, count=1, flags=re.M)
s = re.sub(r'^DESCRIPTION = \(.*?\)\n', 'DESCRIPTION = (' + repr(desc) + ')\n', s, count=1, flags=re.M | re.S)
tmp = os.path.join(here, f"_rec_{name}.py")
open(tmp, "w").write(s)
shutil.copy(tmp, os.path.join(here, "slim_lib", name + ".py"))
log = os.path.join(here, "scratch", f"rec_{name}.log")
with open(log, "w") as f:
    subprocess.run([sys.executable, tmp], cwd=here, stdout=f, stderr=subprocess.STDOUT)
os.remove(tmp)
p = os.path.join(here, "results", "overall_results.csv")
df = pd.read_csv(p, dtype=str, keep_default_na=False)
if (df.model_name == name).sum() == 1:
    df.loc[df.model_name == name, "status"] = status
    df.to_csv(p, index=False)
print("".join(l for l in open(log).readlines()[-11:] if l.split(":")[0] in ("model", "mean_regret", "geo_mean_time", "mean_test_auc", "mean_loss", "n_invalid")))
print("load", os.getloadavg()[0])
