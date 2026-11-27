"""Fetch the two public GitHub datasets used for real, reproducible checks (nothing is bundled in this repo):
  * Law-AI/semantic-segmentation  (MIT)  50 Indian Supreme Court judgments, sentence-level rhetorical roles (JURIX 2019)
  * Law-AI/MILDSum samples        10 Indian judgments with practitioner-written English summaries
Check each repo's current licence/terms before reuse; do not redistribute MILDSum files."""
import os, subprocess, sys
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external")
REPOS = {"semantic-segmentation": "https://github.com/Law-AI/semantic-segmentation", "MILDSum": "https://github.com/Law-AI/MILDSum"}
os.makedirs(DEST, exist_ok=True)
for name, url in REPOS.items():
    path = os.path.join(DEST, name)
    if os.path.isdir(path):
        print("exists:", path); continue
    subprocess.check_call(["git", "clone", "--depth", "1", url, path])
    rev = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    open(os.path.join(path, "REVISION.txt"), "w").write(f"{url}\n{rev}\n")
print("done ->", DEST)
