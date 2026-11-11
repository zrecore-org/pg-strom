#!/bin/sh
# One screen of the state of the work on this machine: the GPU lock and its queue, the disk, running test scripts, the
# latest CI runs and the branch. Cheaper to read than asking five tools.
#   sh test/gpuserv/status.sh
HERE=$(cd "$(dirname "$0")" && pwd)
echo "time: $(date '+%Y-%m-%d %H:%M %Z')   disk free on /: $(df -h / | awk 'NR==2{print $4}')   external: $(df -h /Volumes/CrucialX9 2>/dev/null | awk 'NR==2{print $4}')"
echo "gpu lock: $(sh "$HERE/gpu_lock.sh" status --status 2>&1 | tr '\n' ';')"
echo "running: $(ps -axo etime,command | grep -E 'locked_test|verify_all|gpu_lock' | grep -v grep | sed -E 's#/Volumes/CrucialX9/Projects/pg-strom/##g' | cut -c1-110 | head -4 | tr '\n' '|')"
REPO=$(cd "$HERE" && gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null)
[ -n "$REPO" ] && gh run list --repo "$REPO" --limit 4 --json databaseId,status,conclusion,headBranch,event --jq '.[] | "ci \(.databaseId) \(.status) \(.conclusion // "-") \(.headBranch) \(.event)"' 2>/dev/null
echo "git: $(git -C "$HERE" status -sb | head -1)   unpushed: $(git -C "$HERE" rev-list --count '@{u}..HEAD' 2>/dev/null)"
git -C "$HERE" worktree list 2>/dev/null | grep -c worktrees | sed 's/^/worktrees: /'
