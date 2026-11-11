#!/bin/sh
# What failed in a GitHub Actions run, in a few lines: the failed jobs, their failed steps and the first error lines of each
# job's log (instead of downloading and reading whole logs).
#
#   sh test/gpuserv/ci_log.sh <run-id> [job-name-regex]        (the repository is the one of the current directory)
#   LINES=25 sh test/gpuserv/ci_log.sh <run-id>                  (more error lines per job)
RUN=${1:?usage: ci_log.sh <run-id> [job-name-regex]}
PAT=${2:-.}
LINES=${LINES:-12}
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner) || exit 1
gh api "repos/$REPO/actions/runs/$RUN/jobs?per_page=100" --jq '.jobs[] | select(.conclusion=="failure" or .conclusion=="cancelled") | "\(.id)\t\(.name)\t\(.conclusion)"' |
while IFS="$(printf '\t')" read -r id name concl; do
	echo "$name" | grep -qE "$PAT" || continue
	echo "== $name ($concl)"
	gh api "repos/$REPO/actions/jobs/$id" --jq '.steps[] | select(.conclusion=="failure") | "   failed step \(.number): \(.name)"'
	gh api "repos/$REPO/actions/jobs/$id/logs" --allow-escape-sequences 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' |
		grep -E '##\[error\]|error:|Err:|E: |FAIL|undefined reference|Permission denied|Temporary failure|timed out|Cannot|No such file' |
		cut -c29-230 | head -"$LINES"
done
