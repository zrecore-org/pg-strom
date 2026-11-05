---
name: shell-gotchas
description: Shell and tooling pitfalls on this macOS (zsh, BSD tools) setup for PG-Strom work - word splitting, sed/awk/wc differences, backgrounding, where to put scratch files, removal safety checks, PG_CONFIG. Use before writing shell commands or test scripts here.
---

# Shell and tooling pitfalls (macOS, zsh)

- **zsh does not word-split** an unquoted `$VAR`. `PSQL="psql -h x"; $PSQL -c ...` fails in the Bash tool (it runs zsh). Use a
  shell function (`q(){ psql -h "$D" ... "$@"; }`) or write a `.sh` file and run it with `sh`. The repo scripts run under `sh` and
  do rely on word-splitting.
- **BSD tools**: `sed -i ''` (not `-i`), no `\+`/`\|` in basic sed, no `echo -n` (use `printf`), `wc -l` pads with spaces (compare
  with arithmetic, not as strings), BSD `awk` has no `gensub`. GNU make 3.81 and macOS `sh` (bash 3.2 as sh) must keep working.
- **Two statements in one `psql -c`** run in one transaction: `VACUUM` fails inside it. Use two calls.
- **Always pass `PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config`** to `make`.
- **Scratch files** go under `test-data/scratch/` (external volume; `/tmp` is cleared and the local drive is nearly full).
  Reusable scripts belong in the repo (`test/...`). The session scratchpad is for throwaway notes only.
- **Removals**: a safety check blocks `rm -rf *`. Do not work around it; use a fresh directory or a different name.
- **Long commands**: foreground `sleep` is blocked and the tool times out after 2-10 minutes. Use `run_in_background` and wait
  for the notification, or write a loop with a bounded `sleep` inside a script. Do not poll the output file.
- **Output with binary bytes** (the service log can contain them): `LC_ALL=C grep -a ... | LC_ALL=C cut -c1-200`.
- **Test lock**: tests that install or use the shared `$libdir/pg_strom` go through `sh test/gpuserv/locked_test.sh "$PWD" <script>`.
- **Git**: commit only when asked or when the standing "commit and push when green" applies; messages are short with a
  `vulkan:` / `test:` / `docs:` prefix and the `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` trailer.
