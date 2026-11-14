# svc_log_check <service log> [allow regex]: an "ALL PASSED" line must not hide a GPU failure. A failed task falls back to the CPU and
# the differential test still passes, so scan the service log for the failure messages (A17: T4 join pipelines, A6b: a fallback, a
# device loss) and count every line that the allow-list does not name as a failure. Sourced by the dev scripts, which have $fail.
#   SVC_ALLOW=<egrep>   lines to accept (a known, documented fallback); the scripts set their own default
svc_log_check() {
	_log=$1; _allow=${2:-}
	[ -f "$_log" ] || { echo "  note: no service log at $_log, the fallback scan was skipped"; return 0; }
	_bad=$(LC_ALL=C grep -a -E "task failed:|GPU execution failed|VK_ERROR_DEVICE_LOST|VK_ERROR_UNKNOWN|cannot create the .* pipeline|final failed:" "$_log" \
		| LC_ALL=C grep -a -v "injected fault" | { [ -n "$_allow" ] && LC_ALL=C grep -a -v -E "$_allow" || cat; })
	if [ -n "$_bad" ]; then
		echo "  FAIL  the GPU service reported failures (the CPU fallback hid them from the differential tests):"
		echo "$_bad" | sed 's/^[0-9-]* [0-9:.]* [A-Z]* \[[0-9]*\] //; s/^pgstrom-gpuserv\[[0-9]*\]: //' | cut -c1-200 | sort | uniq -c | sort -rn | head -8 | sed 's/^/        /'
		fail=$((fail+1))
	else
		echo "  ok    no unexpected GPU failure in the service log"
	fi
}
