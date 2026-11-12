#!/bin/sh
# Which tests does a change need? Maps the changed files to the dev scripts (a few minutes each) and says when the full run
# (verify_all.sh, over an hour) is needed. Meant for the edit-and-test loop; run the full verification before a push.
#
#   sh test/gpuserv/affected.sh                  files changed against HEAD (working tree, staged, untracked)
#   sh test/gpuserv/affected.sh origin/vulkan-gpu..HEAD     a commit range
#   RUN=1 sh test/gpuserv/affected.sh            run the suggested scripts, one after the other, under the test lock
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO" || exit 2
if [ $# -gt 0 ]; then
	files=$(git diff --name-only "$@")
else
	files=$( { git diff --name-only HEAD; git ls-files --others --exclude-standard; } | sort -u)
fi
[ -n "$files" ] || { echo "no changed files"; exit 0; }

want=""
add() { case " $want " in *" $1 "*) ;; *) want="$want $1" ;; esac; }
full=0
for f in $files; do
	case $f in
		src/gpuserv/kernels_vm/*|src/gpuserv/kernels/*)
			add spec_compile_check.sh; for s in float8_dev.sh intvar_dev.sh postgis_dev.sh gist_dev.sh arrow_dev.sh gpucache_dev.sh joinagg_dev.sh gpusort_dev.sh spec_dev.sh recheck_dev.sh; do add $s; done ;;
		src/gpuserv/specialize.c|src/gpuserv/spec_*.c|src/gpuserv/spec_*.h)
			add spec_compile_check.sh; add spec_dev.sh ;;
		src/gpuserv/scan_prog.c|src/gpuserv/scan_caps.c|src/gpuserv/cpu_vm.c|src/codegen.c|src/xpu_opcodes.h|src/gpuserv/scan_exec.c|src/gpuserv/arrow_exec.c)
			full=1; add spec_compile_check.sh; add coverage_test.sh ;;
		test/gpuserv/*_dev.sh) add "$(basename "$f")" ;;
		src/gist.c|src/gpu_join.c|test/gpuserv/gist_*) add gist_dev.sh; add coverage_test.sh ;;
		src/gpu_cache.c|src/gpuserv/gcache.c|test/gpuserv/gpucache_*) add gpucache_dev.sh; add recheck_dev.sh ;;
		src/arrow_*|src/gpuserv/arrow*|test/gpuserv/arrow_*) add arrow_dev.sh; add recheck_dev.sh ;;
		test/gpuserv/recheck_*) add recheck_dev.sh ;;
		src/gpu_preagg.c|test/gpuserv/joinagg_*) add joinagg_dev.sh; add coverage_test.sh ;;
		test/gpuserv/coverage_*) add coverage_test.sh ;;
		test/gpuserv/gpusort_*) add gpusort_dev.sh ;;
		test/gpuserv/postgis3d_*|test/gpuserv/postgis/*) add postgis3d_dev.sh; add postgis_dev.sh ;;
		test/gpuserv/postgis_*) add postgis_dev.sh ;;
		test/gpuserv/minmaxtext_*) add minmaxtext_dev.sh ;;
		test/gpuserv/float8_*) add float8_dev.sh ;;
		test/gpuserv/intvar_*) add intvar_dev.sh ;;
		test/gpuserv/spec_*) add spec_dev.sh ;;
		test/gpuserv/jsonb_*) add jsonb_test.sh ;;
		test/gpuserv/numeric_*) add numeric_test.sh ;;
		src/gpuserv/pcache_check.c|test/gpuserv/pcache_test.sh) add pcache_test.sh ;;
		src/gpuserv/gpu_vk.c|src/gpuserv/gpu_vk.h)
			full=1; add service_test.sh; add pcache_test.sh ;;
		src/fallback.c|test/gpuserv/fallback_*) add fallback_dev.sh; full=1 ;;
		src/executor.c|src/gpu_client.c|src/gpuserv/wire.c|src/gpuserv/gpu_vk*|src/gpuserv/gpu_vk.h|src/gpu_launcher.c)
			full=1; add service_test.sh; add fallback_dev.sh ;;
		src/*|test/gpuserv/scan_test.sh|test/gpuserv/types_tests.inc|test/gpuserv/fuzz_regress.inc|test/vulkan_*) full=1 ;;
		*) ;;		# docs, workflows, scripts that are not tests
	esac
done

echo "changed files:"; echo "$files" | sed 's/^/  /'
echo
echo "run these (each under the shared test lock):"
for s in $want; do
	case $s in
		spec_compile_check.sh) echo "  sh test/gpuserv/spec_compile_check.sh" ;;
		service_test.sh) echo "  GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/service_test.sh" ;;
		*) echo "  sh test/gpuserv/locked_test.sh \"\$PWD\" $s" ;;
	esac
done
[ "$full" -eq 1 ] && echo "  and before a push: STAGES=\"installcheck scan\" sh test/gpuserv/verify_all.sh   (this change touches shared planner, program or service code)"
[ -n "$want" ] || [ "$full" -eq 1 ] || echo "  (nothing: documentation or tooling only)"

if [ "${RUN:-0}" = 1 ]; then
	rc=0
	for s in $want; do
		case $s in
			spec_compile_check.sh) sh test/gpuserv/spec_compile_check.sh || rc=1 ;;
			service_test.sh) echo "service_test.sh needs GPUSERV_BIN; skipped by RUN=1" ;;
			*) PGSTROM_GPUSERV_SPECIALIZE=${PGSTROM_GPUSERV_SPECIALIZE:-check} sh test/gpuserv/locked_test.sh "$PWD" $s || rc=1 ;;
		esac
	done
	exit $rc
fi
