#!/bin/sh
# Run a dev test with the Khronos Vulkan validation layer on the GPU service and report what it said.
#   brew install vulkan-validationlayers          (macOS; Linux: apt install vulkan-validationlayers)
#   sh test/gpuserv/vk_validation.sh gist_dev.sh   (any script of locked_test.sh; default gist_dev.sh)
# Messages go to test-data/scratch/vk_validation.log; the script exits non-zero if the layer logged an error or warning of
# its own (VUID) or the test failed. GZ/GP-style sizes of the test are for the caller to set: validation is slow.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
TEST=${1:-gist_dev.sh}
OUT=$REPO/test-data/scratch
mkdir -p "$OUT"
LOG=$OUT/vk_validation.log
rm -f "$LOG"
for d in /opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d /usr/share/vulkan/explicit_layer.d /etc/vulkan/explicit_layer.d; do
	[ -f "$d/VkLayer_khronos_validation.json" ] && export VK_LAYER_PATH=$d && break
done
[ -n "${VK_LAYER_PATH:-}" ] || { echo "the Khronos validation layer is not installed"; exit 2; }
cat > "$OUT/vk_layer_settings.txt" <<SET
khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG
khronos_validation.log_filename = $LOG
khronos_validation.report_flags = error,warn
khronos_validation.validate_sync = true
khronos_validation.validate_best_practices = false
SET
export VK_LAYER_SETTINGS_PATH=$OUT
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LOADER_LAYERS_ENABLE='*validation*'
sh "$REPO/test/gpuserv/locked_test.sh" "$REPO" "$TEST"
rc=$?
n=$(grep -c "VUID\|Validation Error\|Validation Warning" "$LOG" 2>/dev/null)
n=${n:-0}
[ -f "$LOG" ] && echo "validation log: $LOG ($n messages)" || echo "the layer wrote no log file: it was probably not active"
grep "VUID" "$LOG" 2>/dev/null | sed 's/.*\(VUID-[A-Za-z0-9_-]*\).*/\1/' | sort | uniq -c | sort -rn | head -15
[ "$rc" -eq 0 ] && [ "$n" -eq 0 ]
