#!/bin/sh
# One lock hold for the full validation of a tree: scan_test.sh (every differential test, in check mode: the
# specialized kernels against the VM and the CPU), then the regression subset (test/vulkan_installcheck.sh);
# when the subset differs, the alternate expected files are regenerated (test/vulkan_regen_expected.sh: only
# plan text may differ, the rows must equal the CPU oracle's) and the subset is run again. Run it through
# locked_test.sh (it needs the module and the service installed).
D=${0%/*}
T=${D%/*}
sh "$D/scan_test.sh"
rc1=$?
echo "=== scan_test.sh exit $rc1"
echo "=== regression subset"
sh "$T/vulkan_installcheck.sh"
rc2=$?
echo "=== vulkan_installcheck.sh exit $rc2"
if [ $rc2 -ne 0 ]; then
	echo "=== regenerating the alternate expected files"
	sh "$T/vulkan_regen_expected.sh"
	echo "=== regen exit $?; the subset again"
	sh "$T/vulkan_installcheck.sh"
	rc2=$?
	echo "=== vulkan_installcheck.sh exit $rc2 (after the regeneration)"
fi
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
