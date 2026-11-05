# postgis3d_dev.sh: the 3D tests alone (postgis_tests.inc sources postgis3d_tests.inc itself after the 2D ones)
if ! $PSQL -d scantest -c "create extension if not exists postgis" >/dev/null 2>&1; then
	echo "  -- the postgis extension is not available: skipped"
else
. "${0%/*}/postgis3d_tests.inc"
fi
