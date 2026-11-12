# min / max of text, varchar and bpchar (A6b). Sourced by minmaxtext_dev.sh (needs q, GPU, CPU, check, sql, PSQL, MM_EXPECT_GPU and the
# tables tmo / tm1 / tm0 and the Arrow, Parquet and GPU-Cache copies tmoa / tmop / tmog).
echo "min / max of strings: the partial-state functions"

# 1. The SQL functions alone (no GPU involved): pgstrom.pmin / pmax of a column, merged by pgstrom.min_text / max_text (the aggregates the
# Agg node above a GpuPreAgg runs), against the server's min / max under the C collation. The partial states are built in two levels to
# test the merge of states (a state with a value, an empty state, NULL).
rps() { # label, cpu query, partial-state query (both return rows; compared as sorted text)
	c=$(q "$CPU" "$2"); p=$(q "$CPU" "$3")
	if [ "$c" = "$p" ] && ! echo "$p" | grep -q ERROR; then echo "  ok    $1 (${p#*|} rows)"
	else echo "  FAIL  $1"; echo "        min/max: $c"; echo "        partial: $p"; fail=$((fail+1)); fi
}
for tt in "t text" "v text" "w bpchar"; do
	col=${tt% *}; ty=${tt#* }
	rps "$col: ungrouped min, max" \
		"select min($col collate \"C\"), max($col collate \"C\") from tmo" \
		"select pgstrom.min_$ty(p1), pgstrom.max_$ty(p2) from (select pgstrom.pmin($col) p1, pgstrom.pmax($col) p2 from tmo) x"
	rps "$col: grouped min, max" \
		"select g, min($col collate \"C\"), max($col collate \"C\") from tmo group by g" \
		"select g, pgstrom.min_$ty(p1), pgstrom.max_$ty(p2) from (select g, pgstrom.pmin($col) p1, pgstrom.pmax($col) p2 from tmo) x group by g"
	rps "$col: merge of states with empty ones and NULLs mixed in" \
		"select g, min($col collate \"C\"), max($col collate \"C\") from tmo group by g" \
		"select g, pgstrom.min_$ty(p1), pgstrom.max_$ty(p2) from (select g, pgstrom.pmin($col) p1, pgstrom.pmax($col) p2 from tmo union all select g, ''::bytea, ''::bytea from generate_series(0, 11) g union all select g, null, null from generate_series(0, 11) g) x group by g"
	rps "$col: with a FILTER" \
		"select g, min($col collate \"C\") filter (where n > 20), max($col collate \"C\") filter (where n < 30) from tmo group by g" \
		"select g, pgstrom.min_$ty(p1), pgstrom.max_$ty(p2) from (select g, pgstrom.pmin($col, n > 20) p1, pgstrom.pmax($col, n < 30) p2 from tmo) x group by g"
	rps "$col: one row, no rows" \
		"select (select min($col collate \"C\") from tm1), (select max($col collate \"C\") from tm0)" \
		"select (select pgstrom.min_$ty(pgstrom.pmin($col)) from tm1), (select pgstrom.max_$ty(pgstrom.pmax($col)) from tm0)"
done
rps "big: compressed and out-of-line values" \
	"select g2, min(big collate \"C\"), max(ext collate \"C\") from tmo group by g2" \
	"select g2, pgstrom.min_text(pgstrom.pmin(big)), pgstrom.max_text(pgstrom.pmax(ext)) from tmo group by g2"
# '' is a value, not "no value": a group with only '' and NULL has the minimum '' (not NULL), an all-NULL group has NULL
r=$(q "$CPU" "select g, pgstrom.min_text(pgstrom.pmin(t)) is null, pgstrom.min_text(pgstrom.pmin(t)) = '' from tmo where g in (10, 11) group by g")
c=$(q "$CPU" "select g, min(t collate \"C\") is null, min(t collate \"C\") = '' from tmo where g in (10, 11) group by g")
[ "$r" = "$c" ] && echo "  ok    '' is a value, an all-NULL group is NULL" || { echo "  FAIL  '' versus no value"; echo "        $c"; echo "        $r"; fail=$((fail+1)); }
# bpchar: trailing blanks do not count for the comparison (bpcharcmp); the value returned keeps the blanks it was stored with
r=$($PSQL -d scantest -c "select pgstrom.min_bpchar(pgstrom.pmin(x))::text || '|' || pgstrom.max_bpchar(pgstrom.pmax(x))::text from (values ('a '::bpchar), ('a'::bpchar), ('b'), ('b  ')) v(x)")
c=$($PSQL -d scantest -c "select min(x collate \"C\")::text || '|' || max(x collate \"C\")::text from (values ('a '::bpchar), ('a'::bpchar), ('b'), ('b  ')) v(x)")
[ "$r" = "$c" ] && echo "  ok    bpchar blanks ($r)" || { echo "  FAIL  bpchar blanks: $c vs $r"; fail=$((fail+1)); }
# the filtered partial function with a NULL argument and a true filter (the numeric one used to crash on it)
r=$($PSQL -d scantest -c "select pgstrom.pmin(null::text, true) is null, pgstrom.pmin(null::numeric, true) is null, pgstrom.pmax('a'::text, false) is null")
[ "$r" = "t|t|t" ] && echo "  ok    filtered partial function of NULL" || { echo "  FAIL  filtered partial function of NULL: $r"; fail=$((fail+1)); }

echo "min / max of strings: GPU against the CPU"
GPLAN="Custom Scan (GpuPreAgg)"
rmm() { # label, query-with-%T; results equal to the CPU's; the plan has a GpuPreAgg when MM_EXPECT_GPU=1 (and under no other collation)
	label=$1; tbl=${3:-tmo}; qry=$(printf '%s' "$2" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU $set_extra explain $qry" | grep -c "$GPLAN")
	g=$(q "$GPU $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR && { [ "$MM_EXPECT_GPU" != 1 ] || [ "$plan" -ge 1 ] || [ "$ALLOW_CPU" = on ]; }; then
		echo "  ok    $label (${g#*|} rows, GpuPreAgg=$plan)"
	else echo "  FAIL  $label (GpuPreAgg=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
CMM="set pg_strom.gpupreagg_max_groups=1000000; set pg_strom.gpupreagg_max_groups_heap=2000000;"
for src in tmo tmoa tmop tmog; do
	rmm "$src: ungrouped min / max of t, v" 'select min(t collate "C"), max(t collate "C"), min(v collate "C"), max(v collate "C") from %T' $src
	rmm "$src: ungrouped min / max of a char(10)" 'select min(w collate "C"), max(w collate "C"), count(*) from %T' $src
	rmm "$src: GROUP BY an integer" 'select g, min(t collate "C"), max(t collate "C"), min(v collate "C"), max(w collate "C"), count(*), sum(n) from %T group by g' $src
	rmm "$src: GROUP BY two integers" 'select g, g2, max(t collate "C"), min(w collate "C") from %T group by g, g2' $src
	rmm "$src: many groups" 'select gm, min(t collate "C"), max(v collate "C"), count(*) from %T group by gm' $src "$CMM"
	rmm "$src: WHERE and FILTER" 'select g, min(t collate "C") filter (where n > 20), max(v collate "C") from %T where n < 40 group by g' $src
	rmm "$src: the qual removes every row" 'select min(t collate "C"), max(w collate "C"), count(*) from %T where n > 1000' $src
	rmm "$src: GROUP BY with an empty group" 'select g, min(t collate "C"), max(t collate "C") from %T where g in (10, 11) group by g' $src
done
# heap only: bpchar without a length, compressed and out-of-line values, the one-row and empty tables
rmm "heap: bpchar without a length (::text drops the blanks the ties differ in)" 'select g, min(wn collate "C")::text, max(wn collate "C")::text, count(wn) from %T group by g' tmo
rmm "heap: compressed and out-of-line values" 'select g2, min(big collate "C"), max(ext collate "C") from %T group by g2' tmo
rmm "heap: ungrouped compressed values" 'select min(big collate "C"), max(big collate "C"), max(ext collate "C") from %T' tmo
rmm "heap: one row" 'select min(t collate "C"), max(v collate "C"), min(w collate "C") from %T' tm1
rmm "heap: no rows" 'select min(t collate "C"), max(v collate "C"), max(w collate "C") from %T' tm0
rmm "heap: no rows, GROUP BY" 'select g, min(t collate "C") from %T group by g' tm0

# The collation gate: without COLLATE "C" a database whose default collation is not C must keep min / max of a string on the CPU
# (the ordering is locale based), and an expression is not a column value.
dbc=$($PSQL -d scantest -c "select datcollate from pg_database where datname = 'scantest'")
nogpu() { # label, query: no GpuPreAgg in the plan, results equal to the CPU's
	pl=$($PSQL -d scantest -c "$GPU explain $2" | grep -c "$GPLAN")
	g=$(q "$GPU" "$2"); c=$(q "$CPU" "$2")
	if [ "$pl" -eq 0 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $1 (stays on the CPU)"
	else echo "  FAIL  $1 (GpuPreAgg=$pl)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
if [ "$dbc" != C ] && [ "$dbc" != POSIX ]; then
	nogpu "collation $dbc: min(t) stays on the CPU" "select g, min(t), max(v) from tmo group by g"
else
	echo "  skip  default collation is $dbc: no locale-based ordering to refuse"
fi
nogpu "an expression is not a column: min(upper(t))" 'select g, min(upper(t) collate "C") from tmo group by g'
nogpu "an expression is not a column: min(t || v)" 'select g, max((t || v) collate "C") from tmo group by g'
echo "min / max of strings over a join (stays on the CPU)"
nogpu "min over a join" 'select a.g, min(a.t collate "C") from tmo a join tm1 b on a.g = b.g group by a.g'
