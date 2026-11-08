#!/usr/bin/env python3
# A14 coverage assertion: the corpus of join and GROUP BY shapes and the bookkeeping around it (standard library only;
# run as `python3 -I coverage_shapes.py <command> ...` by coverage_test.sh, which owns the cluster and the tables).
#
# A shape is (id, kind, setting, sql):
#   kind     join     the plan must contain a GpuJoin
#            agg      the plan must contain a GpuPreAgg
#            aggjoin  the plan must contain a GpuPreAgg *and* a join on the GPU (GpuJoin, or the join folded into the GpuPreAgg);
#                     a GpuJoin below a CPU Aggregate is reported as level "partial"
#   setting  the planner knobs that let the GPU path win (names in SETTINGS)
# The sql uses the aliases of the tables created by coverage_test.sh (f = cf, d/e/h... = dimensions).
#
# Commands (DIR is the work directory of the test):
#   list                         the ids, one per line
#   emit-explain DIR             DIR/explain_N.sql (EXPLAIN of every shape, N = chunk), DIR/nchunks
#   classify DIR                 reads DIR/explain_N.out -> DIR/class.tsv (id, level, kind), DIR/gpu_ids.txt, DIR/cpu_ids.txt
#   emit-diag DIR                DIR/diag_N.sql: EXPLAIN with debug2 messages for the shapes that are not on the GPU
#   emit-rows DIR gpu|cpu        DIR/rows_<mode>_N.sql: md5 of the sorted result rows of every shape that is on the GPU
#   report DIR ALLOWLIST         compares everything, prints the summary and the gap table, exit status 1 on a failure
# COVERAGE_ONLY=<regex> restricts the corpus to the ids that match (development).
import fnmatch
import os
import re
import sys

CHUNK = 40

COSTS = "set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;"
# an aggregate over a join: enable_seqscan=off puts a 1e10 penalty into every path and PostgreSQL's cost fuzz then lets a
# sorted CPU GroupAggregate "dominate" the GpuPreAgg path (the comment in joinagg_tests.inc)
COSTS_NOSEQ = "set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1;"
BIGGROUPS = "set pg_strom.gpupreagg_max_groups=1000000;"
JOINSET = "set enable_mergejoin=off; set enable_nestloop=off;"
GISTON = "set pg_strom.enable_gpugistindex=on; set enable_seqscan=on;"
SETTINGS = {
    "join": COSTS + " " + JOINSET,
    "agg": COSTS + " " + BIGGROUPS,
    "aggjoin": COSTS_NOSEQ + " " + BIGGROUPS + " " + JOINSET,
    "default": COSTS,
    "nl": COSTS + " set enable_hashjoin=off; set enable_mergejoin=off;",
    "nlagg": COSTS_NOSEQ + " " + BIGGROUPS + " set enable_hashjoin=off; set enable_mergejoin=off;",
    "gist": COSTS + " " + BIGGROUPS + " " + JOINSET + " " + GISTON,
}
GPU = ("set pg_strom.enabled=on; set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; "
       "set max_parallel_workers_per_gather=0;")
CPU = "set pg_strom.enabled=off; set max_parallel_workers_per_gather=0;"

SHAPES = []


def add(sid, kind, setting, sql):
    SHAPES.append((sid, kind, setting, " ".join(sql.split())))


# ---- types: (name, fact key column, group column), dimension tables have the key columns of the fact table
TYPES = [
    ("int2", "k2", "g2"), ("int4", "k4", "g4"), ("int8", "k8", "g8"), ("float4", "kf4", "gf4"),
    ("float8", "kf8", "gf8"), ("numeric", "kn", "gn"), ("text", "kt", "gt"), ("varchar", "kv", "gv"),
    ("bpchar", "kw", "gw"), ("date", "kd", "gd"), ("timestamp", "kts", "gts"), ("uuid", "ku", "gu"),
    ("bool", "kb", "gb"),
]

# ---------------------------------------------------------------- joins
TYPES_ALL = TYPES
for name, k, g in TYPES:
    dim, lim = ("cdb", "") if name == "bool" else ("cd", "")
    fq = " where f.id <= 2000" if name == "bool" else ""
    fa = " and f.id <= 2000" if name == "bool" else ""
    add("join.key.%s.inner" % name, "join", "join",
        "select f.id, d.dp from cf f join %s d on f.%s = d.%s%s" % (dim, k, k, fq))
    add("join.key.%s.left" % name, "join", "join",
        "select f.id, d.dp from cf f left join %s d on f.%s = d.%s%s" % (dim, k, k, fq))
    add("join.key.%s.semi" % name, "join", "join",
        "select f.id, f.v4 from cf f where exists (select 1 from %s d where d.%s = f.%s)%s" % (dim, k, k, fa))
    add("join.key.%s.anti" % name, "join", "join",
        "select f.id, f.v4 from cf f where not exists (select 1 from %s d where d.%s = f.%s%s)%s" %
        (dim, k, k, "" if name == "bool" else " and d.dg < 3", fa))

# two- and three-column keys
add("join.key.multi2.inner", "join", "join",
    "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 and f.k2 = d.k2")
add("join.key.multi3.inner", "join", "join",
    "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 and f.k2 = d.k2 and f.k8 = d.k8")
add("join.key.multi2.text_int.inner", "join", "join",
    "select f.id, d.dp from cf f join cd d on f.kt = d.kt and f.k4 = d.k4")
add("join.key.expr.inner", "join", "join",
    "select f.id, d.dp from cf f join cd d on f.k4 + 1 = d.k4 + 1")
add("join.key.null_safe.inner", "join", "join",
    "select f.id, d.dp from cf f join cd d on f.k4 is not distinct from d.k4")

# join types with one to five inner relations (hash)
J1 = "cf f join cd d on f.k4 = d.k4"
J2 = J1 + " join cd2 e on f.m2 = e.m2"
J3 = J2 + " join cd3 h on f.m3 = h.m3"
J4 = J3 + " join cd4 i4 on f.m4 = i4.m4"
J5 = J4 + " join cd4 i5 on f.m4 = i5.m4"
L1 = "cf f left join cd d on f.k4 = d.k4"
L2 = L1 + " left join cd2 e on f.m2 = e.m2"
L3 = L2 + " left join cd3 h on f.m3 = h.m3"
L4 = L3 + " left join cd4 i4 on f.m4 = i4.m4"
PJ = {1: "d.dp", 2: "d.dp, e.dp2", 3: "d.dp, e.dp2, h.dp3", 4: "d.dp, e.dp2, h.dp3, i4.dp4", 5: "d.dp, e.dp2, h.dp3, i4.dp4, i5.dp4"}
for n, j in ((1, J1), (2, J2), (3, J3), (4, J4), (5, J5)):
    add("join.type.inner.n%d" % n, "join", "join", "select f.id, %s from %s" % (PJ[n], j))
for n, j in ((1, L1), (2, L2), (3, L3), (4, L4)):
    add("join.type.left.n%d" % n, "join", "join", "select f.id, %s from %s" % (PJ[n], j))
add("join.type.right.n1", "join", "join", "select f.id, x.dp from cf f right join cdx x on f.k4 = x.k4")
add("join.type.right.n2", "join", "join",
    "select f.id, e.dp2, x.dp from cf f join cd2 e on f.m2 = e.m2 right join cdx x on f.k4 = x.k4")
add("join.type.right.n3", "join", "join",
    "select f.id, e.dp2, h.dp3, x.dp from cf f join cd2 e on f.m2 = e.m2 join cd3 h on f.m3 = h.m3 right join cdx x on f.k4 = x.k4")
add("join.type.full.n1", "join", "join", "select f.id, x.dp from cf f full join cdx x on f.k4 = x.k4")
add("join.type.full.n2", "join", "join",
    "select f.id, e.dp2, x.dp from cf f join cd2 e on f.m2 = e.m2 full join cdx x on f.k4 = x.k4")
add("join.type.semi.n1", "join", "join", "select f.id from cf f where exists (select 1 from cd d where d.k4 = f.k4)")
add("join.type.semi.n2", "join", "join",
    "select f.id from cf f join cd2 e on f.m2 = e.m2 where exists (select 1 from cd d where d.k4 = f.k4)")
add("join.type.semi.n3", "join", "join",
    "select f.id from cf f where exists (select 1 from cd d where d.k4 = f.k4) and exists (select 1 from cd2 e where e.m2 = f.m2) "
    "and exists (select 1 from cd3 h where h.m3 = f.m3)")
add("join.type.anti.n1", "join", "join", "select f.id from cf f where not exists (select 1 from cd d where d.k4 = f.k4 and d.dg < 3)")
add("join.type.anti.n2", "join", "join",
    "select f.id from cf f where not exists (select 1 from cd d where d.k4 = f.k4 and d.dg < 3) and not exists (select 1 from cd2 e where e.m2 = f.m2)")
add("join.type.semi_anti.n2", "join", "join",
    "select f.id from cf f where exists (select 1 from cd d where d.k4 = f.k4) and not exists (select 1 from cd2 e where e.m2 = f.m2)")
add("join.type.in_subquery", "join", "join", "select f.id from cf f where f.k4 in (select k4 from cd where dg < 4)")
add("join.type.not_in_subquery", "join", "join", "select f.id from cf f where f.k4 not in (select k4 from cdn)")
add("join.type.join_of_join", "join", "join",
    "select f.id, d.dp, e.dp2 from cf f join cd d on f.k4 = d.k4 join cd2 e on d.dg = e.m2")
add("join.type.extra_qual", "join", "join", "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 and f.v4 > d.dp")
add("join.type.left_on_inner_qual", "join", "join", "select f.id, d.dp from cf f left join cd d on f.k4 = d.k4 and d.dg < 3")
add("join.type.where_on_inner_numeric", "join", "join", "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 where d.kn > 100")
add("join.type.numeric_in_projection", "join", "join", "select f.id, d.kn + f.vn from cf f join cd d on f.k4 = d.k4")
add("join.type.float8_in_projection", "join", "join", "select f.id, d.kf8 + f.vf8 from cf f join cd d on f.k4 = d.k4")
add("join.type.text_in_projection", "join", "join", "select f.id, d.dtx, f.vt from cf f join cd d on f.k4 = d.k4")
add("join.type.text_qual", "join", "join", "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 where f.vt like 't1%' and d.dtx > 'dim5'")
add("join.type.computed_text", "join", "join", "select f.id, upper(d.dtx) from cf f join cd d on f.k4 = d.k4")
add("join.type.udf_in_qual", "join", "join", "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 where cov_udf(f.v4) > 3")
add("join.type.composite_key", "join", "join", "select f.id, d.dp from cf f join cd d on row(f.k4, f.k2) = row(d.k4, d.k2)")
add("join.type.lateral", "join", "join",
    "select f.id, x.dp from cf f, lateral (select d.dp from cd d where d.k4 = f.k4 limit 1) x")
add("join.type.cross_small", "join", "join", "select f.id, b.dp from cf f cross join cdb b where f.id <= 2000")
add("join.type.self_join", "join", "join", "select f.id, g.v4 from cf f join cf g on f.id = g.id + 1")
add("join.type.qual_on_outer", "join", "join", "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 where f.g4 = 1")

# nest-loop shapes: the planner is told to avoid hash and merge joins
add("join.nl.equi.inner", "join", "nl", "select f.id, d.dp from cf f join cd d on f.k4 = d.k4 where f.id <= 3000")
add("join.nl.equi.left", "join", "nl", "select f.id, d.dp from cf f left join cd d on f.k4 = d.k4 where f.id <= 3000")
add("join.nl.equi.semi", "join", "nl", "select f.id from cf f where f.id <= 3000 and exists (select 1 from cd d where d.k4 = f.k4)")
add("join.nl.equi.anti", "join", "nl", "select f.id from cf f where f.id <= 3000 and not exists (select 1 from cd d where d.k4 = f.k4 and d.dg < 3)")
add("join.nl.nonequi.inner", "join", "nl", "select f.id, d.dp from cf f join cd d on f.k4 < d.dp where f.id <= 200 and d.k4 <= 20")
add("join.nl.nonequi.left", "join", "nl", "select f.id, d.dp from cf f left join cd d on f.k4 < d.dp and d.k4 <= 20 where f.id <= 200")
add("join.nl.range.inner", "join", "nl", "select f.id, d.dp from cf f join cd d on f.k4 between d.k4 and d.k4 + 1 where f.id <= 300")
add("join.nl.nonequi.semi", "join", "nl",
    "select f.id from cf f where f.id <= 200 and exists (select 1 from cd d where d.k4 <= 20 and f.k4 < d.dp)")
add("join.nl.nonequi.anti", "join", "nl",
    "select f.id from cf f where f.id <= 200 and not exists (select 1 from cd d where d.k4 <= 20 and f.k4 < d.dp)")
add("join.nl.cross", "join", "nl", "select f.id, b.dp from cf f cross join cdb b where f.id <= 2000")

# ---------------------------------------------------------------- aggregation over a join
for name, k, g in TYPES:
    dim = "cdb" if name == "bool" else "cd"
    grp = "d.dp" if name == "bool" else "d.dg"
    fq = " where f.id <= 2000" if name == "bool" else ""
    add("aggjoin.key.%s" % name, "aggjoin", "aggjoin",
        "select %s, count(*), sum(f.v4) from cf f join %s d on f.%s = d.%s%s group by 1" % (grp, dim, k, k, fq))
add("aggjoin.type.left", "aggjoin", "aggjoin", "select d.dg, count(*), sum(f.v4) from cf f left join cd d on f.k4 = d.k4 group by 1")
add("aggjoin.type.right", "aggjoin", "aggjoin", "select x.dp % 7, count(*), sum(f.v4) from cf f right join cdx x on f.k4 = x.k4 group by 1")
add("aggjoin.type.full", "aggjoin", "aggjoin", "select x.dp % 7, count(*), sum(f.v4) from cf f full join cdx x on f.k4 = x.k4 group by 1")
add("aggjoin.type.semi", "aggjoin", "aggjoin",
    "select f.g4, count(*), sum(f.v4) from cf f where exists (select 1 from cd d where d.k4 = f.k4) group by 1")
add("aggjoin.type.anti", "aggjoin", "aggjoin",
    "select f.g4, count(*), sum(f.v4) from cf f where not exists (select 1 from cd d where d.k4 = f.k4 and d.dg < 3) group by 1")
add("aggjoin.type.n2", "aggjoin", "aggjoin",
    "select d.dg, e.dg2, count(*), sum(f.v4) from cf f join cd d on f.k4 = d.k4 join cd2 e on f.m2 = e.m2 group by 1, 2")
add("aggjoin.type.n3", "aggjoin", "aggjoin",
    "select d.dg, e.dg2, count(*), sum(f.v4) from cf f join cd d on f.k4 = d.k4 join cd2 e on f.m2 = e.m2 join cd3 h on f.m3 = h.m3 group by 1, 2")
add("aggjoin.type.n4", "aggjoin", "aggjoin",
    "select d.dg, count(*), sum(f.v4) from cf f join cd d on f.k4 = d.k4 join cd2 e on f.m2 = e.m2 join cd3 h on f.m3 = h.m3 "
    "join cd4 i4 on f.m4 = i4.m4 group by 1")
add("aggjoin.type.n5", "aggjoin", "aggjoin",
    "select d.dg, count(*), sum(f.v4) from cf f join cd d on f.k4 = d.k4 join cd2 e on f.m2 = e.m2 join cd3 h on f.m3 = h.m3 "
    "join cd4 i4 on f.m4 = i4.m4 join cd4 i5 on f.m4 = i5.m4 group by 1")
add("aggjoin.type.ungrouped", "aggjoin", "aggjoin", "select count(*), sum(f.v4), min(d.dp) from cf f join cd d on f.k4 = d.k4")
add("aggjoin.type.group_fact_key", "aggjoin", "aggjoin", "select f.g4, count(*), sum(d.dp) from cf f join cd d on f.k4 = d.k4 group by 1")
add("aggjoin.type.group_two_keys", "aggjoin", "aggjoin", "select f.g4, d.dg, count(*), sum(d.dp) from cf f join cd d on f.k4 = d.k4 group by 1, 2")
add("aggjoin.type.group_three_keys", "aggjoin", "aggjoin",
    "select f.g4, f.g2, d.dg, count(*), sum(d.dp) from cf f join cd d on f.k4 = d.k4 group by 1, 2, 3")
add("aggjoin.type.group_text_key", "aggjoin", "aggjoin", "select f.gt, count(*), sum(d.dp) from cf f join cd d on f.k4 = d.k4 group by 1")
add("aggjoin.type.sum_numeric", "aggjoin", "aggjoin", "select d.dg, sum(f.vn) from cf f join cd d on f.k4 = d.k4 group by 1")
add("aggjoin.type.sum_float8", "aggjoin", "aggjoin", "select d.dg, sum(f.vf8) from cf f join cd d on f.k4 = d.k4 group by 1")
add("aggjoin.type.min_max_text", "aggjoin", "aggjoin", "select d.dg, min(f.vt), max(d.dtx) from cf f join cd d on f.k4 = d.k4 group by 1")
add("aggjoin.type.nestloop", "aggjoin", "nlagg", "select d.dg, count(*) from cf f join cd d on f.k4 < d.dp where f.id <= 200 and d.k4 <= 20 group by 1")
add("aggjoin.type.count_distinct", "aggjoin", "aggjoin", "select d.dg, count(distinct f.v4) from cf f join cd d on f.k4 = d.k4 group by 1")

# ---------------------------------------------------------------- GROUP BY: keys of each type, one to three keys
for name, k, g in TYPES:
    add("agg.group1.%s" % name, "agg", "agg", "select %s, count(*), sum(v4) from cf group by 1" % g)
for name, k, g in TYPES:
    other = "g2" if name == "int4" else "g4"
    add("agg.group2.%s" % name, "agg", "agg", "select %s, %s, count(*), sum(v4) from cf group by 1, 2" % (other, g))
for name, k, g in TYPES:
    o = {"int2": ("g4", "g8"), "int4": ("g2", "g8")}.get(name, ("g2", "g4"))
    add("agg.group3.%s" % name, "agg", "agg",
        "select %s, %s, %s, count(*), sum(v4) from cf group by 1, 2, 3" % (o[0], o[1], g))
add("agg.group4.int", "agg", "agg", "select g2, g4, g8, m4 % 3, count(*) from cf group by 1, 2, 3, 4")
add("agg.group.expr_int", "agg", "agg", "select v4 % 5, count(*), sum(v8) from cf group by 1")
add("agg.group.expr_date", "agg", "agg", "select extract(year from vd), count(*) from cf group by 1")
add("agg.group.expr_text", "agg", "agg", "select upper(gt), count(*) from cf group by 1")
add("agg.group.expr_case", "agg", "agg", "select case when v4 > 500 then 'hi' else 'lo' end, count(*) from cf group by 1")
add("agg.group.having", "agg", "agg", "select g4, count(*) from cf group by 1 having count(*) > 10 and sum(v4) > 0")
add("agg.group.grouping_sets", "agg", "agg", "select g4, g2, count(*) from cf group by grouping sets ((g4), (g2), ())")
add("agg.group.rollup", "agg", "agg", "select g4, g2, count(*) from cf group by rollup (g4, g2)")
add("agg.group.cube", "agg", "agg", "select g4, g2, count(*) from cf group by cube (g4, g2)")
add("agg.group.only_keys", "agg", "agg", "select g4 from cf group by g4")
add("agg.group.order_limit", "agg", "agg", "select g4, count(*) from cf group by 1 order by 2 desc, 1 limit 3")
add("agg.group.where", "agg", "agg", "select g4, count(*), sum(v8) from cf where v4 > 100 and vt like 't1%' group by 1")
add("agg.group.json_key", "agg", "agg", "select (j->>'a') , count(*) from cj group by 1")
add("agg.group.composite_key", "agg", "agg", "select row(g4, g2), count(*) from cf group by 1")
add("agg.group.array_key", "agg", "agg", "select array[g4, g2], count(*) from cf group by 1")
add("agg.group.interval_key", "agg", "agg", "select g4 * interval '1 hour', count(*) from cf group by 1")

# ---------------------------------------------------------------- aggregate functions (grouped by a 5-group key and ungrouped)
NUMCOLS = ["v2", "v4", "v8", "vf4", "vf8", "vn", "vnu"]
for c in NUMCOLS:
    for fn in ("count", "sum", "avg", "min", "max"):
        add("agg.fn.%s.%s" % (fn, c), "agg", "agg", "select g4, %s(%s) from cf group by 1" % (fn, c))
        add("agg.fn.%s.%s.ungrouped" % (fn, c), "agg", "agg", "select %s(%s) from cf" % (fn, c))
AGGS = [
    ("count_star", "count(*)"),
    ("min_text", "min(vt)"), ("max_text", "max(vt)"),
    ("min_date", "min(vd)"), ("max_date", "max(vd)"),
    ("min_timestamp", "min(vts)"), ("max_timestamp", "max(vts)"),
    ("bool_and", "bool_and(vb)"), ("bool_or", "bool_or(vb)"), ("every", "every(vb)"),
    ("string_agg_ordered", "string_agg(vt, ',' order by vt)"),
    ("string_agg_unordered", "length(string_agg(vt, ','))"),
    ("array_agg_ordered", "array_agg(v4 order by v4)"),
    ("array_agg_unordered", "cardinality(array_agg(v4))"),
    ("count_distinct_int", "count(distinct v4)"), ("sum_distinct_int", "sum(distinct v4)"),
    ("count_distinct_text", "count(distinct vt)"),
    ("count_filter", "count(*) filter (where v4 > 500)"),
    ("sum_filter", "sum(v4) filter (where v4 > 500)"),
    ("avg_filter_float8", "avg(vf8) filter (where vf8 > 3)"),
    ("percentile_cont", "percentile_cont(0.5) within group (order by vf8)"),
    ("percentile_disc", "percentile_disc(0.5) within group (order by v4)"),
    ("mode", "mode() within group (order by v4)"),
    ("stddev_float8", "round(stddev(vf8)::numeric, 6)"), ("stddev_int", "round(stddev_samp(v4)::numeric, 6)"),
    ("variance_int", "round(variance(v4)::numeric, 6)"), ("var_pop_float8", "round(var_pop(vf8)::numeric, 6)"),
    ("corr", "round(corr(vf8, vf4)::numeric, 6)"), ("regr_slope", "round(regr_slope(vf8, vf4)::numeric, 6)"),
    ("covar_pop", "round(covar_pop(vf8, vf4)::numeric, 6)"),
    ("bit_and", "bit_and(v4)"), ("bit_or", "bit_or(v4)"), ("bit_xor", "bit_xor(v4)"),
    ("jsonb_agg", "jsonb_agg(v4 order by v4)"),
    ("sum_expr", "sum(v4 * 2 + v2)"), ("sum_case", "sum(case when v4 > 500 then v4 else 0 end)"),
    ("sum_mixed_int", "sum(v2 + v8)"), ("sum_float_product", "sum(vf8 * vf4)"), ("count_expr", "count(v4 + 1)"),
    ("sum_numeric_expr", "sum(vn * 2)"), ("max_numeric_expr", "max(vn + 1)"),
    ("min_uuid", "min(gu::text)"), ("sum_int2_to_int8", "sum(v2::int8)"),
    ("avg_int", "avg(v4)"), ("count_distinct_date", "count(distinct vd)"),
    ("min_char", "min(gw)"), ("max_varchar", "max(gv)"),
]
for n, e in AGGS:
    add("agg.fn.%s" % n, "agg", "agg", "select g4, %s from cf group by 1" % e)
for n, e in AGGS[:3] + [x for x in AGGS if x[0] in ("bool_and", "string_agg_ordered", "count_distinct_int", "count_filter", "percentile_cont", "stddev_float8", "mode")]:
    add("agg.fn.%s.ungrouped" % n, "agg", "agg", "select %s from cf" % e)
add("agg.fn.multi_mix", "agg", "agg",
    "select g4, count(*), sum(v4), avg(v8), min(vf4), max(vf8), sum(vn), min(vd), max(vts), bool_or(vb), count(distinct v2) from cf group by 1")
add("agg.fn.having_avg", "agg", "agg", "select g4, sum(v4) from cf group by 1 having avg(vf8) > 1")

# ---------------------------------------------------------------- group counts: 5, 500, 50,000, 1,000,000 (default limit and raised)
for gname, gcol in (("5", "g5"), ("500", "g500"), ("50000", "g50k"), ("1000000", "g1m")):
    q = "select %s, count(*), sum(v) from cbig group by 1" % gcol
    add("agg.groups.%s.default" % gname, "agg", "default", q)
    add("agg.groups.%s.raised" % gname, "agg", "agg", q)
add("agg.groups.500.two_keys.raised", "agg", "agg", "select g500, g5, count(*), sum(v) from cbig group by 1, 2")
add("agg.groups.50000.text.raised", "agg", "agg", "select tg, count(*) from cbig group by 1")

# ---------------------------------------------------------------- sources: heap / Arrow / Parquet / GPU-Cache
# column names per source (the same logical columns; Arrow and Parquet have no numeric / float8 / uuid / varchar columns)
SRC = {
    "arrow":    dict(t="a1", id="i", dt="dt", ts="ts", extra=[]),
    "parquet":  dict(t="p1", id="i", dt="dt", ts="ts", extra=[]),
    "gpucache": dict(t="gc1", id="id", dt="g", ts="h",
                     extra=[("varchar", "v"), ("bpchar", "w"), ("uuid", "u")]),
}
for s, m in SRC.items():
    t, idc = m["t"], m["id"]
    p = "src.%s." % s
    add(p + "agg.ungrouped", "agg", "agg", "select count(*), sum(b), min(c), max(c), avg(d) from %s" % t)
    add(p + "agg.group_int_expr", "agg", "agg", "select a %% 5, count(*), sum(b), max(c) from %s group by 1" % t)
    add(p + "agg.group_bool", "agg", "agg", "select e, count(*), sum(b) from %s group by 1" % t)
    add(p + "agg.group_text", "agg", "agg", "select f, count(*) from %s where %s < 100 group by 1" % (t, idc))
    add(p + "agg.group_two_keys", "agg", "agg", "select a %% 5, e, count(*) from %s group by 1, 2" % t)
    add(p + "agg.group_three_keys", "agg", "agg", "select a %% 5, e, b %% 3, count(*) from %s group by 1, 2, 3" % t)
    add(p + "agg.float4_sum", "agg", "agg", "select a %% 5, count(d) from %s group by 1" % t)
    add(p + "join.inner", "join", "join", "select t.%s, d.grp from %s t join sd d on t.b = d.k" % (idc, t))
    add(p + "join.left", "join", "join", "select t.%s, d.grp from %s t left join sd d on t.b = d.k" % (idc, t))
    add(p + "join.right", "join", "join", "select t.%s, d.grp from %s t right join sd d on t.b = d.k" % (idc, t))
    add(p + "join.full", "join", "join", "select t.%s, d.grp from %s t full join sd d on t.b = d.k" % (idc, t))
    add(p + "join.semi", "join", "join", "select t.%s from %s t where exists (select 1 from sd d where d.k = t.b)" % (idc, t))
    add(p + "join.anti", "join", "join", "select t.%s from %s t where not exists (select 1 from sd d where d.k = t.b)" % (idc, t))
    add(p + "join.n2", "join", "join",
        "select t.%s, d.grp, e.g2 from %s t join sd d on t.b = d.k join sd2 e on t.a %% 400 = e.k2" % (idc, t))
    add(p + "join.n4", "join", "join",
        "select t.%s, d.grp, e.g2, h.g3 from %s t join sd d on t.b = d.k join sd2 e on t.a %% 400 = e.k2 "
        "join sd3 h on t.a + 20000 = h.k3 join sd2 e2 on t.a %% 400 = e2.k2" % (idc, t))
    add(p + "join.key_c_int8", "join", "join", "select t.%s, d.grp from %s t join sd d on t.c %% 20000 = d.k8" % (idc, t))
    add(p + "join.key_text", "join", "join", "select t.%s, d.grp from %s t join sd d on t.f = d.txt" % (idc, t))
    add(p + "aggjoin.group_dim", "aggjoin", "aggjoin", "select d.grp, count(*), sum(t.a) from %s t join sd d on t.b = d.k group by 1" % t)
    add(p + "aggjoin.group_left", "aggjoin", "aggjoin", "select d.grp, count(*), sum(t.a) from %s t left join sd d on t.b = d.k group by 1" % t)
    add(p + "aggjoin.ungrouped", "aggjoin", "aggjoin", "select count(*), sum(t.a), max(d.pay) from %s t join sd d on t.b = d.k" % t)
    add(p + "aggjoin.n2", "aggjoin", "aggjoin",
        "select d.grp, e.g2, count(*) from %s t join sd d on t.b = d.k join sd2 e on t.a %% 400 = e.k2 group by 1, 2" % t)
    for tn, col in m["extra"]:
        add(p + "agg.group_" + tn, "agg", "agg", "select %s, count(*), sum(b) from %s group by 1" % (col, t))
    if s == "gpucache":
        add(p + "agg.sum_numeric", "agg", "agg", "select a %% 5, sum(n), min(n), max(n) from %s group by 1" % t)
        add(p + "agg.minmax_float8", "agg", "agg", "select a %% 5, min(x), max(x) from %s group by 1" % t)
        add(p + "agg.sum_float8", "agg", "agg", "select a %% 5, sum(x) from %s group by 1" % t)
        add(p + "join.key_numeric", "join", "join", "select t.id, d.grp from %s t join sdn2 d on t.n = d.kn" % t)
# the Arrow table with float8 / dates / times
add("src.arrow.types.agg_float8", "agg", "agg", "select g, count(*), sum(v), min(x), max(x) from at1 group by 1")
add("src.arrow.types.agg_sum_float8", "agg", "agg", "select g, sum(x) from at1 group by 1")
add("src.parquet.dim.join", "join", "join", "select t.i, d.grp from a1 t join adp d on t.b = d.k")
add("src.arrow.dim.join_parquet_outer", "join", "join", "select t.i, d.grp from p1 t join ad d on t.b = d.k")
# the heap fact table in the same shapes (baseline for the source rows above)
add("src.heap.agg.ungrouped", "agg", "agg", "select count(*), sum(v4), min(v8), max(v8), avg(vf4) from cf")
add("src.heap.join.inner", "join", "join", "select f.id, d.dp from cf f join cd d on f.k4 = d.k4")
add("src.heap.partitioned.agg", "agg", "agg", "select g4, count(*), sum(v4) from cpart group by 1")
add("src.heap.partitioned.join", "join", "join", "select f.id, d.dp from cpart f join cd d on f.k4 = d.k4")
add("src.heap.view.agg", "agg", "agg", "select g4, count(*) from cfv group by 1")
add("src.heap.subquery_agg_join", "aggjoin", "aggjoin",
    "select x.g4, count(*) from (select f.g4, d.dp from cf f join cd d on f.k4 = d.k4) x group by 1")
add("src.heap.cte_agg", "agg", "agg", "with c as (select * from cf where v4 > 10) select g4, count(*) from c group by 1")
add("src.heap.union_all_agg", "agg", "agg", "select g4, count(*) from (select g4 from cf union all select g4 from cf) u group by 1")
add("src.heap.agg_after_distinct", "agg", "agg", "select count(*) from (select distinct g4, g2 from cf) u")


# ---------------------------------------------------------------- the same logical table as an Arrow file, a Parquet file and a GPU-Cache
# af / pf / gcf hold the first 30,000 rows of cf with the same columns (pg2arrow writes the files from the heap table)
for s, t in (("arrow", "af"), ("parquet", "pf"), ("gpucache", "gcf")):
    p = "srcfull.%s." % s
    TYPES = [x for x in TYPES_ALL if not (x[0] == "uuid" and s != "gpucache")]      # pg2arrow cannot write uuid
    for name, k, g in TYPES:
        add(p + "group1.%s" % name, "agg", "agg", "select %s, count(*), sum(v4) from %s group by 1" % (g, t))
    for name, k, g in TYPES:
        if name in ("bool", "text", "float8", "date", "numeric", "uuid"):
            add(p + "group2.%s" % name, "agg", "agg", "select g4, %s, count(*), sum(v4) from %s group by 1, 2" % (g, t))
    add(p + "group3.int2", "agg", "agg", "select g4, g8, g2, count(*), sum(v4) from %s group by 1, 2, 3" % t)
    for c in NUMCOLS:
        if c == "vnu" and s != "gpucache":
            continue          # pg2arrow cannot write a numeric without a declared precision either
        for fn in ("sum", "max"):
            add(p + "fn.%s.%s" % (fn, c), "agg", "agg", "select g4, %s(%s) from %s group by 1" % (fn, c, t))
    add(p + "fn.ungrouped_mix", "agg", "agg", "select count(*), sum(v4), min(v8), max(vf8), avg(vf4) from %s" % t)
    for n, e in AGGS:
        if n in ("min_text", "bool_and", "count_distinct_int", "string_agg_unordered", "count_filter"):
            add(p + "fn." + n, "agg", "agg", "select g4, %s from %s group by 1" % (e, t))
    for name, k, g in TYPES:
        dim = "cdb" if name == "bool" else "cd"
        fq = " where f.id <= 2000" if name == "bool" else ""
        add(p + "join.key.%s.inner" % name, "join", "join",
            "select f.id, d.dp from %s f join %s d on f.%s = d.%s%s" % (t, dim, k, k, fq))
    for name, k, g in TYPES:
        if name in ("int4", "text", "date"):
            add(p + "join.key.%s.left" % name, "join", "join",
                "select f.id, d.dp from %s f left join cd d on f.%s = d.%s" % (t, k, k))
    add(p + "join.semi", "join", "join", "select f.id from %s f where exists (select 1 from cd d where d.k4 = f.k4)" % t)
    add(p + "join.n2", "join", "join",
        "select f.id, d.dp, e.dp2 from %s f join cd d on f.k4 = d.k4 join cd2 e on f.m2 = e.m2" % t)
    add(p + "aggjoin.int4", "aggjoin", "aggjoin",
        "select d.dg, count(*), sum(f.v4) from %s f join cd d on f.k4 = d.k4 group by 1" % t)
    add(p + "aggjoin.left", "aggjoin", "aggjoin",
        "select d.dg, count(*), sum(f.v4) from %s f left join cd d on f.k4 = d.k4 group by 1" % t)
    add(p + "aggjoin.group_fact_key", "aggjoin", "aggjoin",
        "select f.g4, count(*), sum(d.dp) from %s f join cd d on f.k4 = d.k4 group by 1" % t)

# ---------------------------------------------------------------- spatial (GiST) shapes, PostGIS
PTS = "st_makepoint(o.x, o.y)"
for sid, kind, setting, sql in [
    ("join", "join", "gist", "select o.id, i.id from pg_gp o join pg_gz i on i.geom && %s" % PTS),
    ("join.left", "join", "gist", "select o.id, i.id from pg_gp o left join pg_gz i on i.geom && %s" % PTS),
    ("join.right", "join", "gist", "select o.id, i.id from pg_gp o right join pg_gz i on i.geom && %s" % PTS),
    ("join.semi", "join", "gist", "select o.id from pg_gp o where exists (select 1 from pg_gz i where i.geom && %s)" % PTS),
    ("join.anti", "join", "gist", "select o.id from pg_gp o where not exists (select 1 from pg_gz i where i.geom && %s)" % PTS),
    ("join.dwithin", "join", "gist", "select o.id, i.id from pg_gp o join pg_gz i on st_dwithin(i.geom, %s, 2.5)" % PTS),
    ("join.contains", "join", "gist", "select o.id, i.id from pg_gp o join pg_gz i on st_contains(i.geom, %s)" % PTS),
    ("join.within", "join", "gist", "select o.id, i.id from pg_gp o join pg_gz i on st_within(%s, i.geom)" % PTS),
    ("join.intersects", "join", "gist", "select o.id, i.id from pg_gp o join pg_gz i on st_intersects(i.geom, %s)" % PTS),
    ("join.n2", "join", "gist",
     "select o.id, i.id, d.grp from pg_gp o join pg_gz i on i.geom && %s join sd d on d.k = i.a" % PTS),
    ("join.box_outer", "join", "gist", "select o.id, i.id from pg_gb o join pg_gz i on i.geom && o.geom"),
    ("join.box_intersects_geom_outer", "join", "gist", "select o.id, i.id from pg_gb o join pg_gz i on st_intersects(i.geom, o.geom)"),
    ("join.distance_no_index", "join", "gist", "select o.id, i.id from pg_gp o join pg_gz i on st_distance(i.geom, %s) < 1.5 where o.id < 300" % PTS),
    ("join.nl_no_index", "join", "nl", "select o.id, i.id from pg_gp o join pg_gn i on i.geom && %s where o.id < 300" % PTS),
    ("join.geometry_equality", "join", "gist", "select o.id, i.id from pg_gb o join pg_gb2 i on o.geom = i.geom"),
    ("join.gist_plus_hash", "join", "gist",
     "select o.id, i.id from pg_gp o join pg_gz i on i.geom && %s and i.b = o.id" % PTS),
]:
    add("spatial." + sid, kind, setting, sql)
for sid, sql in [
    ("agg.count_per_outer", "select o.id, count(i.id) from pg_gp o join pg_gz i on i.geom && %s group by o.id" % PTS),
    ("agg.group_inner_attr", "select i.a, count(*) from pg_gp o join pg_gz i on i.geom && %s group by i.a" % PTS),
    ("agg.ungrouped", "select count(*), sum(i.a) from pg_gp o join pg_gz i on i.geom && %s" % PTS),
    ("agg.dwithin", "select i.a, count(*) from pg_gp o join pg_gz i on st_dwithin(i.geom, %s, 2.5) group by i.a" % PTS),
    ("agg.left", "select i.a, count(*) from pg_gp o left join pg_gz i on i.geom && %s group by i.a" % PTS),
    ("agg.semi", "select o.id %% 7, count(*) from pg_gp o where exists (select 1 from pg_gz i where i.geom && %s) group by 1" % PTS),
    ("agg.sum_distance", "select i.a, sum(st_distance(i.geom, %s)) from pg_gp o join pg_gz i on i.geom && %s group by i.a" % (PTS, PTS)),
]:
    add("spatial." + sid, "aggjoin", "gist", sql)
add("spatial.agg.group_geometry", "agg", "agg", "select geom, count(*) from pg_gz group by 1")
add("spatial.agg.st_extent", "agg", "agg", "select a, st_extent(geom) from pg_gz group by 1")
add("spatial.agg.st_union", "agg", "agg", "select a, st_astext(st_union(geom)) from pg_gz where a < 2 and geom is not null group by 1")
add("spatial.agg.count_filter_bbox", "agg", "agg", "select a, count(*) from pg_gz where geom && st_makeenvelope(0, 0, 50, 50) group by 1")
add("spatial.agg.group_st_x", "agg", "agg", "select round(st_x(geom)::numeric, 0), count(*) from pg_gpt group by 1")


def selected():
    pat = os.environ.get("COVERAGE_ONLY")
    return [s for s in SHAPES if not pat or re.search(pat, s[0])]


# ---------------------------------------------------------------- commands
def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def chunks(shapes):
    return [shapes[i:i + CHUNK] for i in range(0, len(shapes), CHUNK)]


def prefix(setting):
    return "reset all; " + GPU + " " + SETTINGS[setting] + "\n"


def cmd_emit_explain(d):
    ch = chunks(selected())
    for n, c in enumerate(ch):
        out = []
        for sid, kind, setting, sql in c:
            out.append("\\echo @@ %s\n%sexplain (costs off) %s;\n" % (sid, prefix(setting), sql))
        write("%s/explain_%d.sql" % (d, n), "".join(out))
    write(d + "/nchunks", str(len(ch)))


def read_blocks(path):
    """file with '@@ id' marker lines -> {id: [lines]}"""
    blocks, cur = {}, None
    if not os.path.exists(path):
        return blocks
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("@@ "):
                cur = line[3:].strip()
                blocks[cur] = []
            elif cur is not None:
                blocks[cur].append(line)
    return blocks


def read_all_blocks(d, prefix_name):
    blocks = {}
    n = 0
    while os.path.exists("%s/%s_%d.sql" % (d, prefix_name, n)) or os.path.exists("%s/%s_%d.out" % (d, prefix_name, n)):
        blocks.update(read_blocks("%s/%s_%d.out" % (d, prefix_name, n)))
        n += 1
    return blocks


def level_of(kind, plan):
    text = "\n".join(plan)
    if re.search(r"^(ERROR|psql:.*ERROR)", text, re.M) or "ERROR:" in text:
        return "error"
    join_on_gpu = "Custom Scan (GpuJoin)" in text or re.search(r"GPU ([A-Za-z]+ )*(Join Quals|GiST Join) \[\d+\]", text) is not None
    agg_on_gpu = "Custom Scan (GpuPreAgg)" in text
    if kind == "join":
        return "gpu" if join_on_gpu else "cpu"
    if kind == "agg":
        return "gpu" if agg_on_gpu else "cpu"
    # aggjoin
    if agg_on_gpu and join_on_gpu:
        return "gpu"
    if agg_on_gpu:
        return "partial-agg"      # GpuPreAgg over a CPU join
    if join_on_gpu:
        return "partial-join"     # GpuJoin below a CPU aggregate
    return "cpu"


def cmd_classify(d):
    blocks = read_all_blocks(d, "explain")
    gpu, cpu, rows = [], [], []
    for sid, kind, setting, sql in selected():
        plan = blocks.get(sid)
        lvl = "missing" if plan is None else level_of(kind, plan)
        rows.append("%s\t%s\t%s" % (sid, lvl, kind))
        (gpu if lvl == "gpu" else cpu).append(sid)
    write(d + "/class.tsv", "\n".join(rows) + "\n")
    write(d + "/gpu_ids.txt", "\n".join(gpu) + "\n")
    write(d + "/cpu_ids.txt", "\n".join(cpu) + "\n")


def ids_of(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(l.strip() for l in f if l.strip())


def cmd_emit_diag(d):
    cpu = ids_of(d + "/cpu_ids.txt")
    ch = chunks([s for s in selected() if s[0] in cpu])
    for n, c in enumerate(ch):
        out = []
        for sid, kind, setting, sql in c:
            out.append("\\echo @@ %s\n%sset client_min_messages=debug2;\nexplain (costs off) %s;\n" % (sid, prefix(setting), sql))
        write("%s/diag_%d.sql" % (d, n), "".join(out))


def cmd_emit_rows(d, mode):
    gpu = ids_of(d + "/gpu_ids.txt")
    ch = chunks([s for s in selected() if s[0] in gpu])
    for n, c in enumerate(ch):
        out = ["\\timing on\n"]
        for sid, kind, setting, sql in c:
            pre = prefix(setting) if mode == "gpu" else "reset all; " + CPU + "\n"
            out.append("\\echo @@ %s\n%sselect md5(coalesce(string_agg(x::text, E'\\n' order by x::text), '')), count(*) from (%s) x;\n"
                       % (sid, pre, sql))
        write("%s/rows_%s_%d.sql" % (d, mode, n), "".join(out))


def load_allow(path):
    allow = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3 or not re.match(r"^(by-design|gap:[A-Za-z0-9_.+/-]+)$", parts[2]):
                sys.exit("%s:%d: format is 'shape-id | reason | by-design|gap:<milestone>'" % (path, ln))
            allow.append((parts[0], parts[1], parts[2]))
    return allow


DIAG_NOISE = re.compile(r"(gpusort: disabled|trying on top of|JIT provider|disabling JIT|rehashing catalog|StartTransaction|CommitTransaction|name: unnamed|ProcessUtility|StartTransactionCommand|CommitTransactionCommand|"
                        r"parse |rewrite|planner|pg_stat|EXPLAIN|shmem|proc_exit|InitPostgres|autovac|relation .* does not exist)", re.I)


def diag_lines(blocks, sid):
    out = []
    for l in blocks.get(sid, []):
        if "DEBUG" in l and not DIAG_NOISE.search(l):
            m = re.sub(r"^.*DEBUG:\s*", "", l).strip()
            if m and m not in out:
                out.append(m)
    return out


def cmd_report(d, allow_path):
    shapes = selected()
    allow = load_allow(allow_path)
    cls = {}
    for line in open(d + "/class.tsv"):
        sid, lvl, kind = line.rstrip("\n").split("\t")
        cls[sid] = (lvl, kind)
    diag = read_all_blocks(d, "diag")
    rg = read_all_blocks(d, "rows_gpu")
    rc = read_all_blocks(d, "rows_cpu")
    fails = []
    times = []

    def match(sid):
        for pat, reason, kind in allow:
            if fnmatch.fnmatchcase(sid, pat):
                return pat, reason, kind
        return None

    used = set()
    ongpu = 0
    listed = {}
    for sid, kind, setting, sql in shapes:
        lvl, _ = cls.get(sid, ("missing", kind))
        m = match(sid)
        if m:
            used.add(m[0])
        if lvl == "error":
            fails.append("%s: EXPLAIN failed: %s" % (sid, " ".join(l for l in [x for x in read_all_blocks(d, 'explain').get(sid, [])] if 'ERROR' in l)[:200]))
            continue
        if lvl == "missing":
            fails.append("%s: no EXPLAIN output" % sid)
            continue
        if lvl == "gpu":
            ongpu += 1
            if m:
                fails.append("%s: listed in the allowlist (%s) but it reaches the GPU now: delete the line" % (sid, m[0]))
            if os.environ.get("COVERAGE_ROWS", "1") == "0":
                continue
            for blk in (rg, rc):
                for l in blk.get(sid, []):
                    m = re.match(r"^Time: ([0-9.]+) ms", l)
                    if m:
                        times.append((float(m.group(1)) / 1000.0, "gpu" if blk is rg else "cpu", sid))
            g = [l for l in rg.get(sid, []) if l.strip() and not l.startswith("Time:")]
            c = [l for l in rc.get(sid, []) if l.strip() and not l.startswith("Time:")]
            gl = g[-1] if g else "(no output)"
            cl = c[-1] if c else "(no output)"
            if gl != cl or "ERROR" in gl or not re.match(r"^[0-9a-f]{32}\|\d+$", gl):
                fails.append("%s: rows differ or failed\n      gpu: %s\n      cpu: %s\n      sql: %s" % (sid, " / ".join(g)[:300] or gl, cl[:200], sql))
        else:
            if not m:
                fails.append("%s: stays on the CPU (%s) and is not in the allowlist\n      sql: %s\n      why: %s" %
                             (sid, lvl, sql, "; ".join(diag_lines(diag, sid))[:300] or "(no debug message)"))
            else:
                listed.setdefault((m[1], m[2]), []).append(sid)
    for pat, reason, kind in allow:
        if pat not in used:
            fails.append("allowlist line '%s' matches no shape in this corpus: delete or fix it" % pat)

    with open(d + "/cpu_shapes.tsv", "w") as f:
        for sid, kind, setting, sql in shapes:
            lvl = cls.get(sid, ("missing", kind))[0]
            if lvl != "gpu":
                f.write("%s\t%s\t%s\t%s\n" % (sid, lvl, " ; ".join(diag_lines(diag, sid))[:240], sql))
    total = len(shapes)
    nlisted = sum(len(v) for v in listed.values())
    bydesign = sum(len(v) for (r, k), v in listed.items() if k == "by-design")
    gaps = {}
    for (r, k), v in listed.items():
        if k.startswith("gap:"):
            gaps.setdefault(k[4:], []).append(len(v))
    lines = []
    lines.append("")
    lines.append("coverage: %d shapes, %d on the GPU, %d on the CPU (allowlisted: %d by-design, %d gap)" %
                 (total, ongpu, total - ongpu, bydesign, nlisted - bydesign))
    lines.append("")
    tot = {}
    for (r, k), v in listed.items():
        tot[k] = tot.get(k, 0) + len(v)
    lines.append("blocked shapes by kind: " + ", ".join("%s %d" % (k, n) for k, n in sorted(tot.items(), key=lambda x: (-x[1], x[0]))))
    lines.append("")
    lines.append("gaps by reason (shapes blocked, milestone, reason):")
    rows = sorted(((len(v), k, r) for (r, k), v in listed.items()), key=lambda x: (-x[0], x[2]))
    for n, k, r in rows:
        lines.append("  %4d  %-14s  %s" % (n, k, r))
    lines.append("")
    summary = "\n".join(lines)
    print(summary)
    times.sort(reverse=True)
    if times:
        print("slowest shapes (s): " + ", ".join("%s %s %.1f" % (sid, m, t) for t, m, sid in times[:8]))
        print("total query time (s): gpu %.0f, cpu %.0f" % (sum(t for t, m, _ in times if m == "gpu"), sum(t for t, m, _ in times if m == "cpu")))
    write(d + "/report.txt", summary + "\n\nlisted shapes:\n" +
          "\n".join("  [%s] %s: %s" % (k, r, ", ".join(v)) for (r, k), v in sorted(listed.items(), key=lambda x: -len(x[1]))) + "\n")
    if fails:
        print("FAILURES (%d):" % len(fails))
        for f in fails:
            print("  FAIL  " + f)
        return 1
    print("COVERAGE: ALL PASSED")
    return 0


def main():
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    if c == "list":
        for s in selected():
            print(s[0])
    elif c == "emit-explain":
        cmd_emit_explain(sys.argv[2])
    elif c == "classify":
        cmd_classify(sys.argv[2])
    elif c == "emit-diag":
        cmd_emit_diag(sys.argv[2])
    elif c == "emit-rows":
        cmd_emit_rows(sys.argv[2], sys.argv[3])
    elif c == "report":
        sys.exit(cmd_report(sys.argv[2], sys.argv[3]))
    else:
        sys.exit("usage: coverage_shapes.py list|emit-explain|classify|emit-diag|emit-rows|report (see the header)")


if __name__ == "__main__":
    main()
