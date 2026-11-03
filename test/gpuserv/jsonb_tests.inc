# A8: jsonb on the GPU. Sourced by scan_test.sh (it provides run, run_any, runagg, runagg_any, q, check, sql,
# $PSQL, $GPU, $CPU, $fail). New tables only (jb_*).
#
# A jsonb register points at the container inside the heap page; the VM walks the binary form (keys ordered
# by length, then bytes; the offset of every 32nd entry stored as an end offset). What it cannot decide exactly
# (a number that does not fit { int64, scale }, text of a number or a container that must be formatted,
# a value stored compressed or out of line) is evaluated by the backend ("CPU re-evaluation of N tuples" with
# client_min_messages = debug1). Results are compared as text with the CPU's.
echo "jsonb (A8)"
JBROWS=${JBROWS:-200000}

# One document per row, 21 shapes (i % 100 picks the big ones): objects with 40 keys (the stride of 32 offsets),
# nested objects, arrays (also with more than 32 elements), raw scalars (number, string, true, null), empty
# containers, escapes and UTF-8, numbers with exponents/fractions/more than 64 bits, equal-length and very long
# keys, strings that look like numbers, and documents over 2 KB (compressed: repeats; out of line: md5 chains).
# Every document that has "num" has an integer there, "dec" a number with 2 decimals, so that casts do not fail.
$PSQL -d scantest >/dev/null <<'SQL'
create function jb_doc(i int) returns jsonb language sql immutable as $$
select case
 when i % 100 = 14 then jsonb_build_object('t', repeat('compressible text ', 200), 'k', i % 7, 'num', i % 1000 - 500)
 when i % 100 = 15 then jsonb_build_object('t', (select string_agg(md5((i * k)::text), '') from generate_series(1, 100) k), 'k', i % 7)
 else case i % 20
 when 0 then null
 when 1 then format('{"a":%s,"b":"s%s","c":%s,"d":null,"e":%s,"f":[1,2,%s],"g":{"x":"y%s","z":%s},"num":%s,"dec":%s}',
                    i, i % 100, (i % 2 = 0)::text, (i % 1000) / 10.0, i % 7, i % 13, i % 5, i % 1000 - 500, (i % 20000) / 100.0)::jsonb
 when 2 then format('[%s,"s%s",null,%s,%s,{"a":%s},[1,%s]]', i, i % 50, (i % 3 = 0)::text, (i % 100) / 4.0, i % 9, i % 4)::jsonb
 when 3 then to_jsonb((i % 1000) / 8.0)
 when 4 then to_jsonb('str' || (i % 30))
 when 5 then 'null'::jsonb
 when 6 then to_jsonb(i % 2 = 0)
 when 7 then '{}'::jsonb
 when 8 then '[]'::jsonb
 when 9 then jsonb_build_object('s', E'he said "hi"\n\t é 日本語' || (i % 10), 'name', 'héllo' || (i % 5), 'q', E'back\\slash', 'num', i % 50)
 when 10 then format('{"n1":1e3,"n2":-0.000001,"n3":12345678901234567890,"n4":1.0,"n5":1.50,"n6":9223372036854775807,"n7":0,"n8":-0,"n9":3.14159265358979323846,"n10":%s,"n11":%s,"num":%s}', i, (i % 5000) - 2500, i % 1000)::jsonb
 when 11 then (select jsonb_object_agg('k' || lpad(n::text, 2, '0'), (n * i) % 97) from generate_series(1, 40) n) || jsonb_build_object('num', i % 33)
 when 12 then jsonb_build_object('aa', 1, 'ab', i % 3, 'b', 3, 'ba', 4, 'zzz', 5, repeat('L', 100), i % 4, repeat('L', 99) || 'M', i % 6)
 when 13 then format('{"a":{"b":{"c":{"d":{"e":%s}}}}}', i % 10)::jsonb
 when 14 then jsonb_build_array(i, i + 1, i + 2)
 when 15 then jsonb_build_object('arr', (select jsonb_agg(n * (i % 5)) from generate_series(1, 40) n), 'o', '{"p":1}'::jsonb, 'num', 7)
 when 16 then format('{"a":"%s","b":%s,"num":%s,"snum":"%s"}', i % 11, i % 11, i % 11, i % 11)::jsonb
 when 17 then format('["s%s","s%s",%s,true]', i % 5, i % 7, i % 3)::jsonb
 else format('{"a":%s,"b":"s%s","num":%s,"dec":%s}', i % 100, i % 100, i % 1000 - 500, (i % 9000) / 100.0)::jsonb
 end
end $$;
SQL
jb_create() { # table name
	sql "create table $1 as select i::int4 as i, (i % 7)::int4 as g, jb_doc(i) as d from generate_series(1, $JBROWS) i"
}
jb_create jb1; sql "vacuum analyze jb1"                                  # all-visible pages
jb_create jb2; sql "update jb2 set g = g + 1 where i % 5 = 0"; sql "delete from jb2 where i % 13 = 0"; sql "analyze jb2"
n=$($PSQL -d scantest -c "select count(*) from jb1 where pg_column_compression(d) is not null")
[ "$n" -gt 0 ]; check "the data has compressed jsonb values ($n rows)" $?
toid=$($PSQL -d scantest -c "select oid from pg_class where relname = 'jb1'")
n=$($PSQL -d scantest -c "select count(distinct chunk_id) from pg_toast.pg_toast_$toid")
[ "$n" -gt 0 ]; check "the data has jsonb stored out of line ($n toast values)" $?

for t in jb1 jb2; do
	echo "table $t"
	echo "  -- -> and ->> with constant keys"
	run $t "->> key = text constant" "select i, d from %T where d->>'b' = 's21'"
	run $t "->> key <> text constant (numbers there are deferred text)" "select i, d from %T where d->>'b' <> 's21'"
	run $t "->> key LIKE (prefix, suffix, substring)" "select i, d from %T where d->>'b' like 's1%' or d->>'b' like '%7' or d->>'name' like '%llo2%'"
	run $t "-> key IS NULL / IS NOT NULL" "select i, d from %T where d->'b' is null"
	run $t "-> key IS NOT NULL" "select i, d from %T where d->'d' is not null"
	run $t "->> IS NULL (a JSON null is SQL NULL, a number is not)" "select i, d from %T where d->>'d' is null and d->>'b' is not null"
	run $t "->> IS NOT NULL on numbers and containers" "select i from %T where d->>'g' is not null or d->>'n1' is not null"
	run $t "array element ->> index" "select i, d from %T where d->>1 = 's5' or d->>0 = 'str4' or d->>1 = 's4'"
	run $t "array element negative index" "select i, d from %T where d->>-1 = 'true' or d->>-2 = 's3' or d->-6 is not null"
	run $t "array element -> index IS NULL" "select i from %T where d->3 is null and d->0 is not null"
	run $t "index past the end" "select i from %T where d->>50 is null and d->>1 = 's4'"
	run $t "chain ->'g'->>'x'" "select i, d from %T where d->'g'->>'x' = 'y3'"
	run $t "chain through arrays and objects" "select i, d from %T where d->2->>'a' = '3' or d->'f'->>1 = '2' or d->'f'->2 is not null"
	run $t "deep chain" "select i, d from %T where d->'a'->'b'->'c'->'d'->>'e' = '3'"
	run $t "->> on a raw scalar (index 0)" "select i, d from %T where d->>0 = 'str5' or d->>0 = 'true'"
	run $t "wrong container for the operator" "select i from %T where d->'0' is not null or d->>0 = '5'"
	echo "  -- numbers (text form, casts)"
	run $t "(->>)::int > constant" "select i, d from %T where (d->>'num')::int > 100"
	run $t "(->>)::int2 and ::int8" "select i from %T where (d->>'num')::int2 < -490 or (d->>'num')::int8 = 7"
	run $t "(->>)::numeric compare" "select i from %T where (d->>'dec')::numeric > 50.25 and (d->>'dec')::numeric <= 150"
	run $t "numeric equality across scales" "select i from %T where (d->>'dec')::numeric = 1.5 or (d->>'n5')::numeric = 1.5 or (d->>'n4')::numeric = 1"
	run $t "numbers with exponent, fraction, zero, -0" "select i from %T where (d->>'n1')::numeric = 1000 or (d->>'n2')::numeric < 0 or (d->>'n7')::numeric = 0 or (d->>'n8')::numeric = 0"
	run $t "numbers beyond 64 bits (rechecked) and the int8 maximum" "select i from %T where (d->>'n3')::numeric > 1 or (d->>'n6')::numeric > 9223372036854775806 or (d->>'n9')::numeric > 3"
	run $t "array element cast ->> index" "select i from %T where (d->>2)::numeric > 20 or (d->>2)::int = 1"
	run $t "chain then cast" "select i from %T where (d->'g'->>'z')::int = 1 or (d->'a'->'b'->'c'->'d'->>'e')::int4 > 2"
	run $t "string that looks like a number cast to int (rechecked)" "select i from %T where (d->>'snum')::int > 5 or (d->>'snum')::numeric = 3"
	run $t "cast of a JSON null is SQL NULL" "select i from %T where (d->>'d')::int is null and d->'d' is not null"
	run $t "arithmetic on casts" "select i from %T where (d->>'num')::int * 2 + 1 > 500 or (d->>'dec')::numeric * 2 > 300"
	run $t "string number equality (->> text vs constant)" "select i from %T where d->>'n10' = '5' or d->>'b' = '5' or d->>'b' = 's41'"
	echo "  -- ?, jsonb_typeof, @>"
	run $t "? key of an object" "select i, d from %T where d ? 'a'"
	run $t "? key past 32 entries" "select i, d from %T where d ? 'k17' or d ? 'k39' or d ? 'k00' or d ? 'k40'"
	run $t "? key of different lengths (length-then-bytes order)" "select i from %T where d ? 'b' or d ? 'ba' or d ? 'aa' or d ? 'zzz' or d ? 'ab'"
	run $t "? string element of an array" "select i, d from %T where d ? 's5' or d ? 's3'"
	run $t "? on a raw scalar string" "select i from %T where d ? 'str4' or d ? 'str24' or d ? 'null'"
	run $t "? on a nested value" "select i from %T where d->'g' ? 'x' or (d->'o') ? 'p'"
	run $t "? long keys" "select i from %T where (d ? repeat('L', 100)) or (d ? (repeat('L', 99) || 'M')) or (d ? repeat('L', 99))"
	run $t "jsonb_typeof of the column" "select i, jsonb_typeof(d) from %T where jsonb_typeof(d) = 'object' or jsonb_typeof(d) = 'number'"
	run $t "jsonb_typeof of scalars and arrays" "select i from %T where jsonb_typeof(d) = 'string' or jsonb_typeof(d) = 'boolean' or jsonb_typeof(d) = 'null' or jsonb_typeof(d) = 'array'"
	run $t "jsonb_typeof of members" "select i from %T where jsonb_typeof(d->'b') = 'string' or jsonb_typeof(d->'g') = 'object' or jsonb_typeof(d->'f') = 'array' or jsonb_typeof(d->'d') = 'null' or jsonb_typeof(d->'c') = 'boolean'"
	run $t "jsonb_typeof of a number member" "select i from %T where jsonb_typeof(d->'dec') = 'number' and d ? 'b'"
	run $t "@> object with a number" "select i, d from %T where d @> '{\"a\":1}'"
	run $t "@> numeric equality of 1 and 1.0" "select i from %T where d @> '{\"a\":1.0}' or d @> '{\"n4\":1}'"
	run $t "@> object with a string and a bool" "select i from %T where d @> '{\"b\":\"s21\"}' or d @> '{\"c\":true}'"
	run $t "@> JSON null" "select i from %T where d @> '{\"d\":null}'"
	run $t "@> nested object" "select i from %T where d @> '{\"g\":{\"x\":\"y3\"}}' or d @> '{\"a\":{\"b\":{\"c\":{\"d\":{\"e\":5}}}}}'"
	run $t "@> array member of an object" "select i from %T where d @> '{\"f\":[1,2]}' or d @> '{\"f\":[]}' or d @> '{\"arr\":[0,5]}'"
	run $t "@> several keys" "select i from %T where d @> '{\"a\":5,\"b\":\"s5\"}' or d @> '{\"num\":7,\"o\":{}}'"
	run $t "@> empty object" "select i, d from %T where d @> '{}'"
	run_any $t "@> empty array (not exact on the device: stays on the CPU)" "select i, d from %T where d @> '[]'"
	run $t "@> root array of scalars" "select i, d from %T where d @> '[\"s5\"]' or d @> '[\"s1\",true]' or d @> '[null]'"
	run $t "@> on a nested value" "select i from %T where d->'g' @> '{\"z\":1}' or d->5 @> '{\"a\":3}'"
	run $t "@> and a scalar document" "select i from %T where d @> '{\"a\":1}' and jsonb_typeof(d) = 'number'"
	run $t "@> and ->> quals together" "select i, d from %T where d @> '{\"c\":false}' and d->>'b' like 's1%' and i % 3 = 0"
	echo "  -- aggregates"
	runagg $t "count and sum of casts" "select count(*), sum((d->>'num')::int), max((d->>'num')::int), min((d->>'num')::int2) from %T"
	runagg_any $t "sum of numeric values (a scale that depends on the row stays on the CPU)" "select sum((d->>'dec')::numeric), count((d->>'dec')::numeric) from %T where d ? 'dec'"
	runagg $t "aggregate with a jsonb qual" "select count(*), sum(i) from %T where d @> '{\"c\":true}' or d ? 'k17'"
	runagg $t "GROUP BY an ordinary key, jsonb filter" "select g, count(*), sum((d->>'num')::int) from %T where d ? 'num' group by g"
	runagg_any $t "GROUP BY a key computed from jsonb (many groups: overflow on the service CPU path)" "select count(*), sum(k), sum(c), sum(s) from (select (d->>'num')::int as k, count(*) as c, sum(i) as s from %T where d ? 'num' and jsonb_typeof(d) = 'object' group by (d->>'num')::int) x"
	runagg $t "GROUP BY with jsonb qual and aggregate argument" "select count(*), sum(g), sum(c), sum(m) from (select g, count(*) as c, max((d->>'num')::int8) as m from %T where d->>'b' like 's1%' group by g) x"
	echo "  -- stays on the CPU (same results)"
	run_any $t "jsonb value in the projection" "select i, d->'g', d->>'b', d->0 from %T where d ? 'a'"
	run_any $t "float casts" "select i from %T where (d->>'dec')::float8 > 100 or (d->>'dec')::float4 > 100"
	run_any $t "jsonb equality" "select i from %T where d = '{}' or d->'g' = '{\"x\":\"y3\",\"z\":3}'"
	run_any $t "@> an array of arrays (not exact on the device)" "select i from %T where d @> '[[1]]' or d @> '{\"a\":[[1]]}'"
	run_any $t "@> a scalar pattern" "select i from %T where d @> '\"s5\"' or d @> '5'"
	run_any $t "jsonb_array_length / jsonb_object_keys-free functions" "select i from %T where jsonb_array_length(case when jsonb_typeof(d) = 'array' then d else '[]' end) > 3"
	run_any $t "?| and ?&" "select i from %T where d ?| array['a','b'] or d ?& array['a','c']"
	run_any $t "-> text array path #>" "select i from %T where d #>> '{g,x}' = 'y3'"
	run_any $t "IS DISTINCT FROM" "select i from %T where d->>'b' is distinct from 's5'"
	run_any $t "COALESCE of ->>" "select i from %T where coalesce(d->>'b', 'none') = 'none'"
done

# values stored compressed or out of line are evaluated by the backend (the column is read before any qual runs)
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from jb1 where d->>'k' = '3' or d ? 'k17'" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -ge 1 ]; check "toasted jsonb is re-evaluated by the backend (debug message seen $msgs times)" $?
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from jb1 where d->>'num' = '1' and i % 100 < 14 and i % 20 = 1" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -ge 1 ]; check "the text of a number is formatted by the backend: rechecked ($msgs)" $?
plan=$($PSQL -d scantest -c "$GPU explain select i from jb1 where d ? 'a' and (d->>'num')::int > 5 and d @> '{\"c\":true}'" | grep -c "GpuScan")
[ "$plan" -ge 1 ]; check "a jsonb qual runs under GpuScan" $?
plan=$($PSQL -d scantest -c "$GPU explain verbose select i from jb1 where d ? 'a' and (d->>'num')::int > 5 and d @> '{\"c\":true}'" | grep -c "jsonb_exists\|?")
[ "$plan" -ge 1 ]; check "the GPU qual shows the jsonb operators" $?

# errors are PostgreSQL's own (the row goes back to the backend)
for t in jb1; do
	g=$($PSQL -d scantest -c "$GPU select count(*) from $t where (d->>'dec')::int > 5" 2>&1 | head -1)
	c=$($PSQL -d scantest -c "$CPU select count(*) from $t where (d->>'dec')::int > 5" 2>&1 | head -1)
	echo "$c" | grep -q "invalid input syntax for type integer" ; check "$t: (->>)::int of a decimal is an error on the CPU" $?
	echo "$g" | grep -q "invalid input syntax for type integer" ; check "$t: ... and the same error on the GPU ($g)" $?
	g=$($PSQL -d scantest -c "$GPU select count(*) from $t where (d->>'n3')::int8 > 5" 2>&1 | head -1)
	c=$($PSQL -d scantest -c "$CPU select count(*) from $t where (d->>'n3')::int8 > 5" 2>&1 | head -1)
	echo "$c" | grep -q "out of range for type bigint" ; check "$t: (->>)::int8 beyond the range is an error on the CPU" $?
	echo "$g" | grep -q "out of range for type bigint" ; check "$t: ... and the same error on the GPU ($g)" $?
done

# ---- random documents: up to 44 keys out of about 270 (keys of 3 to 5 bytes, so the length-then-bytes order and the 32-entry
# offset stride matter), values of every kind, arrays of scalars, nested objects; "nK" always an integer, "sK" a string.
$PSQL -d scantest >/dev/null <<'SQL'
create function jb_h(i int, salt text, m int) returns int language sql immutable as
$$ select (('x' || substr(md5(i::text || salt), 1, 7))::bit(28)::int % m) $$;
create function jb_rnd(i int) returns jsonb language sql immutable as $$
select coalesce((select jsonb_object_agg(k, v) from (
   select ('w' || lpad(jb_h(i, 'k' || j, 90)::text, 2, '0') || repeat('x', jb_h(i, 'l' || j, 3))) as k,
     case jb_h(i, 'v' || j, 8)
       when 0 then to_jsonb(jb_h(i, 'a' || j, 2000) - 1000)
       when 1 then to_jsonb((jb_h(i, 'a' || j, 20000) - 10000) / 100.0)
       when 2 then to_jsonb('s' || jb_h(i, 'a' || j, 60))
       when 3 then to_jsonb(jb_h(i, 'a' || j, 30)::text)
       when 4 then to_jsonb(jb_h(i, 'a' || j, 2) = 0)
       when 5 then 'null'::jsonb
       when 6 then (select coalesce(jsonb_agg(case when jb_h(i, 'e' || j || m, 2) = 0 then to_jsonb(jb_h(i, 'f' || j || m, 100))
                                                   else to_jsonb('s' || jb_h(i, 'f' || j || m, 20)) end), '[]')
                      from generate_series(1, jb_h(i, 'n' || j, 5)) m)
       else jsonb_build_object('a' || jb_h(i, 'p' || j, 4), jb_h(i, 'q' || j, 50), 'b', 's' || jb_h(i, 'r' || j, 10))
     end as v
   from generate_series(1, jb_h(i, 'nk', 45)) j) t), '{}')
 || jsonb_build_object('n' || (i % 10), (i % 1000) - 500, 's' || (i % 5), 's' || (i % 40))
$$;
SQL
sql "create table jb_rnd as select i::int4 as i, (i % 7)::int4 as g, jb_rnd(i) as d from generate_series(1, $JBROWS) i"
sql "vacuum analyze jb_rnd"
t=jb_rnd
echo "table $t (random documents)"
# nothing is stored out of line here: plain values are never handed back to the backend
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from jb_rnd where d->>'s2' = 's7' or d ? 'w05' or (d->>'n3')::int > 5" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -eq 0 ]; check "plain jsonb is never handed back to the backend ($msgs)" $?
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from jb_rnd where d->>'w05' is null and d @> '{\"n1\":1}' and jsonb_typeof(d->'w17x') = 'array'" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -eq 0 ]; check "IS NULL, @> and jsonb_typeof on plain values need no recheck ($msgs)" $?
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from jb_rnd where d->>'w05' = 'x'" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -ge 1 ]; check "numbers and containers as text are formatted by the backend: rechecked ($msgs)" $?
$PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; explain select (d->>'num')::int as k, count(*) from jb1 where d ? 'num' group by 1" 2>&1 | head -12
for k in w05 w17x w33xx w60 w89 w00x w42; do
	run $t "$k: ? and typeof" "select i, d from %T where d ? '$k' or jsonb_typeof(d->'$k') = 'number'"
	run $t "$k: ->> = string" "select i from %T where d->>'$k' = 's7' or d->>'$k' = '12' or d->>'$k' = 'true'"
	run $t "$k: ->> IS NULL, -> IS NULL" "select i from %T where d->>'$k' is null and d->'$k' is not null"
	run $t "$k: -> index 0, -> key b" "select i from %T where d->'$k'->>0 = '5' or d->'$k'->>'b' = 's3' or d->'$k'->1 is not null"
	run $t "$k: typeof of member" "select i from %T where jsonb_typeof(d->'$k') = 'array' or jsonb_typeof(d->'$k'->'a1') = 'number'"
done
run $t "casts of the always-present integer keys" "select i from %T where (d->>'n3')::int > 100 or (d->>'n7')::int8 < -400 or (d->>'n0')::int2 = 0"
run $t "keys present in every row" "select i from %T where d->>'s2' = 's7' or d->>'s4' = 's39' or d ? 'n9'"
for spec in "100 1" "200 2" "3000 3" "45000 5" "77777 8" "123456 12"; do
	set -- $spec
	pat=$($PSQL -d scantest -c "select (select jsonb_object_agg(key, value) from (select * from jsonb_each(jb_rnd($1)) order by key limit $2) t)::text")
	run $t "@> a subset of the document of row $1 ($2 keys)" "select i from %T where d @> '$pat'"
	run $t "@> a subset of the document of row $1 ($2 keys), with a false key" "select i from %T where d @> '$pat' or d @> '{\"zz\":1}'"
	full=$($PSQL -d scantest -c "select jb_rnd($1)::text")
	run $t "@> the whole document of row $1" "select i from %T where d @> '$full'"
done

# ---- joins: a jsonb column passes through a GpuJoin as it does any other column; jsonb expressions are not in join programs
echo "jsonb in joins"
sql "create table jb_dim as select g::int4 as k, 'dim' || g as nm, jsonb_build_object('k', g, 'name', 'n' || g) as j from generate_series(0, 8) g"
sql "analyze jb_dim"
for t in jb1 jb2; do
	ANYPLAN=0
	runjoin $t "a jsonb column of the outer side passes through" "select t.i, t.d, x.nm from %T t join jb_dim x on t.g = x.k where t.i < 3000"
	runjoin $t "a jsonb column of the inner side passes through" "select t.i, x.j, x.nm from %T t join jb_dim x on t.g = x.k where t.i < 3000"
	runjoin $t "jsonb of both sides and an integer qual" "select t.i, t.d, x.j from %T t join jb_dim x on t.g = x.k where t.i < 3000 and x.k > 2"
	runjoin $t "IS NULL on a jsonb column in a join" "select t.i, x.nm from %T t join jb_dim x on t.g = x.k where t.d is null and t.i < 30000"
	runjoin $t "a jsonb qual keeps the outer scan on the CPU" "select t.i, x.nm from %T t join jb_dim x on t.g = x.k where t.d ? 'a' and t.i < 50000"
	ANYPLAN=1
	runjoin $t "a jsonb join qual stays on the CPU" "select t.i, x.nm from %T t join jb_dim x on t.g = x.k and t.d->>'b' = x.j->>'name' where t.i < 20000"
	runjoin $t "an aggregate of jsonb values over a join" "select x.k, count(*), sum((t.d->>'num')::int) from %T t join jb_dim x on t.g = x.k where t.d ? 'num' group by 1 order by 1"
	ANYPLAN=0
done
