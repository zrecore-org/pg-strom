#!/bin/sh
# Differential fuzzer of the Vulkan GpuScan / GpuJoin / GpuPreAgg (see fuzz.py for the generator and the
# comparison rule): a throwaway cluster with the settings of scan_test.sh, edge-case tables in three physical
# variants (fz1 all-visible, fz2 with updates and deletes, fz3 a vacuumed mix) and a small dimension table fz_d,
# then FUZZ_COUNT random queries (or FUZZ_TIME seconds) with the seed FUZZ_SEED, each compared with the CPU result.
# The corpus of minimal reproducers (test/gpuserv/fuzz_corpus.txt) is replayed first.
#
#   sh test/gpuserv/locked_test.sh "$PWD" fuzz_test.sh   (env: FUZZ_SEED FUZZ_COUNT FUZZ_TIME FUZZ_START FUZZ_ARGS
#                                                         NO_REPLAY KEEP REUSE ALLOW_CPU NROWS)
#
# Exit status 1 if a query failed (it is printed with its minimized form and appended to FUZZ_CORPUS).
# KEEP=1 keeps the cluster, REUSE=1 runs against a kept cluster (no new data); FUZZ_ARGS="--only N" reproduces
# query N of the seed, "--verbose" prints every query.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/fz_test}
PORT=${PORT:-54402}
NROWS=${NROWS:-30000}
D=$(cd "$(dirname "$0")" && pwd)
FUZZ_SEED=${FUZZ_SEED:-1}
FUZZ_COUNT=${FUZZ_COUNT:-300}
FUZZ_TIME=${FUZZ_TIME:-0}
FUZZ_START=${FUZZ_START:-0}
FUZZ_CORPUS=${FUZZ_CORPUS:-$D/fuzz_corpus.txt}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
sql() { $PSQL -d fuzz -c "$1"; }
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -t 20 -w stop >/dev/null 2>&1 || $PGBIN/pg_ctl -D "$ROOT/data" -m immediate -w stop >/dev/null 2>&1; }

if [ -z "$REUSE" ]; then
stop
rm -rf "$ROOT"; mkdir -p "$ROOT"
$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
pg_strom.gpu_allow_cpu_device = $ALLOW_CPU
max_worker_processes = 24
CONF
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
[ $n -lt 40 ] || { echo "service did not become ready"; stop; exit 1; }
$PSQL -d postgres -c "create database fuzz" >/dev/null
sql "create extension pg_strom" >/dev/null

# ---- data. Deterministic: every value is picked by a hash of the row number. "n*" columns are narrow (a few
# operations cannot overflow them), "e*" columns hold boundary values, about 7% of every column is NULL.
$PSQL -d fuzz >/dev/null <<SQL
create function fz_pick(i int, salt int, n int) returns int language sql immutable as
	\$\$ select (hashint4(i * 31 + salt) & 2147483647) % n \$\$;
create function fz_null(i int, salt int) returns bool language sql immutable as
	\$\$ select fz_pick(i, salt + 500, 100) < 7 \$\$;
create table fz_base as select i,
	case when fz_null(i, 1) then null else (fz_pick(i, 1, 201) - 100)::int2 end as n2,
	case when fz_null(i, 2) then null else fz_pick(i, 2, 2001) - 1000 end as n4,
	case when fz_null(i, 3) then null else (fz_pick(i, 3, 2001) - 1000)::int8 end as n8,
	case when fz_null(i, 4) then null else ((fz_pick(i, 4, 2001) - 1000) / 8.0)::float4 end as nf,
	case when fz_null(i, 5) then null else (array[-32768,-32767,-1,0,1,32766,32767,255,256,181,182,-181,-182,100])[1 + fz_pick(i, 5, 14)]::int2 end as e2,
	case when fz_null(i, 6) then null else (array[-2147483648,-2147483647,-65536,-1,0,1,65535,65536,2147483646,2147483647,46340,46341,-46340,-46341,32768,-32769,16777216,16777217,1000000007])[1 + fz_pick(i, 6, 19)]::int4 end as e4,
	case when fz_null(i, 7) then null else (array['-9223372036854775808','-9223372036854775807','-4294967296','-2147483649','-2147483648','-1','0','1','2147483647','2147483648','4294967296','3037000499','3037000500','-3037000500','4611686018427387904','9223372036854775806','9223372036854775807','9007199254740993','1000000007000000007'])[1 + fz_pick(i, 7, 19)]::int8 end as e8,
	case when fz_null(i, 8) then null else (array['NaN','Infinity','-Infinity','0','-0','1','-1','3.4e38','-3.4e38','1e-38','1.4e-45','16777216','16777217','0.1','2147483648','9.3e18','-9.3e18','0.5','2.5','-2.5','1.5'])[1 + fz_pick(i, 8, 21)]::float4 end as ef,
	case when fz_null(i, 9) then null else fz_pick(i, 9, 3) = 0 end as bo,
	case when fz_null(i, 10) then null when fz_pick(i, 11, 3) = 0 then (array['infinity','-infinity','4713-01-01 BC','5874897-12-31','1970-01-01','2000-02-29','1999-12-31','2038-01-19','294276-12-31','294277-01-01','1900-01-01'])[1 + fz_pick(i, 12, 11)]::date
	     else date '2000-01-01' + (fz_pick(i, 13, 20000) - 10000) end as d,
	case when fz_null(i, 14) then null else (array['00:00:00','12:34:56.789','23:59:59.999999','24:00:00','00:00:00.000001','12:00:00'])[1 + fz_pick(i, 15, 6)]::time end as tm,
	case when fz_null(i, 16) then null when fz_pick(i, 17, 3) = 0 then (array['infinity','-infinity','4713-01-01 00:00:00 BC','294276-12-31 23:59:59.999999','1970-01-01 00:00:00','2000-02-29 12:34:56.789','2000-01-01 00:00:00.000001','1999-12-31 23:59:59.999999'])[1 + fz_pick(i, 18, 8)]::timestamp
	     else timestamp '2000-01-01' + (fz_pick(i, 19, 1000000000) - 500000000) * interval '1 millisecond' end as ts,
	case when fz_null(i, 20) then null when fz_pick(i, 21, 3) = 0 then (array['infinity','-infinity','1970-01-01 00:00:00+00','2000-02-29 12:34:56.789+00','2000-01-01 00:00:00+05','2038-01-19 03:14:07+00'])[1 + fz_pick(i, 22, 6)]::timestamptz
	     else timestamptz '2000-01-01 00:00:00+00' + (fz_pick(i, 23, 1000000000) - 500000000) * interval '1 millisecond' end as tz,
	case when fz_null(i, 24) then null
	     when i % 997 = 0 then (select string_agg(md5((i + g)::text), '') from generate_series(1, 120) g)
	     when i % 1009 = 0 then repeat('ab', 2000)
	     else (array['','a','abc','ABC','xabcx','h' || chr(233) || 'llo','abcabc','a%b',' ','abc ','z',chr(26085) || chr(26412) || chr(35486),'abcdefghij','b','a_b','abcabcabcabcabcabc' || repeat('x', 150)])[1 + fz_pick(i, 25, 16)] end as s,
	case when fz_null(i, 26) then null else (array['','a','abc','ABC','xabcx','h' || chr(233),'abcabc','a%b',' ','abc ','z',chr(26085),'abcdefghij'])[1 + fz_pick(i, 27, 13)] end::varchar(10) as v,
	case when fz_null(i, 28) then null else (array['','a','abc','abc   ','xyz','ab','h' || chr(233)])[1 + fz_pick(i, 29, 7)] end::char(6) as c,
	case when fz_null(i, 30) then null when fz_pick(i, 31, 4) = 0 then (array[0,99999999.99,-99999999.99,0.01,-0.01,1,-1,12345678.90])[1 + fz_pick(i, 32, 8)] else (fz_pick(i, 33, 200000) - 100000) / 100.0 end::numeric(10,2) as nm,
	case when fz_null(i, 34) then null when fz_pick(i, 35, 4) = 0 then (array[0,999999999999.999999,-999999999999.999999,0.000001,-0.000001,1,-1])[1 + fz_pick(i, 36, 7)] else (fz_pick(i, 37, 2000000) - 1000000) / 1000.0 end::numeric(18,6) as nl,
	case when fz_null(i, 38) then null else (array['NaN','Infinity','-Infinity','0','0.00','1','-1','0.5','1.25','100.50','12345678.9012','922337203685477.5807','-0.0','1e20','0.0000001','826563.6313334870','467842.677','9223372036854775807','-9223372036854775808','4611686018427387904','18446744073709551616','-18446744073709551616','123456789012345678.123456','1e-10','3','-7.5','99999999999999999999999999.5'])[1 + fz_pick(i, 39, 27)]::numeric end as nw
	from generate_series(1, $NROWS) i;
create table fz1 as select * from fz_base; vacuum analyze fz1;
create table fz2 as select * from fz_base; update fz2 set n4 = n4 + 1 where i % 5 = 0; delete from fz2 where i % 13 = 0; analyze fz2;
create table fz3 as select * from fz_base; vacuum fz3; update fz3 set n8 = n8 + 7, s = s || 'x' where i % 17 = 0; analyze fz3;
create table fz_d as select j as id,
	case when j % 40 = 7 then null else (j - 151 + (j % 25 = 0)::int * 3)::int4 end as k4,
	case when j % 40 = 9 then null when j % 50 = 3 then (array['9223372036854775807','-9223372036854775808','2147483648','3037000500'])[1 + j % 4]::int8 else ((j - 151) * 2)::int8 end as k8,
	case when j % 30 = 5 then null when j % 31 = 0 then 'infinity'::date else date '2000-01-01' + (j - 200) * 25 end as kd,
	case when j % 30 = 6 then null when j % 37 = 0 then 'infinity'::timestamp else timestamp '2000-01-01' + (j - 200) * interval '25 days 3 hours' end as kts,
	case when j % 30 = 4 then null else (array['','a','abc','ABC','xabcx','abcabc'])[1 + j % 6] end as ks,
	case when j % 11 = 0 then null else (j % 201 - 100) end as p4,
	case when j % 13 = 0 then null else (array['-9223372036854775808','9223372036854775807','0','2147483648','-1','1'])[1 + j % 6]::int8 end as pe,
	case when j % 17 = 0 then null else (j * 37 % 20000 - 10000) / 100.0 end::numeric(10,2) as pn,
	case when j % 19 = 0 then null else j % 3 = 0 end as pb,
	case when j % 23 = 0 then null else repeat(chr(97 + j % 26), j % 12) end as ps,
	case when j % 29 = 0 then null else date '2000-01-01' + j * 7 end as pd
	from generate_series(1, 400) j;
analyze fz_d;
create table fz_w as select i, case when fz_null(i, 101) then null else fz_pick(i, 101, 21) - 10 end as w1,
	case when fz_null(i, 102) then null else fz_pick(i, 102, 21) - 10 end as w2,
	case when fz_null(i, 103) then null else fz_pick(i, 103, 21) - 10 end as w3,
	case when fz_null(i, 104) then null else fz_pick(i, 104, 21) - 10 end as w4,
	case when fz_null(i, 105) then null else fz_pick(i, 105, 21) - 10 end as w5,
	case when fz_null(i, 106) then null else fz_pick(i, 106, 21) - 10 end as w6,
	case when fz_null(i, 107) then null else fz_pick(i, 107, 21) - 10 end as w7,
	case when fz_null(i, 108) then null else fz_pick(i, 108, 21) - 10 end as w8,
	case when fz_null(i, 109) then null else fz_pick(i, 109, 21) - 10 end as w9,
	case when fz_null(i, 110) then null else fz_pick(i, 110, 21) - 10 end as w10,
	case when fz_null(i, 111) then null else fz_pick(i, 111, 21) - 10 end as w11,
	case when fz_null(i, 112) then null else fz_pick(i, 112, 21) - 10 end as w12,
	case when fz_null(i, 113) then null else fz_pick(i, 113, 21) - 10 end as w13,
	case when fz_null(i, 114) then null else fz_pick(i, 114, 21) - 10 end as w14,
	case when fz_null(i, 115) then null else fz_pick(i, 115, 21) - 10 end as w15,
	case when fz_null(i, 116) then null else fz_pick(i, 116, 21) - 10 end as w16,
	case when fz_null(i, 117) then null else fz_pick(i, 117, 21) - 10 end as w17,
	case when fz_null(i, 118) then null else fz_pick(i, 118, 21) - 10 end as w18,
	case when fz_null(i, 119) then null else fz_pick(i, 119, 21) - 10 end as w19,
	case when fz_null(i, 120) then null else fz_pick(i, 120, 21) - 10 end as w20,
	case when fz_null(i, 121) then null else fz_pick(i, 121, 21) - 10 end as w21,
	case when fz_null(i, 122) then null else fz_pick(i, 122, 21) - 10 end as w22,
	case when fz_null(i, 123) then null else fz_pick(i, 123, 21) - 10 end as w23,
	case when fz_null(i, 124) then null else fz_pick(i, 124, 21) - 10 end as w24,
	case when fz_null(i, 125) then null else fz_pick(i, 125, 21) - 10 end as w25,
	case when fz_null(i, 126) then null else fz_pick(i, 126, 21) - 10 end as w26,
	case when fz_null(i, 127) then null else fz_pick(i, 127, 21) - 10 end as w27,
	case when fz_null(i, 128) then null else fz_pick(i, 128, 21) - 10 end as w28,
	case when fz_null(i, 129) then null else fz_pick(i, 129, 21) - 10 end as w29,
	case when fz_null(i, 130) then null else fz_pick(i, 130, 21) - 10 end as w30,
	case when fz_null(i, 131) then null else fz_pick(i, 131, 21) - 10 end as w31,
	case when fz_null(i, 132) then null else fz_pick(i, 132, 21) - 10 end as w32,
	case when fz_null(i, 133) then null else fz_pick(i, 133, 21) - 10 end as w33,
	case when fz_null(i, 134) then null else fz_pick(i, 134, 21) - 10 end as w34,
	case when fz_null(i, 135) then null else fz_pick(i, 135, 21) - 10 end as w35,
	case when fz_null(i, 136) then null else fz_pick(i, 136, 21) - 10 end as w36,
	case when fz_null(i, 137) then null else fz_pick(i, 137, 21) - 10 end as w37,
	case when fz_null(i, 138) then null else fz_pick(i, 138, 21) - 10 end as w38,
	case when fz_null(i, 139) then null else fz_pick(i, 139, 21) - 10 end as w39,
	case when fz_null(i, 140) then null else fz_pick(i, 140, 21) - 10 end as w40,
	case when fz_null(i, 141) then null else fz_pick(i, 141, 21) - 10 end as w41,
	case when fz_null(i, 142) then null else fz_pick(i, 142, 21) - 10 end as w42,
	case when fz_null(i, 143) then null else fz_pick(i, 143, 21) - 10 end as w43,
	case when fz_null(i, 144) then null else fz_pick(i, 144, 21) - 10 end as w44,
	case when fz_null(i, 145) then null else (fz_pick(i, 145, 21) - 10)::int8 end as w45,
	case when fz_null(i, 146) then null else (fz_pick(i, 146, 21) - 10)::int8 end as w46,
	case when fz_null(i, 147) then null else (fz_pick(i, 147, 21) - 10)::int8 end as w47,
	case when fz_null(i, 148) then null else (fz_pick(i, 148, 21) - 10)::int8 end as w48,
	case when fz_null(i, 149) then null else (fz_pick(i, 149, 21) - 10)::int8 end as w49,
	case when fz_null(i, 150) then null else (fz_pick(i, 150, 21) - 10)::int8 end as w50,
	case when fz_null(i, 151) then null else repeat(chr(97 + fz_pick(i, 151, 26)), fz_pick(i, 251, 10)) end as w51,
	case when fz_null(i, 152) then null else repeat(chr(97 + fz_pick(i, 152, 26)), fz_pick(i, 252, 10)) end as w52,
	case when fz_null(i, 153) then null else repeat(chr(97 + fz_pick(i, 153, 26)), fz_pick(i, 253, 10)) end as w53,
	case when fz_null(i, 154) then null else repeat(chr(97 + fz_pick(i, 154, 26)), fz_pick(i, 254, 10)) end as w54,
	case when fz_null(i, 155) then null else ((fz_pick(i, 155, 41) - 20) / 4.0)::float4 end as w55,
	case when fz_null(i, 156) then null else ((fz_pick(i, 156, 41) - 20) / 4.0)::float4 end as w56,
	case when fz_null(i, 157) then null else ((fz_pick(i, 157, 41) - 20) / 4.0)::float4 end as w57,
	case when fz_null(i, 158) then null else ((fz_pick(i, 158, 41) - 20) / 4.0)::float4 end as w58,
	case when fz_null(i, 159) then null else date '2000-01-01' + fz_pick(i, 159, 1000) end as w59,
	case when fz_null(i, 160) then null else ((fz_pick(i, 160, 2001) - 1000) / 100.0)::numeric(10,2) end as w60
	from generate_series(1, $NROWS) i;
vacuum analyze fz_w;
SQL
fi

STATUS=0
if [ -n "$DBG_SQL" ]; then		# ad-hoc: run a SQL file (psql output and stderr) against the fuzz tables, then stop
	# chunks separated by lines "-- NEXT" run in separate sessions (a crash only ends its own chunk)
	rm -f "$ROOT"/chunk.*
	awk -v d="$ROOT" 'BEGIN { n = 0 } /^-- NEXT/ { n++; next } { print > (d "/chunk." n ".sql") }' "$DBG_SQL"
	for f in "$ROOT"/chunk.*.sql; do
		$PGBIN/psql -h "$ROOT" -p "$PORT" -U postgres -X -d fuzz -f "$f" 2>&1
		sleep 2
	done
	[ -n "$DBG_LOG" ] && grep -E "$DBG_LOG" "$ROOT/log" | tail -20 | cut -c1-600
	[ -z "$KEEP" ] && stop
	exit 0
fi
if [ -s "$FUZZ_CORPUS" ] && [ -z "$NO_REPLAY" ]; then
	echo "replaying $FUZZ_CORPUS"
	python3 -I "$D/fuzz.py" --sock "$ROOT" --port "$PORT" --db fuzz --replay "$FUZZ_CORPUS" || STATUS=1
fi
python3 -I "$D/fuzz.py" --sock "$ROOT" --port "$PORT" --db fuzz --seed "$FUZZ_SEED" --count "$FUZZ_COUNT" \
	--time "$FUZZ_TIME" --start "$FUZZ_START" --corpus "$FUZZ_CORPUS" $FUZZ_ARGS || STATUS=1
[ -z "$KEEP" ] && stop
if [ $STATUS -eq 0 ]; then echo "FUZZ TEST: PASSED"; else echo "FUZZ TEST: FAILURES"; fi
exit $STATUS
