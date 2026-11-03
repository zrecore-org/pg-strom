#!/usr/bin/env python3
"""
Differential fuzzer for the Vulkan GpuScan / GpuJoin / GpuPreAgg: random queries run on the GPU path and with
pg_strom.enabled = off, and the results are compared.  Standard library only; run with `python3 -I`.

  python3 -I test/gpuserv/fuzz.py --sock DIR --port N [--db fuzz] [--seed S] [--count N] [--time SEC]
                                  [--corpus FILE] [--replay FILE] [--no-minimize] [--verbose]

Normally started by fuzz_test.sh (which builds the throwaway cluster and the tables, see there for the data).
Environment of the wrapper: FUZZ_SEED, FUZZ_COUNT, FUZZ_TIME (seconds; 0 = no limit), FUZZ_CORPUS.

How a query is generated
  A typed random expression generator (depth 1..4) over the columns of the fz tables (narrow columns whose arithmetic
  rarely overflows, "edge" columns full of boundary values, NULLs, NaN/Infinity, infinite dates, UTF-8 and empty
  text, numerics beyond 64 bits), literals that favour boundaries, comparisons, AND/OR/NOT, IS [NOT] NULL, casts,
  LIKE patterns of all shapes.  Operators and functions the VM does not implement are generated too (about one
  in six), so that the planner's gating is fuzzed as well.  Shapes: scan (qual + projection), aggregate without
  and with GROUP BY (FILTER, expression arguments, one or two keys), joins (inner, left, right, full, semi, anti,
  with extra join quals and computed keys against a small dimension table), aggregate over a join.  Half of the
  queries ask for parallel workers.  Each query is generated from `random.Random(seed ^ index)`, so
  "--seed S --only N" reproduces query N of a run.

Comparison rule (the result of the GPU run G and of the CPU run C, CPU = pg_strom.enabled off)
  * both succeed: the sorted result sets must be equal (text of every row, then count+md5).  A mismatch is re-run
    once (a flaky difference is reported as "unstable").  FAIL.
  * both fail: OK.  Equal messages count as "same error"; different messages (PostgreSQL evaluates quals in cost
    order and rows in physical order, the GPU in its own order, so a query with two kinds of errors may raise
    either) count as "different error" and are only tallied.
  * exactly one fails: PostgreSQL may reorder quals and evaluates AND/OR lazily, so an error can legitimately be
    avoided or reached by one plan only.  The query is "strict" when it has a single top-level condition tree
    with every operand evaluated on every row (one expression, no AND/OR/IN, no outer join, no CPU recheck):
    then error asymmetry is a FAIL.  Otherwise it is tallied as "asymmetric error" and shown in the summary.
  * an error that only the GPU raises and that names the service (translation failures such as "needs more than
    32 registers", "gpuserv:") is always a FAIL ("service refused the program"): the planner must not offer what
    the service cannot translate.
  * errors that PostgreSQL itself raises for both (operator does not exist, ...) are generator mistakes and are
    counted as "invalid" (their share is printed; it should stay small).
  Float results are compared as text, i.e. bit-exact for float4.  Unordered results are sorted by row text.

A failing query is minimized (drop projection columns / conditions / aggregates / keys, replace subexpressions by
a child, a column or a literal while the same kind of difference remains) and appended to the corpus file.
"""
import argparse
import hashlib
import os
import random
import re
import subprocess
import sys
import time

# ------------------------------------------------------------------ expression model


STRESS_RATE = float(os.environ.get("FUZZ_STRESS", "0.15"))


class N:
    """expression node: sql text is fmt.format(*args); typ is the SQL value type family"""
    __slots__ = ("fmt", "args", "typ", "leaf", "pr")

    def __init__(self, fmt, args, typ, leaf=False, pr=False):
        self.fmt = fmt
        self.args = list(args)
        self.typ = typ
        self.leaf = leaf        # a column or literal
        self.pr = pr            # may raise an error / is evaluation-order sensitive (informational)

    def sql(self):
        return self.fmt.format(*[a.sql() for a in self.args])

    def size(self):
        return 1 + sum(a.size() for a in self.args)


def col(name, typ):
    return N(name, [], typ, leaf=True)


def lit(text, typ):
    return N(text, [], typ, leaf=True)


INTS = ("i2", "i4", "i8")
INTSQL = {"i2": "int2", "i4": "int4", "i8": "int8"}
NUMERICS = ("i2", "i4", "i8", "f4", "num")

# columns of the fz tables: name -> (type, narrow?)
COLS = {
    "n2": ("i2", 1), "n4": ("i4", 1), "n8": ("i8", 1), "nf": ("f4", 1),
    "e2": ("i2", 0), "e4": ("i4", 0), "e8": ("i8", 0), "ef": ("f4", 0),
    "i": ("i4", 1), "bo": ("bool", 0),
    "d": ("date", 0), "tm": ("time", 0), "ts": ("ts", 0), "tz": ("tz", 0),
    "s": ("text", 0), "v": ("text", 0), "c": ("bpchar", 0),
    "nm": ("num", 1), "nl": ("num", 1), "nw": ("num", 0),
}
DCOLS = {   # the dimension table fz_d
    "k4": ("i4", 1), "k8": ("i8", 1), "kd": ("date", 0), "kts": ("ts", 0), "ks": ("text", 0),
    "p4": ("i4", 1), "pe": ("i8", 0), "pn": ("num", 1), "pb": ("bool", 0), "ps": ("text", 0), "pd": ("date", 0),
}

LITS = {
    "i2": ["0", "1", "-1", "2", "5", "100", "-100", "32767", "-32768", "32766", "255", "256", "181", "182", "-32767"],
    "i4": ["0", "1", "-1", "2", "3", "7", "10", "100", "1000", "-1000", "46340", "46341", "65535", "65536",
           "2147483647", "-2147483648", "2147483646", "-2147483647", "32768", "-32769", "16777216", "16777217"],
    "i8": ["0", "1", "-1", "2", "10", "1000", "-1000", "2147483647", "2147483648", "-2147483648", "-2147483649",
           "4294967296", "3037000499", "3037000500", "4611686018427387904", "9223372036854775807",
           "-9223372036854775808", "9223372036854775806", "-9223372036854775807", "1000000007", "9007199254740993"],
    "f4": ["0", "1", "-1", "0.5", "1.5", "100.5", "16777216", "3.4e38", "-3.4e38", "'NaN'", "'Infinity'", "'-Infinity'",
           "1e-38", "1.4e-45", "0.1", "2147483648", "9.3e18", "-0.0"],
    "num": ["0", "1", "-1", "0.5", "1.25", "100.50", "12345678.9012", "99999999.99", "922337203685477.5807", "'NaN'",
            "-0.0", "1e20", "0.0000001", "826563.6313334870", "467842.677", "9223372036854775807",
            "-9223372036854775808", "4611686018427387904", "18446744073709551616", "'Infinity'", "0.00", "10.00",
            "3", "-7.5", "1e-10", "123456789012345678.123456"],
    "date": ["1970-01-01", "2000-02-29", "1999-12-31", "infinity", "-infinity", "4713-01-01 BC", "5874897-12-31",
             "2038-01-19", "1900-01-01", "2000-01-01", "294276-12-31", "294277-01-01"],
    "ts": ["1970-01-01 00:00:00", "2000-02-29 12:34:56.789", "infinity", "-infinity", "4713-01-01 00:00:00 BC",
           "294276-12-31 23:59:59.999999", "2000-01-01 00:00:00.000001", "1999-12-31 23:59:59.999999"],
    "tz": ["1970-01-01 00:00:00+00", "2000-02-29 12:34:56.789+00", "infinity", "-infinity", "2000-01-01 00:00:00+05",
           "2038-01-19 03:14:07+00"],
    "time": ["00:00:00", "12:34:56.789", "23:59:59.999999", "24:00:00", "00:00:00.000001", "12:00:00"],
    "text": ["", "a", "abc", "ABC", "xabcx", "héllo", "abcabc", "a%b", " ", "abc ", "z", "日本語",
             "abcdefghij", "b"],
    "bpchar": ["", "a", "abc", "abc   ", "xyz"],
}

LIKE_PATS = ["abc", "abc%", "%abc", "%abc%", "a%", "%a", "%", "", "a_c", "%a%b%", "a%b", "ab\\%%", "%é%",
             "h%", "%x", "_bc", "abc_", "%%", "a\\%b", "x%x"]
REGEX_PATS = ["^abc", "c$", "a.c", "[a-c]+"]


def tok(r, typ):
    """a literal of the type"""
    if typ == "bool":
        return lit(r.choice(["true", "false"]), "bool")
    if typ in ("text", "bpchar"):
        v = r.choice(LITS[typ]).replace("'", "''")
        return lit("'" + v + "'::" + ("text" if typ == "text" else "char(6)"), typ)
    if typ in ("date", "ts", "tz", "time"):
        nm = {"date": "date", "ts": "timestamp", "tz": "timestamptz", "time": "time"}[typ]
        return lit("'" + r.choice(LITS[typ]) + "'::" + nm, typ)
    small = r.random() < 0.7        # mostly small values: the boundary ones make most queries raise errors
    if typ in INTS:
        return lit("(" + r.choice(LITS[typ][:6] if small else LITS[typ]) + ")::" + INTSQL[typ], typ)
    if typ == "f4":
        v = r.choice(LITS["f4"][:6] if small else LITS["f4"])
        return lit("(" + v + ")::float4", typ)
    if typ == "num":
        v = r.choice(LITS["num"][:8] if small else LITS["num"])
        return lit("(" + v + ")::numeric", typ)
    if typ == "f8":
        return lit("(" + r.choice(LITS["f4"]) + ")::float8", typ)
    raise ValueError(typ)


class Gen:
    def __init__(self, r, tables):
        self.r = r
        self.tabs = tables          # list of (alias prefix, {name: (type, narrow)})
        self.allow = r.random() < 0.22   # this query may use operators the VM does not implement (gating)

    def columns(self, typ):
        out = []
        for pfx, cols in self.tabs:
            for n, (t, narrow) in cols.items():
                if t == typ:
                    out.append((pfx + n, narrow))
        return out

    def leaf(self, typ):
        r = self.r
        cs = self.columns(typ)
        if cs and r.random() < 0.8:
            narrow = [c for c, nw in cs if nw]
            edge = [c for c, nw in cs if not nw]
            if narrow and (not edge or r.random() < 0.65):
                return col(r.choice(narrow), typ)
            if edge:
                return col(r.choice(edge), typ)
            return col(r.choice(narrow), typ)
        return tok(r, typ)

    def expr(self, typ, depth):
        r = self.r
        if depth <= 0 or r.random() < 0.15:
            return self.leaf(typ)
        g = getattr(self, "g_" + typ)
        return g(depth)

    # ---- helpers
    def u(self, p=0.5):
        """true if an unsupported production may be used here"""
        return self.allow and self.r.random() < p

    def anyval(self, depth):
        return self.expr(self.r.choice(["i2", "i4", "i8", "f4", "num", "date", "ts", "tz", "time", "text", "bool", "bpchar"]), depth)

    def cast(self, e, to):
        return N("(" + "{0}" + ")::" + to, [e], {"int2": "i2", "int4": "i4", "int8": "i8", "float4": "f4",
                                                    "float8": "f8", "numeric": "num", "bool": "bool", "text": "text",
                                                    "date": "date", "timestamp": "ts", "timestamptz": "tz",
                                                    "time": "time"}.get(to, "text"), pr=True)

    # ---- bool
    def g_bool(self, d):
        r = self.r
        k = r.random()
        if k < 0.34:                      # comparison of a type
            t = r.choice(["i2", "i4", "i4", "i8", "i8", "f4", "num", "num", "date", "ts", "tz", "time", "text", "text", "bpchar", "bool"])
            if t == "f4" and not self.allow:
                t = "i4"
            op = r.choice(["<", "<=", ">", ">=", "=", "<>"])
            if t == "bool":
                op = r.choice(["=", "<>"])
            a = self.expr(t, d - 1)
            b = self.expr(r.choice(self.mix(t) if self.allow else [t]), d - 1)
            return N("({0} " + op + " {1})", [a, b], "bool")
        if k < 0.50:                      # AND / OR / NOT
            op = r.choice(["AND", "OR", "NOT"])
            if op == "NOT":
                return N("(NOT {0})", [self.expr("bool", d - 1)], "bool")
            return N("({0} " + op + " {1})", [self.expr("bool", d - 1), self.expr("bool", d - 1)], "bool")
        if k < 0.62:                      # IS [NOT] NULL
            t = r.choice(["i2", "i4", "i8", "f4", "num", "date", "ts", "time", "text", "bool", "bpchar", "tz"])
            return N("({0} IS " + r.choice(["", "NOT "]) + "NULL)", [self.expr(t, d - 1)], "bool")
        if k < 0.77:                      # text LIKE / equality
            e = self.expr("text", d - 1)
            p = r.choice(LIKE_PATS).replace("'", "''")
            k2 = r.random()
            if k2 < 0.55:
                return N("({0} " + r.choice(["LIKE", "NOT LIKE"]) + " '" + p + "')", [e], "bool")
            if k2 < 0.65 and self.u(0.7):
                return N("({0} " + r.choice(["ILIKE", "~"]) + " '" + r.choice(LIKE_PATS + REGEX_PATS).replace("'", "''") + "')", [e], "bool")
            if k2 < 0.85:
                return N("({0} = {1})", [e, self.expr("text", d - 1)], "bool")
            c = self.leaf("bpchar")
            return N("({0} " + r.choice(["LIKE", "NOT LIKE"]) + " '" + p + "')", [c], "bool")
        if k < 0.83:                      # BETWEEN / IN (not on the GPU: ScalarArrayOp, but BETWEEN is two comparisons)
            t = r.choice(["i2", "i4", "i8", "date", "ts", "num"])
            if not self.u(0.3):
                return N("({0} BETWEEN {1} AND {2})", [self.expr(t, d - 1), self.leaf(t), self.leaf(t)], "bool")
            return N("({0} IN ({1}, {2}, {3}))", [self.expr(t, d - 1), self.leaf(t), self.leaf(t), self.leaf(t)], "bool")
        if k < 0.88 and self.u(0.8):
            return N("({0} IS " + r.choice(["TRUE", "FALSE", "NOT TRUE", "NOT FALSE", "UNKNOWN"]) + ")", [self.expr("bool", d - 1)], "bool")
        if k < 0.93:                      # casts to bool, bool columns
            return N("({0})::bool", [self.expr("i4", d - 1)], "bool", pr=True)
        if k < 0.97 and self.u(0.8):                      # coalesce / case: the VM has none
            return N("COALESCE({0}, {1})", [self.expr("bool", d - 1), self.expr("bool", d - 1)], "bool")
        return self.leaf("bool")

    def mix(self, t):
        """types that can be compared with t"""
        if t in INTS:
            return [t] * 4 + list(INTS) + ["num", "f4"]
        if t == "f4":
            return ["f4"] * 4 + ["i4", "i2", "num"]
        if t == "num":
            return ["num"] * 4 + ["i4", "i8", "i2"]
        if t == "date":
            return ["date"] * 4 + ["ts"]
        if t == "ts":
            return ["ts"] * 4 + ["date"]
        if t == "text":
            return ["text"] * 5 + ["bpchar"]
        if t == "bpchar":
            return ["bpchar"] * 5
        return [t]

    # ---- integers
    def g_int(self, w, d):
        r = self.r
        t = {2: "i2", 4: "i4", 8: "i8"}[w]
        k = r.random()
        if k < 0.60:
            op = r.choice(["+", "+", "+", "-", "-", "-", "*", "*", "*", "/", "%"])
            a = self.expr(t, d - 1)
            ot = t
            if op != "%" and r.random() < 0.25:
                ot = r.choice(INTS)
            b = self.expr(ot, d - 1)
            if op == "%" and ot != t:
                b = self.cast(b, INTSQL[t])
            if op in ("/", "%") and r.random() < 0.6:
                b = lit("(" + r.choice(["1", "-1", "2", "3", "7", "10", "-3", "1", "-1", "0"]) + ")::" + INTSQL[ot if op != "%" else t], ot)
            res = t if ot == t else max(t, ot, key=lambda x: int(x[1:]))
            return N("({0} " + op + " {1})", [a, b], res, pr=True)
        if k < 0.70:
            return N("(-{0})", [self.expr(t, d - 1)], t, pr=True)
        if k < 0.74:
            return N("(+{0})", [self.expr(t, d - 1)], t)
        if k < 0.88:                      # casts between int types
            src = r.choice(INTS)
            return self.cast(self.expr(src, d - 1), INTSQL[t])
        if k < 0.93 and w == 4:
            return N("length({0})", [self.expr("text", d - 1)], "i4")
        if k < 0.96 and w == 4:
            return N("({0} - {1})", [self.expr("date", d - 1), self.expr("date", d - 1)], "i4", pr=True)
        if k < 0.98 and self.u(0.6):
            return N("abs({0})", [self.expr(t, d - 1)], t, pr=True)
        if not self.u(0.5):
            return self.cast(self.expr(r.choice(INTS), d - 1), INTSQL[t])
        # float / numeric -> int casts
        src = r.choice(["f4", "num"])
        return self.cast(self.expr(src, d - 1), INTSQL[t])

    def g_i2(self, d): return self.g_int(2, d)
    def g_i4(self, d): return self.g_int(4, d)
    def g_i8(self, d): return self.g_int(8, d)

    # ---- float4 / float8 / numeric
    def g_f4(self, d):
        r = self.r
        k = r.random()
        if k < 0.6:
            op = r.choice(["+", "-", "*", "/"])
            return N("({0} " + op + " {1})", [self.expr("f4", d - 1), self.expr("f4", d - 1)], "f4", pr=True)
        if k < 0.7:
            return N("(-{0})", [self.expr("f4", d - 1)], "f4")
        if k < 0.9:
            return self.cast(self.expr(r.choice(INTS + (("num",) if self.allow else ())), d - 1), "float4")
        return N("abs({0})", [self.expr("f4", d - 1)], "f4")

    def g_f8(self, d):
        r = self.r
        if r.random() < 0.5:
            return self.cast(self.expr("f4", d - 1), "float8")
        return N("({0} " + r.choice(["+", "*", "-"]) + " {1})", [self.expr("f4", d - 1), self.expr("i4", d - 1)], "f8", pr=True)

    def g_num(self, d):
        r = self.r
        k = r.random()
        if k < 0.55:
            op = r.choice(["+", "-", "*", "*", "/", "%"] if self.u(0.4) else ["+", "-", "*"])
            a = self.expr("num", d - 1)
            b = self.expr("num" if r.random() < 0.7 else r.choice(INTS), d - 1)
            return N("({0} " + op + " {1})", [a, b], "num", pr=True)
        if k < 0.65:
            return N("(-{0})", [self.expr("num", d - 1)], "num")
        if k < 0.72:
            return N("abs({0})", [self.expr("num", d - 1)], "num")
        if k < 0.88:
            return self.cast(self.expr(r.choice(INTS), d - 1), "numeric")
        if k < 0.94 and self.u(0.8):
            return N("round({0}, " + str(r.choice([0, 1, 2, -1])) + ")", [self.expr("num", d - 1)], "num")
        if k < 0.97 and self.u(0.8):
            return N("({0})::numeric(10,2)", [self.expr("num", d - 1)], "num", pr=True)
        if self.u(0.8):
            return self.cast(self.expr("f4", d - 1), "numeric")
        return self.cast(self.expr(r.choice(INTS), d - 1), "numeric")

    # ---- temporal
    def g_date(self, d):
        r = self.r
        k = r.random()
        if k < 0.35:
            return N("({0} + {1})", [self.expr("date", d - 1), self.expr("i4", d - 1)], "date", pr=True)
        if k < 0.5:
            return N("({0} + {1})", [self.expr("i4", d - 1), self.expr("date", d - 1)], "date", pr=True)
        if k < 0.7:
            return N("({0} - {1})", [self.expr("date", d - 1), self.expr("i4", d - 1)], "date", pr=True)
        if k < 0.85:
            return N("({0})::date", [self.expr("ts", d - 1)], "date", pr=True)
        if k < 0.92 and self.u(0.7):
            return N("({0})::date", [self.expr("tz", d - 1)], "date", pr=True)
        if not self.u(0.7):
            return N("({0} + {1})", [self.expr("date", d - 1), self.expr("i4", d - 1)], "date", pr=True)
        return N("date_trunc('month', {0})::date", [self.expr("ts", d - 1)], "date", pr=True)

    def g_ts(self, d):
        r = self.r
        k = r.random()
        if k < 0.6:
            return N("({0})::timestamp", [self.expr("date", d - 1)], "ts", pr=True)
        if k < 0.8 and self.u(0.7):
            return N("({0} + interval '1 day')", [self.expr("ts", d - 1)], "ts", pr=True)
        if self.u(0.7):
            return N("({0})::timestamp", [self.expr("tz", d - 1)], "ts", pr=True)
        return self.cast(self.expr("date", d - 1), "timestamp")

    def g_tz(self, d):
        r = self.r
        if r.random() < 0.5 or not self.u(0.7):
            return N("({0})::timestamptz", [self.expr("ts", d - 1)], "tz", pr=True)
        return N("({0} - interval '1 hour')", [self.expr("tz", d - 1)], "tz", pr=True)

    def g_time(self, d):
        if not self.u(0.7):
            return self.leaf("time")
        return N("({0} + interval '1 minute')", [self.expr("time", d - 1)], "time")

    # ---- text
    def g_text(self, d):
        r = self.r
        k = r.random()
        if not self.u(0.7):
            return self.leaf("text")
        if k < 0.3:
            return N("upper({0})", [self.expr("text", d - 1)], "text")
        if k < 0.5:
            return N("({0} || {1})", [self.expr("text", d - 1), self.expr("text", d - 1)], "text")
        if k < 0.7:
            return N("substr({0}, " + str(r.choice([1, 2, 3])) + ")", [self.expr("text", d - 1)], "text")
        if k < 0.85:
            return N("({0})::text", [self.expr(r.choice(["i4", "i8", "date", "bool"]), d - 1)], "text", pr=True)
        return N("({0})::varchar(8)", [self.expr("text", d - 1)], "text", pr=True)

    def g_bpchar(self, d):
        if not self.u(0.5):
            return self.leaf("bpchar")
        return N("({0})::char(6)", [self.expr("text", d - 1)], "bpchar", pr=True)


# ------------------------------------------------------------------ queries


class Q:
    """a query: kind in scan, agg, group, join, joinagg, semi"""

    def __init__(self):
        self.kind = "scan"
        self.table = "fz1"
        self.projs = []          # N
        self.where = []          # N (ANDed)
        self.aggs = []           # (func, arg N or None, filter N or None)
        self.keys = []           # N
        self.join = None         # dict: jt, on (list of N), keys
        self.settings = ""       # settings group name
        self.order_limit = None
        self.distinct = False
        self.strict = False

    def clone(self):
        q = Q()
        q.__dict__.update(self.__dict__)
        q.projs = list(self.projs)
        q.where = list(self.where)
        q.aggs = list(self.aggs)
        q.keys = list(self.keys)
        q.join = dict(self.join) if self.join else None
        if q.join:
            q.join["on"] = list(q.join["on"])
        return q

    def sql(self):
        j = self.join
        frm = self.table + " t"
        wh = list(self.where)
        if j:
            if j["jt"] in ("exists", "not exists", "in", "not in"):
                cond = " AND ".join(x.sql() for x in j["on"])
                if j["jt"] in ("exists", "not exists"):
                    wh.append(N(j["jt"].upper() + " (select 1 from fz_d d where " + cond.replace("{", "{{").replace("}", "}}") + ")", [], "bool"))
                else:
                    wh.append(N("{0} " + j["jt"].upper() + " (select d.k4 from fz_d d)", [j["lhs"]], "bool"))
            else:
                frm += " " + j["jt"] + " join fz_d d on " + " AND ".join(x.sql() for x in j["on"])
        w = ""
        if wh:
            w = " where " + " AND ".join(x.sql() for x in wh)
        if self.kind in ("scan", "semi", "join"):
            sel = ", ".join(p.sql() + " as c%d" % i for i, p in enumerate(self.projs)) or "t.i"
            return "select " + ("distinct " if self.distinct else "") + sel + " from " + frm + w
        parts = []
        for i, k in enumerate(self.keys):
            parts.append(k.sql() + " as k%d" % i)
        for i, (f, a, flt) in enumerate(self.aggs):
            s = f + "(" + (a.sql() if a is not None else "*") + ")"
            if flt is not None:
                s += " filter (where " + flt.sql() + ")"
            parts.append(s + " as a%d" % i)
        s = "select " + ", ".join(parts) + " from " + frm + w
        if self.keys:
            s += " group by " + ", ".join(str(i + 1) for i in range(len(self.keys)))
        return s

    def has_anything(self):
        return bool(self.projs or self.aggs or self.keys)


SETTINGS = {
    "scan": "set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;",
    "agg": "set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on; "
           "set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;",
    "join": "set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on; "
            "set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; "
            "set enable_mergejoin=off; set enable_nestloop=off;",
}
PAR = " set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
CPUSET = "set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"


# ------------------------------------------------------------------ stress shapes (translation limits)
# fz_w has w1..w44 int4 (small values), w45..w50 int8, w51..w54 text, w55..w58 float4, w59 date, w60 numeric(10,2)
WINT = ["w%d" % i for i in range(1, 45)]
WBIG = ["w%d" % i for i in range(45, 51)]
WTXT = ["w%d" % i for i in range(51, 55)]
WFLT = ["w%d" % i for i in range(55, 59)]


def wcol(r, typ="i4"):
    if typ == "i4":
        return col(r.choice(WINT), "i4")
    if typ == "i8":
        return col(r.choice(WBIG), "i8")
    return col(r.choice(WFLT), "f4")


def chain(r, depth, shape):
    """a deep arithmetic expression over fz_w columns: left chain, right chain, balanced, or random"""
    def bal(d):
        if d <= 0:
            return wcol(r) if r.random() < 0.8 else lit("(%d)::int4" % r.choice([1, 2, 3, -1, 7]), "i4")
        return N("({0} " + r.choice("+-") + " {1})", [bal(d - 1), bal(d - 1)], "i4", pr=True)
    if shape == "left":
        e = wcol(r)
        for _ in range(depth):
            e = N("({0} " + r.choice("+-*") + " {1})", [e, wcol(r) if r.random() < 0.7 else lit("(%d)::int4" % r.choice([1, 2, 3]), "i4")], "i4", pr=True)
        return e
    if shape == "right":
        e = wcol(r)
        for _ in range(depth):
            e = N("({0} " + r.choice("+-") + " {1})", [wcol(r) if r.random() < 0.7 else lit("(%d)::int4" % r.choice([1, 2, 3]), "i4"), e], "i4", pr=True)
        return e
    if shape == "bal":
        return bal(min(depth, 9))
    e = wcol(r)
    for _ in range(depth):
        o = wcol(r)
        e = N("({0} " + r.choice("+-") + " {1})", [e, o] if r.random() < 0.5 else [o, e], "i4", pr=True)
    return e


def bchain(r, n, op):
    """AND/OR chain of n comparisons, as one expression"""
    parts = [N("({0} " + r.choice(["<", ">", "<>", "=", ">="]) + " {1})", [wcol(r), lit("(%d)::int4" % r.choice([-5, 0, 3, 8]), "i4")], "bool") for _ in range(n)]
    e = parts[0]
    for x in parts[1:]:
        e = N("({0} " + op + " {1})", [e, x], "bool")
    return e


def gen_stress(r, index):
    q = Q()
    q.table = "fz_w"
    q.kind = r.choice(["scan", "scan", "agg", "group", "join", "joinagg"])
    sub = r.choice(["wide", "manyproj", "deepproj", "deepqual", "manyqual", "manyagg", "bigsum", "manyjoin", "allcols"])
    isjoin = q.kind in ("join", "joinagg")
    if isjoin:
        j = r.choice(["inner", "left", "inner"])
        lhs = col("t.w1", "i4") if False else lit("t.w1", "i4")
        q.join = {"jt": j, "on": [N("(t.w2 = d.k4)", [], "bool")], "lhs": lhs}
    pfx = "t." if isjoin else ""
    def W(typ="i4"):
        c = wcol(r, typ)
        return col(pfx + c.fmt, typ)
    def chain_p(depth, shape):
        e = chain(r, depth, shape)
        return _prefix(e, pfx)
    if q.kind in ("scan", "join"):
        if sub in ("wide", "allcols"):
            n = r.choice([8, 16, 24, 31, 32, 33, 40, 55, 60]) if sub == "wide" else 60
            names = ["w%d" % i for i in range(1, 61)]
            r.shuffle(names)
            for nm in names[:n]:
                q.projs.append(col(pfx + nm, "any"))
            if r.random() < 0.5:
                q.where.append(N("({0} > {1})", [W(), lit("(%d)::int4" % r.choice([-3, 0, 3]), "i4")], "bool"))
        elif sub == "manyproj":
            n = r.choice([6, 10, 12, 13, 16, 20, 24, 32, 40])
            for _ in range(n):
                q.projs.append(N("({0} + {1})", [W(), W()], "i4", pr=True))
            if r.random() < 0.5:
                q.where.append(N("({0} > {1})", [W(), W()], "bool"))
        elif sub == "deepproj":
            for _ in range(r.choice([1, 2, 4])):
                q.projs.append(chain_p(r.choice([4, 6, 8, 12, 20, 40]), r.choice(["left", "right", "bal", "rand"])))
        elif sub == "deepqual":
            q.where.append(N("({0} > {1})", [chain_p(r.choice([4, 8, 12, 20, 30]), r.choice(["left", "right", "bal", "rand"])), W()], "bool"))
            q.projs.append(W())
        elif sub == "manyqual":
            for _ in range(r.choice([5, 10, 20, 35])):
                q.where.append(N("({0} " + r.choice(["<", ">", "<>"]) + " {1})", [W(), W()], "bool"))
            q.projs.append(W())
            if r.random() < 0.5:
                q.where = [_prefix(bchain(r, r.choice([5, 12, 24, 40]), r.choice(["AND", "OR"])), pfx)]
        else:
            n = r.choice([8, 16, 24, 33, 44])
            e = W()
            for k in range(n - 1):
                e = N("({0} + {1})", [e, col(pfx + "w%d" % (k + 2), "i4")], "i4", pr=True)
            q.projs = [e]
    else:
        n = r.choice([1, 3, 6, 11, 12, 13, 14, 20, 30])
        if sub in ("deepproj", "deepqual"):
            n = r.choice([1, 2, 3])
        for _ in range(n):
            f = r.choice(["count", "sum", "min", "max", "avg"])
            if sub in ("deepproj", "deepqual"):
                a = chain_p(r.choice([4, 8, 14, 24]), r.choice(["left", "right", "bal", "rand"]))
            elif sub == "bigsum":
                a = N("({0} + {1})", [W(), W()], "i4", pr=True)
            else:
                a = W() if r.random() < 0.5 else N("({0} - {1})", [W(), W()], "i4", pr=True)
            q.aggs.append((f, a, N("({0} > {1})", [W(), lit("(0)::int4", "i4")], "bool") if r.random() < 0.3 else None))
        if q.kind in ("group", "joinagg") and (q.kind == "group" or r.random() < 0.5):
            for _ in range(r.choice([1, 2, 2, 3, 4])):
                q.keys.append(N("({0} % 5)", [W()], "i4", pr=True))
        if sub in ("manyqual", "wide"):
            for _ in range(r.choice([3, 12, 30])):
                q.where.append(N("({0} " + r.choice(["<", ">", "<>"]) + " {1})", [W(), W()], "bool"))
    if q.kind == "group" and not q.keys:
        q.keys.append(N("({0} % 5)", [wcol(r)], "i4", pr=True))
    q.settings = ("scan" if q.kind == "scan" else "agg" if q.kind in ("agg", "group") else "join") + ("+par" if r.random() < 0.3 else "")
    q.strict = False
    return q


def _prefix(e, pfx):
    if not pfx or e.leaf is False and not e.args:
        return e
    if e.leaf and e.args == [] and re.fullmatch(r"w\d+", e.fmt):
        return col(pfx + e.fmt, e.typ)
    ne = N(e.fmt, [_prefix(a, pfx) for a in e.args], e.typ, e.leaf, e.pr)
    return ne


def gen_query(seed, index):
    r = random.Random((seed * 1000003) ^ (index * 7919 + 17))
    if r.random() < STRESS_RATE:
        return gen_stress(r, index)
    q = Q()
    q.table = r.choice(["fz1", "fz1", "fz2", "fz3"])
    kind = r.choice(["scan", "scan", "scan", "agg", "agg", "group", "group", "join", "join", "semi", "joinagg"])
    q.kind = kind
    tabs = [("", COLS)]
    isjoin = kind in ("join", "semi", "joinagg")
    if isjoin:
        tabs = [("t.", COLS), ("d.", DCOLS)] if kind != "semi" else [("t.", COLS)]
    else:
        tabs = [("", COLS)]
    g = Gen(r, tabs)
    depth = r.choice([1, 2, 2, 2, 3, 3, 4])

    def tcols(typ):                  # an expression over the outer table only
        gt = Gen(r, [("t.", COLS)] if isjoin else [("", COLS)])
        return gt.expr(typ, depth)

    nwhere = r.choice([0, 1, 1, 1, 2, 3])
    gw = g
    for _ in range(nwhere):
        q.where.append(gw.expr("bool", depth))
    if isjoin:
        jt = {"semi": r.choice(["exists", "not exists", "in", "not in"]),
              "join": r.choice(["inner", "inner", "left", "right", "full"]),
              "joinagg": r.choice(["inner", "inner", "left", "inner"])}[kind]
        kt = r.choice(["i4", "i4", "i8", "date", "i2", "ts", "i4"])
        gt = Gen(r, [("t.", COLS)])
        if kt in ("i4", "i2"):
            lhs, dk = gt.expr(kt, min(depth, 2)), "d.k4"
        elif kt == "i8":
            lhs, dk = gt.expr("i8", min(depth, 2)), "d.k8"
        elif kt == "date":
            lhs, dk = gt.expr("date", min(depth, 2)), "d.kd"
        else:
            lhs, dk = gt.expr("ts", min(depth, 2)), "d.kts"
        if r.random() < 0.08:        # a key on the CPU only (text)
            lhs, dk = gt.expr("text", 1), "d.ks"
        on = [N("({0} = " + dk + ")", [lhs], "bool")]
        for _ in range(r.choice([0, 0, 1, 1, 2])):
            on.append(g.expr("bool", min(depth, 2)))
        if jt in ("exists", "not exists", "in", "not in"):
            # correlated subquery: only the outer table's columns on one side
            gx = Gen(r, [("t.", COLS), ("d.", DCOLS)])
            on = [N("({0} = " + dk + ")", [lhs], "bool")]
            for _ in range(r.choice([0, 1, 1])):
                on.append(gx.expr("bool", 2))
            q.join = {"jt": jt, "on": on, "lhs": lhs}
        else:
            q.join = {"jt": jt, "on": on, "lhs": lhs}
    nproj = r.choice([1, 1, 2, 3, 4, 6])
    if kind in ("scan", "join", "semi"):
        for _ in range(nproj):
            if r.random() < 0.12:
                q.projs.append(g.leaf(r.choice(["text", "i4", "date", "num", "bool"])))
            else:
                q.projs.append(g.expr(r.choice(["i2", "i4", "i4", "i8", "i8", "f4", "num", "num", "date", "ts", "bool", "bool", "text", "time", "tz"]),
                                      depth))
        if isjoin and kind == "semi":
            pass
    else:
        naggs = r.choice([1, 2, 2, 3, 4, 6])
        for _ in range(naggs):
            f = r.choice(["count", "count", "sum", "sum", "min", "max", "avg", "count*"])
            if f == "count*":
                a = None
                f = "count"
            else:
                at = r.choice(["i2", "i4", "i4", "i8", "i8", "i8", "num", "f4", "date", "ts", "time", "text", "bool"] if f in ("count", "min", "max")
                              else ["i2", "i4", "i4", "i8", "i8", "num"])   # sum/avg of float4 depends on the order the rows arrive in
                a = g.expr(at, depth)
            flt = g.expr("bool", min(depth, 2)) if r.random() < 0.3 else None
            q.aggs.append((f, a, flt))
        if kind in ("group", "joinagg") and (kind == "group" or r.random() < 0.6):
            nk = r.choice([1, 1, 1, 2, 2, 3])
            for _ in range(nk):
                kt = r.choice(["i2", "i4", "i4", "i8", "date", "bool", "ts", "time", "text", "num"])
                kx = g.expr(kt, min(depth, 3))
                if kt == "num":
                    # a numeric key's printed scale depends on the row that created the group (0 vs 0.00):
                    # only fixed-scale columns, or the result would depend on the scan order
                    kx = col(("t." if isjoin else "") + r.choice(["nm", "nl"]), "num")
                if kt == "f4":
                    kx = col(("t." if isjoin else "") + "nf", "f4")
                if kt in INTS and r.random() < 0.8:
                    kx = N("(({0}) % {1})", [kx, lit(str(r.choice([2, 3, 5, 10, 100, 1000])), kt)], kt, pr=True) if r.random() < 0.8 else kx
                q.keys.append(kx)
        if kind == "group" and not q.keys:
            q.keys.append(g.expr(r.choice(["i2", "i4", "bool", "date"]), 1))
    if kind in ("scan", "join") and r.random() < 0.08 and all(
            x.typ in ("i2", "i4", "i8", "date", "bool", "text", "time", "ts", "bpchar") for x in q.projs):
        q.distinct = True       # (float and numeric values that compare equal print differently: 0 and 0.00, -0)
    # settings
    par = r.random() < 0.4
    sg = "scan" if kind == "scan" else ("agg" if kind in ("agg", "group") else "join")
    q.settings = sg + ("+par" if par else "")
    # strict: one expression tree, every operand evaluated on every row
    txt = " ".join([x.sql() for x in q.where + q.projs + q.keys] +
                   [(x[1].sql() if x[1] is not None else "") + (x[2].sql() if x[2] is not None else "") for x in q.aggs])
    q.strict = (not isjoin and len(q.where) <= 1 and
                not any(k in txt for k in (" AND ", " OR ", "COALESCE", " IN (", "BETWEEN", "NOT (")))
    return q


def settings_text(name, weak=False):
    base, _, par = name.partition("+")
    s = SETTINGS[base]
    if par:
        s += PAR
    return s


# ------------------------------------------------------------------ running psql


class Runner:
    def __init__(self, psql, sock, port, db, timeout=120):
        self.cmd = [psql, "-h", sock, "-p", str(port), "-U", "postgres", "-X", "-At", "-q", "-d", db,
                    "-v", "ON_ERROR_STOP=0", "-v", "VERBOSITY=terse"]
        self.timeout = timeout

    def run(self, script):
        try:
            p = subprocess.run(self.cmd, input=script, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return "", "ERROR:  timeout of the fuzzer"
        return p.stdout, p.stderr

    def result(self, settings, sql, explain=False, full=False):
        """-> (plan text, ok, rows-or-error); the query is wrapped to return one md5 line and a count"""
        sc = "set statement_timeout='90s'; set extra_float_digits=1; " + settings + "\n"
        if explain:
            sc += "explain (costs off) " + sql + ";\n\\echo @@\n"
        if full:
            sc += "select x::text from (" + sql + ") x order by 1 limit 40;\n"
        else:
            sc += ("select md5(coalesce(string_agg(x::text, E'\\n' order by x::text), '')), count(*) from (" + sql + ") x;\n")
        out, err = self.run(sc)
        plan = ""
        body = out
        if explain:
            plan, _, body = out.partition("@@\n")
        errs = [l for l in err.splitlines() if l.startswith("ERROR")]
        if errs:
            return plan, False, errs[0]
        if err.strip() and not body.strip():
            # no result and no ERROR: the connection was lost (a crashed backend or service)
            return plan, False, "ERROR:  fuzzer: no result: " + err.strip().replace("\n", " | ")[:300]
        if "SET" in body.split("\n")[0] and False:
            pass
        return plan, True, body.strip()


ERR_NUM = re.compile(r"[0-9]+")


def invalid_query(msg):
    return ("does not exist" in msg or "cannot be matched" in msg or "could not determine" in msg or
            "is not unique" in msg or "must be type" in msg or "syntax error" in msg or "cannot cast" in msg or
            "argument of" in msg or "invalid input syntax" in msg or "regular expression" in msg or
            "operator does not" in msg or "is ambiguous" in msg or "must appear in the GROUP BY" in msg or
            "invalid escape" in msg or "LIKE pattern" in msg)


# messages of scan_prog.c's fail() (the service refused to translate the program), see scan_prog.c
SERVICE_ERR = ("registers", "kernel variables", "by the Vulkan", "by this service", "variable slot", "program is too large",
               "does not have the type the expression expects", "has no value", "malformed", "output columns (up to",
               "GROUP BY keys (up to", "of an unsupported type", "unexpected ", "not loaded from the scan relation",
               "is out of range", "too many", "cannot create the", "cannot allocate", "dispatch failed",
               "not a GpuJoin session", "join and sort", "does not exist")
PREFIX = re.compile(r"^ERROR:  gpuserv:\d+\s+")
SUFFIX = re.compile(r"\s+\(in [^)]*\)\s*$")


def norm_err(msg):
    """error text without the service's prefix and its location suffix"""
    m = PREFIX.sub("ERROR:  ", msg)
    return SUFFIX.sub("", m)


def is_service_error(msg):
    low = norm_err(msg).lower()
    if "fuzzer: no result" in msg or "terminating connection" in msg:
        return True				# a crashed backend or service
    return msg.startswith("ERROR:  gpuserv") and any(k.lower() in low for k in SERVICE_ERR) and "integer out of range" not in low \
        and "smallint out of range" not in low and "bigint out of range" not in low


def classify(run, q, quick=False):
    """-> (verdict, detail). verdict: ok, same_error, diff_error, asym, invalid, fail_rows, fail_service, fail_asym, unstable"""
    sql = q.sql()
    gset = settings_text(q.settings)
    plan, gok, gres = run.result(gset, sql, explain=True)
    cplan, cok, cres = run.result(CPUSET, sql)
    q.plan = plan
    if gok and cok:
        if gres == cres:
            return "ok", (gres, cres)
        _, ok2, r2 = run.result(gset, sql)
        if ok2 and r2 != gres:
            return "unstable", (gres, cres)
        return "fail_rows", (gres, cres)
    if not gok and not cok:
        if invalid_query(cres) or invalid_query(gres):
            return "invalid", (gres, cres)
        if is_service_error(gres) and not is_service_error(cres):
            return "fail_service", (gres, cres)
        a = ERR_NUM.sub("#", norm_err(gres))
        b = ERR_NUM.sub("#", norm_err(cres))
        return ("same_error" if a == b else "diff_error"), (gres, cres)
    # one side only
    if not gok:
        if is_service_error(gres):
            return "fail_service", (gres, cres)
    if (not gok and invalid_query(gres)) or (not cok and invalid_query(cres)):
        return "invalid", (gres, cres)
    return ("fail_asym" if q.strict else "asym"), (gres, cres)


FAILS = ("fail_rows", "fail_service", "fail_asym")

# ------------------------------------------------------------------ minimizer


def shrink_candidates(q):
    """smaller variants of the query"""
    out = []
    for i in range(len(q.projs)):
        if len(q.projs) > 1:
            c = q.clone(); del c.projs[i]; out.append(c)
    for i in range(len(q.where)):
        c = q.clone(); del c.where[i]; out.append(c)
    for i in range(len(q.aggs)):
        if len(q.aggs) > 1:
            c = q.clone(); del c.aggs[i]; out.append(c)
        f, a, flt = q.aggs[i]
        if flt is not None:
            c = q.clone(); c.aggs[i] = (f, a, None); out.append(c)
    for i in range(len(q.keys)):
        if len(q.keys) > 1 or q.kind == "joinagg":
            c = q.clone(); del c.keys[i]; out.append(c)
    if q.join:
        for i in range(1, len(q.join["on"])):
            c = q.clone(); del c.join["on"][i]; out.append(c)
    # subexpression replacement
    def variants(e):
        res = []
        for i, a in enumerate(e.args):
            for v in variants(a):
                ne = N(e.fmt, e.args, e.typ, e.leaf, e.pr)
                ne.args = list(e.args); ne.args[i] = v
                res.append(ne)
        for a in e.args:
            if a.typ == e.typ:
                res.append(a)
        if not e.leaf:
            for t in ("0", "1"):
                if e.typ in INTS:
                    res.append(lit("(" + t + ")::" + INTSQL[e.typ], e.typ))
            if e.typ == "bool":
                res.append(lit("true", "bool"))
        return res
    for i, p in enumerate(q.projs):
        for v in variants(p):
            c = q.clone(); c.projs[i] = v; out.append(c)
    for i, p in enumerate(q.where):
        for v in variants(p):
            c = q.clone(); c.where[i] = v; out.append(c)
    for i, (f, a, flt) in enumerate(q.aggs):
        if a is not None:
            for v in variants(a):
                c = q.clone(); c.aggs[i] = (f, v, flt); out.append(c)
        if flt is not None:
            for v in variants(flt):
                c = q.clone(); c.aggs[i] = (f, a, v); out.append(c)
    for i, p in enumerate(q.keys):
        for v in variants(p):
            c = q.clone(); c.keys[i] = v; out.append(c)
    return out


def minimize(run, q, verdict, budget=120):
    cur = q
    tries = 0
    improved = True
    while improved and tries < budget:
        improved = False
        cands = shrink_candidates(cur)
        cands.sort(key=lambda c: len(c.sql()))
        for c in cands:
            if len(c.sql()) >= len(cur.sql()):
                continue
            if not c.has_anything():
                continue
            tries += 1
            if tries > budget:
                break
            v, _ = classify(run, c)
            if v == verdict:
                cur = c
                improved = True
                break
    return cur


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--db", default="fuzz")
    ap.add_argument("--psql", default=os.environ.get("PSQL_BIN", "/opt/homebrew/opt/postgresql@17/bin/psql"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("FUZZ_SEED", "1")))
    ap.add_argument("--count", type=int, default=int(os.environ.get("FUZZ_COUNT", "200")))
    ap.add_argument("--time", type=float, default=float(os.environ.get("FUZZ_TIME", "0")))
    ap.add_argument("--corpus", default=os.environ.get("FUZZ_CORPUS", ""))
    ap.add_argument("--replay", default="")
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--no-minimize", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--samples", type=int, default=int(os.environ.get("FUZZ_SAMPLES", "3")))
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    run = Runner(a.psql, a.sock, a.port, a.db)
    tally = {}
    plans = {}
    samples = {}
    nfail = 0
    t0 = time.time()
    done = 0

    def note_plan(q):
        p = getattr(q, "plan", "")
        key = "+".join(sorted(set(re.findall(r"Custom Scan \((Gpu\w+)\)", p)))) or "cpu-only"
        plans[key] = plans.get(key, 0) + 1

    if a.replay:
        # corpus lines: settings TAB sql  (one query per line; results must be equal or both errors)
        bad = 0
        n = 0
        for line in open(a.replay, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            st, _, sql = line.partition("\t")
            q = Q()
            q.sql = lambda sql=sql: sql
            q.settings = st
            n += 1
            v, d = classify(run, q)
            tally[v] = tally.get(v, 0) + 1
            if v in FAILS:
                bad += 1
                print("REPLAY FAIL (%s): %s\n  gpu: %s\n  cpu: %s" % (v, sql, d[0], d[1]))
        print("replay: %d queries, %d failing %s" % (n, bad, tally))
        return 1 if bad else 0

    idx = a.start
    while True:
        if a.only >= 0:
            if done:
                break
            idx = a.only
        elif done >= a.count:
            break
        if a.time and time.time() - t0 > a.time:
            break
        q = gen_query(a.seed, idx)
        v, d = classify(run, q)
        tally[v] = tally.get(v, 0) + 1
        if v in ("asym", "diff_error", "invalid", "unstable") and len(samples.setdefault(v, [])) < a.samples:
            samples[v].append((idx, q.settings, q.sql(), d[0], d[1]))
        note_plan(q)
        if a.verbose:
            print("[%d] %s %s: %s" % (idx, v, q.settings, q.sql()))
        if v in FAILS or v == "unstable":
            nfail += 1 if v in FAILS else 0
            print("=" * 72)
            print("%s  seed=%d index=%d settings=%s" % (v.upper(), a.seed, idx, q.settings))
            print("query: " + q.sql())
            print("GPU: " + d[0])
            print("CPU: " + d[1])
            if v in FAILS and not a.no_minimize:
                m = minimize(run, q, v)
                mv, md = classify(run, m)
                print("minimized: " + m.sql())
                print("  GPU: " + md[0])
                print("  CPU: " + md[1])
                if a.corpus:
                    with open(a.corpus, "a", encoding="utf-8") as f:
                        f.write("# %s seed=%d index=%d GPU: %s | CPU: %s\n%s\t%s\n" % (
                            v, a.seed, idx, md[0][:100].replace("\n", " "), md[1][:100].replace("\n", " "),
                            m.settings, m.sql().replace("\n", " ")))
            sys.stdout.flush()
        done += 1
        idx += 1
    el = time.time() - t0
    for v, lst in sorted(samples.items()):
        print("samples of %s:" % v)
        for (i, st, sql, g, c) in lst:
            print("  [%d] %s: %s" % (i, st, sql[:600]))
            print("      GPU: %s" % g[:200])
            print("      CPU: %s" % c[:200])
    print("-" * 72)
    print("fuzz: seed=%d queries=%d (%.0fs, %.1f q/s)" % (a.seed, done, el, done / max(el, 1e-9)))
    print("  results: " + ", ".join("%s=%d" % kv for kv in sorted(tally.items())))
    print("  GPU plan nodes: " + ", ".join("%s=%d" % kv for kv in sorted(plans.items())))
    nbad = sum(tally.get(k, 0) for k in FAILS)
    inv = tally.get("invalid", 0)
    print("  failing: %d, invalid (generator mistakes): %d" % (nbad, inv))
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
