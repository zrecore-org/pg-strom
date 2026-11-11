#!/usr/bin/env python3
# Second bisect step: inside SOP_NUMARITH of vm_num (the case whose presence makes the T4 reject the scan kernel). Each variant
# edits scan_common.slang by exact text replacement; a few are candidate fixes. env: SLANGC, GPUSERV_BIN, BISECT_OUT, DEVICE
import os, shutil, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, '..', '..', 'src', 'gpuserv')
out = os.environ.get('BISECT_OUT', '/tmp/nvbisect3')
slangc = os.environ['SLANGC']; gbin = os.environ['GPUSERV_BIN']
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
kv = open(os.path.join(src, 'kernels_vm', 'scan_common.slang')).read()
FLAGS = ['-DVM_MAX_REGS=8', '-DVM_NO_JSON', '-DVM_NO_F8', '-DVM_NO_GEOM']
MUL = "if (ma != 0 && mb > 9223372036854775807 / ma)"
PL = "if ((b > 0 && a > 9223372036854775807 - b) || (b < 0 && a < -9223372036854775807 - b))"
MI = "if ((b < 0 && a > 9223372036854775807 + b) || (b > 0 && a < -9223372036854775807 + b))"
INEX = '    bool inexact = (x.sc >= 20u) ? (m != 0ul) : (m != q * num_p10u(x.sc));\n'
FIN = '    if (x.v >= 0)\n        r = int64_t(q) + ((k0 == SOP_NUMOP_CEIL && inexact) ? 1 : 0);\n    else\n        r = -int64_t(q) - ((k0 == SOP_NUMOP_FLOOR && inexact) ? 1 : 0);\n'
SIGN = '    if (k0 == SOP_NUMOP_SIGN)\n    {\n        r = (x.v > 0) ? 1 : ((x.v < 0) ? -1 : 0);\n        return;\n    }\n'
variants = [
    ('U2_no_inexact', [(INEX, "    bool inexact = false;\n")]),
    ('U3_inexact_simple', [(INEX, "    bool inexact = (m != q);\n")]),
    ('U4_no_sc20', [(INEX, "    bool inexact = (m != q * num_p10u(x.sc));\n")]),
    ('U5_no_down', [("    uint64_t q = num_down(m, x.sc, false);\n", "    uint64_t q = m;\n")]),
    ('U6_no_sign', [(SIGN, "")]),
    ('U7_simple_final', [(FIN, "    r = (x.v >= 0) ? int64_t(q) : -int64_t(q);\n")]),
    ('U8_ifs_final', [(FIN, "    if (x.v >= 0)\n    {\n        r = int64_t(q);\n        if (k0 == SOP_NUMOP_CEIL && inexact)\n            r += 1;\n    }\n    else\n    {\n        r = -int64_t(q);\n        if (k0 == SOP_NUMOP_FLOOR && inexact)\n            r -= 1;\n    }\n")]),
    ('U9_back', [(INEX, "    uint64_t back = (x.sc >= 20u) ? 0ul : q * num_p10u(x.sc);\n    bool inexact = (m != back);\n")]),
    ('U10_u8_u9', [(INEX, "    uint64_t back = (x.sc >= 20u) ? 0ul : q * num_p10u(x.sc);\n    bool inexact = (m != back);\n"),
                   (FIN, "    if (x.v >= 0)\n    {\n        r = int64_t(q);\n        if (k0 == SOP_NUMOP_CEIL && inexact)\n            r += 1;\n    }\n    else\n    {\n        r = -int64_t(q);\n        if (k0 == SOP_NUMOP_FLOOR && inexact)\n            r -= 1;\n    }\n")]),
    ('U11_sign_if', [(SIGN, "    if (k0 == SOP_NUMOP_SIGN)\n    {\n        int64_t sg = 0;\n        if (x.v > 0)\n            sg = 1;\n        else if (x.v < 0)\n            sg = -1;\n        r = sg;\n        return;\n    }\n")]),
]
def build(name, edits, flags, target='scan'):
    text = kv
    for old, new in edits:
        if text.count(old) != 1:
            print('  (%s: edit text found %d times)' % (name, text.count(old))); return
        text = text.replace(old, new)
    d = os.path.join(out, name); shutil.copytree(os.path.join(src, 'kernels_vm'), d)
    open(os.path.join(d, 'scan_common.slang'), 'w').write(text)
    spv = os.path.join(out, name + '.spv')
    if target == 'scan':
        cmd = [slangc, os.path.join(d, 'scan.slang'), '-I', d] + flags + ['-target', 'spirv', '-entry', 'scan_main', '-stage', 'compute', '-o', spv]
        nb, pc = '5', '16'
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('  SLANGC-FAIL %s: %s' % (name, r.stderr[:300])); return
    try:
        p = subprocess.run([gbin, '--prewarm', nb, pc, spv, '--device', os.environ.get('DEVICE', '0')],
                           capture_output=True, text=True, timeout=90, env=dict(os.environ, PGSTROM_GPUSERV_PIPELINE_CACHE='off'))
        ok = p.returncode == 0; msg = (p.stdout + p.stderr).replace('\n', ' ')[:100]
    except subprocess.TimeoutExpired:
        ok = False; msg = 'timeout'
    print('  %s %s (%d bytes)%s' % ('OK  ' if ok else 'FAIL', name, os.path.getsize(spv), '' if ok else ': ' + msg)); sys.stdout.flush()
    shutil.rmtree(d)
for name, edits in variants:
    build(name, edits, FLAGS)
