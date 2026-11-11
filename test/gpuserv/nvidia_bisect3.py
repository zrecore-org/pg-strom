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
variants = [
    ('A_mulcheck_false', [(MUL, "if (false)")]),
    ('F1_mul_udiv', [(MUL, "if (ma != 0 && uint64_t(mb) > 9223372036854775807ul / uint64_t(ma))")]),
    ('B_negabs', [("regs[dst].v = (k0 == SOP_NUMOP_NEG || x.v < 0) ? -x.v : x.v;", "regs[dst].v = x.v;")]),
    ('C_unary', [("num_unary(k0, x, ur);", "ur = 0;")]),
    ('D_numup', [("if (!num_up(a, sc - x.sc) || !num_up(b, sc - y.sc))", "if (false)")]),
    ('E_pl', [(PL, "if (false)")]),
    ('G_mi', [(MI, "if (false)")]),
    ('H_final', [("if (r == (-9223372036854775807 - 1) || sc > 255u)", "if (sc > 255u)")]),
    ('I_pl_mi', [(PL, "if (false)"), (MI, "if (false)")]),
    ('J_mul_branch', [("            r = a * b;\n        }\n        else\n        {\n            sc = max(x.sc, y.sc);", "            r = 0;\n        }\n        else\n        {\n            sc = max(x.sc, y.sc);"),
                      (MUL, "if (false)")]),
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
