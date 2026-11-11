#!/usr/bin/env python3
# Bisect inside vm_num (kernels_vm/scan_common.slang) for the T4's "NVVM compilation failed" on the scan kernel with the numeric
# group: for each case of vm_num, a variant of the kernel without that case, and one with only that case. Prints OK / FAIL per
# variant (pipeline created by `pgstrom-gpuserv --prewarm`).   env: SLANGC, GPUSERV_BIN, BISECT_OUT, DEVICE
import os, re, shutil, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, '..', '..', 'src', 'gpuserv')
out = os.environ.get('BISECT_OUT', '/tmp/nvbisect2')
slangc = os.environ['SLANGC']; gbin = os.environ['GPUSERV_BIN']
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
kv = open(os.path.join(src, 'kernels_vm', 'scan_common.slang')).read()
a = kv.index('\nuint vm_num(')
b = kv.index('\nuint vm_json(')
body = kv[a:b]
labels = [(m.start(), m.group(1)) for m in re.finditer(r'^    (case SOP_\w+|default):', body, re.M)]
spans = []
for i, (pos, name) in enumerate(labels):
    end = labels[i + 1][0] if i + 1 < len(labels) else len(body)
    # the end of the function: the switch's closing brace stays with the last span; cut it at the first "\n    }\n"
    if i + 1 == len(labels):
        end = body.index('\n    }\n', pos) + 1
    spans.append((pos, end, name))

def build(name, text, flags):
    d = os.path.join(out, name)
    shutil.copytree(os.path.join(src, 'kernels_vm'), d)
    open(os.path.join(d, 'scan_common.slang'), 'w').write(text)
    spv = os.path.join(out, name + '.spv')
    r = subprocess.run([slangc, os.path.join(d, 'scan.slang'), '-I', d] + flags +
                       ['-target', 'spirv', '-entry', 'scan_main', '-stage', 'compute', '-o', spv],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print('  SLANGC-FAIL %s: %s' % (name, r.stderr[:300])); return
    try:
        p = subprocess.run([gbin, '--prewarm', '5', '16', spv, '--device', os.environ.get('DEVICE', '0')],
                           capture_output=True, text=True, timeout=90, env=dict(os.environ, PGSTROM_GPUSERV_PIPELINE_CACHE='off'))
        ok = p.returncode == 0
        msg = (p.stdout + p.stderr).replace('\n', ' ')[:120]
    except subprocess.TimeoutExpired:
        ok = False; msg = 'timeout'
    print('  %s %s (%d bytes)%s' % ('OK  ' if ok else 'FAIL', name, os.path.getsize(spv), '' if ok else ': ' + msg))
    sys.stdout.flush()
    shutil.rmtree(d)

FLAGS = ['-DVM_MAX_REGS=8', '-DVM_NO_JSON', '-DVM_NO_F8', '-DVM_NO_GEOM']
stub = lambda n: '    %s:\n        return SOP_RECHECK_CODE;\n' % n
build('full', kv, FLAGS)
for (s, e, n) in spans:
    nb = body[:s] + stub(n) + body[e:]
    build('without_' + n.replace('case ', ''), kv[:a] + nb + kv[b:], FLAGS)
for (s0, e0, n0) in spans:
    nb = body
    for (s, e, n) in reversed(spans):
        if n != n0:
            nb = nb[:s] + stub(n) + nb[e:]
    build('only_' + n0.replace('case ', ''), kv[:a] + nb + kv[b:], FLAGS)
