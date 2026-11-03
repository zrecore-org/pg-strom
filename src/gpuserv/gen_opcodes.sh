#!/bin/sh
# Extract the TypeOpCode / FuncOpCode enum definitions from xpu_common.h
# verbatim, so the numbering seen by the C test harness and by the Slang VM is
# the one the real code base uses (codegen.c emits these values).
#
# usage: gen_opcodes.sh <path/to/xpu_common.h> <outdir>
set -e
src=$1
out=$2
mkdir -p "$out"

# C view: the block as-is (it #includes xpu_opcodes.h itself)
awk '/^#define TYPE_OPCODE\(NAME,a,b\)/ { p = 1 }
     p { print }
     /^} FuncOpCode;/ { exit }' "$src" > "$out/opcodes.inc"

# Slang view: same block, typedef-enums rewritten as unscoped Slang enums
awk '/^typedef enum \{/ { n++
                          print "[UnscopedEnum] enum " (n == 1 ? "TypeOpCode" : "FuncOpCode") " : uint {"
                          next }
     /^} (TypeOpCode|FuncOpCode);/ { print "}"; next }
     { print }' "$out/opcodes.inc" > "$out/opcodes.slang"
