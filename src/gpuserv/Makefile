#
# Standalone Makefile for the PG-Strom GPU service Vulkan layer (no PGXS).
# Works with GNU make 3.81 (macOS) and GNU make 4.x.
#
#   make            build libgpuserv.a, gpuserv-probe and pgstrom-gpuserv
#   make check      run the probe and a proto round-trip sanity check
#   make clean
#
# SLANGC=/path/to/slangc, VK_CFLAGS=..., VK_LIBS=... can be overridden on the command line.
#
UNAME_S := $(shell uname -s)

ifeq ($(origin SLANGC),undefined)
SLANGC := $(firstword $(shell command -v slangc 2>/dev/null) \
                      $(wildcard $(HOME)/tools/slang-2026.20/bin/slangc /opt/slang/bin/slangc))
endif

ifeq ($(UNAME_S),Darwin)
VK_CFLAGS ?= -I/usr/local/include
VK_LIBS   ?= -L/usr/local/lib -lvulkan -Wl,-rpath,/usr/local/lib
else
VK_CFLAGS ?= $(shell pkg-config --cflags vulkan 2>/dev/null)
VK_LIBS   ?= $(shell pkg-config --libs vulkan 2>/dev/null || printf '%s' '-lvulkan')
endif

CC      ?= cc
CFLAGS  ?= -O2 -Wall -std=gnu11
CFLAGS  := $(CFLAGS) -I. $(VK_CFLAGS)
ifeq ($(findstring -std=,$(CFLAGS)),)
CFLAGS  += -std=gnu11
endif

SLANG_SRCS := $(sort $(wildcard kernels/*.slang))
AGG_SPV    := $(foreach r,8 16 32,$(foreach a,4 12,build/gpuagg$(r)_$(a).spv))
GAGG_SPV   := $(foreach r,16 32,$(foreach a,4 12,build/gagg$(r)_$(a).spv))
# GROUP BY over many groups: one global hash table per session (gtab.slang)
GTAB_SPV   := $(foreach r,16 32,build/gtab$(r).spv)
JOIN_SPV   := $(foreach r,8 16 32,build/gpujoin$(r).spv)
# aggregation over a join: <regs>_<aggregates>
JAGG_SPV   := $(foreach r,16 32,$(foreach a,4 12,build/jagg$(r)_$(a).spv))
JGAGG_SPV  := $(foreach r,16 32,$(foreach a,4 12,build/jgagg$(r)_$(a).spv))
SPV_FILES  := $(patsubst kernels/%.slang,build/%.spv,$(SLANG_SRCS)) build/vm_main.spv build/gpuscan8.spv build/gpuscan16.spv build/gpuscan32.spv $(AGG_SPV) $(GAGG_SPV) $(GTAB_SPV) $(JOIN_SPV) $(JAGG_SPV) $(JGAGG_SPV)
# per-kernel extra slangc flags (the memory model kernel must declare the Vulkan memory model)
SLANGFLAGS_memmodel := -capability vk_mem_model

LIB_OBJS := gpu_vk.o devcaps.o kernels_spv.o gpuserv_proto.o vm_selftest.o wire.o scan_prog.o scan_exec.o scan_caps.o cpu_vm.o
# Arrow_Fdw scans (A5)
ARROW_SPV  := $(foreach r,8 16 32,build/gpuarrow$(r).spv)
SPV_FILES  += $(ARROW_SPV)
# Arrow record batches as the outer relation of aggregations and joins (-DARROW_SRC; 16 and 32 registers)
AAGG_SPV   := $(foreach r,16 32,$(foreach a,4 12,build/aagg$(r)_$(a).spv))
AGAGG_SPV  := $(foreach r,16 32,$(foreach a,4 12,build/agagg$(r)_$(a).spv))
AJOIN_SPV  := $(foreach r,16 32,build/ajoin$(r).spv)
AJAGG_SPV  := $(foreach r,16 32,$(foreach a,4 12,build/ajagg$(r)_$(a).spv))
AGTAB_SPV  := $(foreach r,16 32,build/agtab$(r).spv)
AJGAGG_SPV := $(foreach r,16 32,$(foreach a,4 12,build/ajgagg$(r)_$(a).spv))
SPV_FILES  += $(AAGG_SPV) $(AGAGG_SPV) $(AGTAB_SPV) $(AJOIN_SPV) $(AJAGG_SPV) $(AJGAGG_SPV)
LIB_OBJS   += arrow_exec.o
# GPU-Cache (A7 part 2)
LIB_OBJS   += gcache.o
# GPU-Sort (A7): the bitonic sort kernel and the host side
# Every VM kernel depends on every kernel source: the per-rule lists missed includes of includes (f64_soft.inc and
# area_ring.inc are included by scan_common.slang and were in no rule), so an edit of one of them left stale SPIR-V in
# the service while the planner and the program builder already spoke the new instruction set (wrong or empty results
# right after a merge that changed such a file). Identical output costs nothing: the driver's caches key on the SPIR-V.
KDEPS := $(wildcard kernels_vm/*.slang kernels_vm/*.inc)
# the lite and nogeom variants of every VM kernel (see LITE_FLAGS / NOGEOM_FLAGS)
SPV_VMBASE := build/gpuscan8.spv build/gpuscan16.spv build/gpuscan32.spv $(AGG_SPV) $(GAGG_SPV) $(GTAB_SPV) $(JOIN_SPV) $(JAGG_SPV) $(JGAGG_SPV) $(ARROW_SPV) $(AAGG_SPV) $(AGAGG_SPV) $(AGTAB_SPV) $(AJOIN_SPV) $(AJAGG_SPV) $(AJGAGG_SPV)
SPV_LITE   := $(patsubst build/%.spv,build/%_lite.spv,$(SPV_VMBASE))
SPV_NOGEOM := $(patsubst build/%.spv,build/%_nogeom.spv,$(SPV_VMBASE))
SPV_FILES  += $(SPV_LITE) $(SPV_NOGEOM)
$(SPV_VMBASE) $(SPV_LITE) $(SPV_NOGEOM): $(KDEPS)
SPV_FILES  += build/gsort.spv
LIB_OBJS   += gpusort.o gpu_vk_seq.o
# A6: specialized kernels generated from the programs and compiled at run time (specialize.c, spec_mgr.c); the
# kernel sources are embedded for the compiler
KSRC_FILES := $(sort $(wildcard kernels_vm/*.slang kernels_vm/*.inc))
LIB_OBJS   += specialize.o spec_mgr.o spec_verify.o kernels_src.o
SLANGC_ABS := $(if $(SLANGC),$(realpath $(SLANGC)))

all: libgpuserv.a gpuserv-probe pgstrom-gpuserv

libgpuserv.a: $(LIB_OBJS)
	rm -f $@
	$(AR) rcs $@ $(LIB_OBJS)

gpuserv-probe: probe_main.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ probe_main.o libgpuserv.a $(VK_LIBS) -lpthread

# the service program: --probe and --serve (see gpuserv_main.c)
pgstrom-gpuserv: gpuserv_main.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ gpuserv_main.o libgpuserv.a $(VK_LIBS) -lpthread

proto_check: proto_check.o gpuserv_proto.o
	$(CC) $(CFLAGS) -o $@ proto_check.o gpuserv_proto.o

build:
	mkdir -p build

build/%.spv: kernels/%.slang | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) $< -target spirv -entry main -stage compute $(SLANGFLAGS_$*) -o $@

# The opcode-switch VM: TypeOpCode/FuncOpCode come from the real xpu_common.h/xpu_opcodes.h
# (the enums are extracted verbatim, so host and kernel cannot disagree on the numbering).
gen/opcodes.inc gen/opcodes.slang: gen_opcodes.sh ../xpu_common.h ../xpu_opcodes.h
	sh ./gen_opcodes.sh ../xpu_common.h gen

build/vm_main.spv: kernels_vm/vm.slang kernels_vm/common.slang kernels_vm/common_out.slang gen/opcodes.slang | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/vm.slang -I gen -I kernels_vm -I .. -target spirv -entry vm_main -stage compute -o $@

SCAN_SLANGFLAGS ?=
# Every kernel exists in three tiers (scan_caps.h: SCAN_TIER_*, picked by scan_prog.feat): the full one, <name>_lite without the
# optional instruction groups (numeric, jsonb, float8, geometry; kernels_vm/scan_common.slang, -DVM_NO_*) for a program that uses
# none of them (feat == 0), and <name>_nogeom (-DVM_NO_GEOM only) for a program that uses numeric, jsonb or float8 but no
# geometry. The lite ones are a fourth to a tenth of the size and Metal needs minutes to compile the full ones (the join kernel is
# 9.6 MB of SPIR-V, 2.98 MB without geometry, whose soft-float code is most of it), once after every change of the sources.
# The stem of the rules carries the suffix: $(LITE_OF) adds the flags.
LITE_FLAGS := -DVM_NO_NUM -DVM_NO_JSON -DVM_NO_F8 -DVM_NO_GEOM
NOGEOM_FLAGS := -DVM_NO_GEOM
LITE_OF = $(if $(findstring _lite,$*),$(LITE_FLAGS),$(if $(findstring _nogeom,$*),$(NOGEOM_FLAGS)))
# the GpuScan kernel (heap pages -> VM -> matches); its instruction numbers are shared with scan_prog.c
# one variant per register-file size: the register count dominates the kernel's speed
build/gpuscan%.spv: kernels_vm/scan.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/scan.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry scan_main -stage compute -o $@

scan_caps.o: scan_caps.c scan_caps.h kernels_vm/scan_vm_ops.inc ../xpu_common.h ../xpu_opcodes.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

# the aggregation kernel: variants by register count and by number of aggregates (the size of the
# per-thread accumulator arrays matters a lot); the stem is <regs>_<aggs>
build/gpuagg%.spv: kernels_vm/agg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/agg.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DAGG_MAX=$(word 2,$(subst _, ,$*)) -DSCAN_WG=128 $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry agg_main -stage compute -o $@

cpu_vm.o: cpu_vm.c cpu_vm.h scan_prog.h scan_caps.h kernels_vm/scan_vm_ops.inc ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

# the GROUP BY kernel: <regs>_<aggregates>; the capacity of the per-workgroup table shrinks as the
# accumulators grow, to fit shared memory (the host has the same table in scan_exec.c)
build/gagg%.spv: kernels_vm/gagg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/gagg.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DGA_NA=$(word 2,$(subst _, ,$*)) -DGA_C=$(if $(filter 4,$(word 2,$(subst _, ,$*))),128,64) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry gagg_main -stage compute -o $@

# the GROUP BY kernel with one global hash table per session (many groups): one variant per register count
build/gtab%.spv: kernels_vm/gtab.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/gtab.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry gagg_main -stage compute -o $@

# the join kernel: one variant per register count
JOIN_INCS := kernels_vm/join_common.inc kernels_vm/join_walk.inc
build/gpujoin%.spv: kernels_vm/join.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(JOIN_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/join.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry join_main -stage compute -o $@

# aggregation over a join, without GROUP BY: <regs>_<aggregates>
build/jagg%.spv: kernels_vm/jagg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(JOIN_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/jagg.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DAGG_MAX=$(word 2,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry jagg_main -stage compute -o $@

# GROUP BY over a join: <regs>_<aggregates>; the table capacity as for gagg
build/jgagg%.spv: kernels_vm/jgagg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(JOIN_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/jgagg.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DGA_NA=$(word 2,$(subst _, ,$*)) -DGA_C=$(if $(filter 4,$(word 2,$(subst _, ,$*))),128,64) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry jgagg_main -stage compute -o $@

# the Arrow scan kernel: one variant per register count
build/gpuarrow%.spv: kernels_vm/arrow.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc kernels_vm/arrow_load.inc | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/arrow.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry arrow_main -stage compute -o $@

build/gsort.spv: kernels_vm/gsort.slang | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/gsort.slang -I kernels_vm -target spirv -entry gsort_main -stage compute -o $@

gpusort.o: gpusort.c gpusort.h scan_exec.h kernels_spv.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

# offline check of the kernel generator: spec_check <dir> writes the sources of every kernel kind
# (test/gpuserv/spec_compile_check.sh compiles them with slangc)
spec_check.o: spec_check.c specialize.h scan_prog.h scan_caps.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

spec_check: spec_check.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ spec_check.o libgpuserv.a $(VK_LIBS) -lpthread

# offline check of the kernel tier selection (every GEOM* opcode sets SCAN_FEAT_GEOM; no kernels needed)
feat_check.o: feat_check.c scan_caps.h kernels_vm/scan_vm_ops.inc ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

feat_check: feat_check.o scan_caps.o
	$(CC) $(CFLAGS) -o $@ feat_check.o scan_caps.o

# self-check of GPU-Sort against the CPU sort and naive window-rank (not part of "all")
gpusort_check.o: gpusort_check.c gpusort.h scan_exec.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

gpusort_check: gpusort_check.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ gpusort_check.o libgpuserv.a $(VK_LIBS) -lpthread

# time to create the VM pipelines with and without the persisted pipeline cache (not part of "all"; test/gpuserv/pcache_test.sh)
pcache_check.o: pcache_check.c gpu_vk.h kernels_spv.h
	$(CC) $(CFLAGS) -c -o $@ $<

pcache_check: pcache_check.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ pcache_check.o libgpuserv.a $(VK_LIBS) -lpthread

# self-check of the software binary64 (kernels_vm/f64_soft.inc): the library as C against the host's hardware double,
# and the same source compiled by slangc against the device (not part of "all"; test/gpuserv/f64soft_check.sh)
build/f64soft_check.spv: kernels_check/f64soft_check.slang kernels_vm/f64_soft.inc | build
	$(SLANGC) kernels_check/f64soft_check.slang -I kernels_vm -target spirv -entry main -stage compute -o $@

f64soft_check.o: f64soft_check.c kernels_vm/f64_soft.inc gpu_vk.h
	$(CC) $(CFLAGS) -ffp-contract=off -fno-fast-math -c -o $@ $<

f64soft_check: f64soft_check.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ f64soft_check.o libgpuserv.a $(VK_LIBS) -lpthread -lm

# self-check of the string hash of the join kernels (kernels_vm/hash_bytes.inc) against PostgreSQL's hashtext() (test/gpuserv/hash_check.sh)
build/hash_check.spv: kernels_check/hash_check.slang kernels_vm/hash_bytes.inc | build
	$(SLANGC) kernels_check/hash_check.slang -I kernels_vm -target spirv -entry main -stage compute -o $@

hash_check.o: hash_check.c gpu_vk.h
	$(CC) $(CFLAGS) -c -o $@ $<

hash_check: hash_check.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ hash_check.o libgpuserv.a $(VK_LIBS) -lpthread

ARROW_INCS := kernels_vm/arrow_load.inc
build/aagg%.spv: kernels_vm/agg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(ARROW_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/agg.slang -I kernels_vm -DARROW_SRC=1 -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DAGG_MAX=$(word 2,$(subst _, ,$*)) -DSCAN_WG=128 $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry agg_main -stage compute -o $@

build/agagg%.spv: kernels_vm/gagg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(ARROW_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/gagg.slang -I kernels_vm -DARROW_SRC=1 -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DGA_NA=$(word 2,$(subst _, ,$*)) -DGA_C=$(if $(filter 4,$(word 2,$(subst _, ,$*))),128,64) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry gagg_main -stage compute -o $@

# GROUP BY over many groups on an Arrow record batch (and the GPU-Cache): gtab.slang with -DARROW_SRC
build/agtab%.spv: kernels_vm/gtab.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(ARROW_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/gtab.slang -I kernels_vm -DARROW_SRC=1 -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry gagg_main -stage compute -o $@

build/ajoin%.spv: kernels_vm/join.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(JOIN_INCS) $(ARROW_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/join.slang -I kernels_vm -DARROW_SRC=1 -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry join_main -stage compute -o $@

build/ajagg%.spv: kernels_vm/jagg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(JOIN_INCS) $(ARROW_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/jagg.slang -I kernels_vm -DARROW_SRC=1 -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DAGG_MAX=$(word 2,$(subst _, ,$*)) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry jagg_main -stage compute -o $@

build/ajgagg%.spv: kernels_vm/jgagg.slang kernels_vm/scan_common.slang kernels_vm/vagg.inc kernels_vm/hash_bytes.inc kernels_vm/scan_vm_ops.inc $(JOIN_INCS) $(ARROW_INCS) | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/jgagg.slang -I kernels_vm -DARROW_SRC=1 -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DGA_NA=$(word 2,$(subst _, ,$*)) -DGA_C=$(if $(filter 4,$(word 2,$(subst _, ,$*))),128,64) $(LITE_OF) $(SCAN_SLANGFLAGS) -target spirv -entry jgagg_main -stage compute -o $@

arrow_exec.o: arrow_exec.c scan_exec.h scan_prog.h scan_caps.h kernels_spv.h gcache.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

gcache.o: gcache.c gcache.h gcache_proto.h scan_prog.h scan_caps.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

# the kernel sources, embedded (one generator run produces both files)
kernels_src.c: kernels_src.h
kernels_src.h: $(KSRC_FILES) gen_ksrc.sh
	sh ./gen_ksrc.sh kernels_src.c kernels_src.h $(KSRC_FILES)

specialize.o: specialize.c specialize.h scan_prog.h scan_caps.h kernels_vm/scan_vm_ops.inc ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

spec_verify.o: spec_verify.c specialize.h scan_exec.h scan_prog.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

spec_mgr.o: spec_mgr.c specialize.h scan_exec.h scan_prog.h scan_caps.h kernels_src.h gpu_vk.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 $(if $(SLANGC_ABS),-DSPEC_SLANGC_PATH='"$(SLANGC_ABS)"') -I .. -MMD -MP -c -o $@ $<

scan_prog.o: scan_prog.c scan_prog.h kernels_vm/scan_vm_ops.inc ../xpu_common.h ../xpu_opcodes.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

scan_exec.o: scan_exec.c scan_exec.h scan_prog.h scan_caps.h kernels_spv.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

vm_selftest.o: vm_selftest.c vm_selftest.h kernels_spv.h gen/opcodes.inc
	$(CC) $(CFLAGS) -I gen -I .. -c -o $@ $<

# the wire server reads the real backend protocol structs from ../xpu_common.h
wire.o: wire.c wire.h scan_exec.h scan_prog.h gpu_vk.h gpuserv_proto.h gcache.h ../xpu_common.h ../xpu_opcodes.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

# one generator run produces both files
kernels_spv.c: kernels_spv.h
kernels_spv.h: $(SPV_FILES) gen_kspv.sh
	sh ./gen_kspv.sh kernels_spv.c kernels_spv.h $(SPV_FILES)

# -MMD writes <name>.d next to each object: a changed header (scan_exec.h, gpu_vk.h, ...) rebuilds what includes it (without it,
# objects with the old struct layout stayed and the service crashed or failed its self-test)
%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<
-include $(wildcard *.d)

gpu_vk.o devcaps.o probe_main.o: gpu_vk.h
devcaps.o probe_main.o: gpuserv_probe.h gpuserv_proto.h
devcaps.o kernels_spv.o: kernels_spv.h
gpuserv_main.o: gpuserv_main.c gpuserv_probe.h gpuserv_proto.h vm_selftest.h gpu_vk.h wire.h scan_exec.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

# self-check of the GPU-Cache store without a GPU (not part of "all")
gcache_check.o: gcache_check.c gcache.h gcache_proto.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -MMD -MP -c -o $@ $<

gcache_check: gcache_check.o gcache.o scan_caps.o
	$(CC) $(CFLAGS) -o $@ gcache_check.o gcache.o scan_caps.o

check: gpuserv-probe proto_check
	./gpuserv-probe > build/probe.out
	./proto_check build/probe.out
	@cat build/probe.out

clean:
	rm -rf build gen *.o libgpuserv.a gpuserv-probe pgstrom-gpuserv proto_check gcache_check spec_check feat_check f64soft_check pcache_check kernels_spv.c kernels_spv.h kernels_src.c kernels_src.h

.PHONY: all check clean
