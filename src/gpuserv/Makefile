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
AGG_SPV    := $(foreach r,8 16 32,$(foreach a,4 8 12,build/gpuagg$(r)_$(a).spv))
SPV_FILES  := $(patsubst kernels/%.slang,build/%.spv,$(SLANG_SRCS)) build/vm_main.spv build/gpuscan8.spv build/gpuscan16.spv build/gpuscan32.spv $(AGG_SPV)
# per-kernel extra slangc flags (the memory model kernel must declare the Vulkan memory model)
SLANGFLAGS_memmodel := -capability vk_mem_model

LIB_OBJS := gpu_vk.o devcaps.o kernels_spv.o gpuserv_proto.o vm_selftest.o wire.o scan_prog.o scan_exec.o scan_caps.o

all: libgpuserv.a gpuserv-probe pgstrom-gpuserv

libgpuserv.a: $(LIB_OBJS)
	rm -f $@
	$(AR) rcs $@ $(LIB_OBJS)

gpuserv-probe: probe_main.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ probe_main.o libgpuserv.a $(VK_LIBS)

# the service program: --probe and --serve (see gpuserv_main.c)
pgstrom-gpuserv: gpuserv_main.o libgpuserv.a
	$(CC) $(CFLAGS) -o $@ gpuserv_main.o libgpuserv.a $(VK_LIBS)

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
# the GpuScan kernel (heap pages -> VM -> matches); its instruction numbers are shared with scan_prog.c
# one variant per register-file size: the register count dominates the kernel's speed
build/gpuscan%.spv: kernels_vm/scan.slang kernels_vm/scan_common.slang kernels_vm/scan_vm_ops.inc | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/scan.slang -I kernels_vm -DVM_MAX_REGS=$* $(SCAN_SLANGFLAGS) -target spirv -entry scan_main -stage compute -o $@

scan_caps.o: scan_caps.c scan_caps.h kernels_vm/scan_vm_ops.inc ../xpu_common.h ../xpu_opcodes.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -c -o $@ $<

# the aggregation kernel: variants by register count and by number of aggregates (the size of the
# per-thread accumulator arrays matters a lot); the stem is <regs>_<aggs>
build/gpuagg%.spv: kernels_vm/agg.slang kernels_vm/scan_common.slang kernels_vm/scan_vm_ops.inc | build
	@test -n "$(SLANGC)" || { echo "slangc not found: set SLANGC=/path/to/slangc"; exit 1; }
	$(SLANGC) kernels_vm/agg.slang -I kernels_vm -DVM_MAX_REGS=$(word 1,$(subst _, ,$*)) -DAGG_MAX=$(word 2,$(subst _, ,$*)) -DSCAN_WG=128 $(SCAN_SLANGFLAGS) -target spirv -entry agg_main -stage compute -o $@

scan_prog.o: scan_prog.c scan_prog.h kernels_vm/scan_vm_ops.inc ../xpu_common.h ../xpu_opcodes.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -c -o $@ $<

scan_exec.o: scan_exec.c scan_exec.h scan_prog.h scan_caps.h kernels_spv.h ../xpu_common.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -c -o $@ $<

vm_selftest.o: vm_selftest.c vm_selftest.h kernels_spv.h gen/opcodes.inc
	$(CC) $(CFLAGS) -I gen -I .. -c -o $@ $<

# the wire server reads the real backend protocol structs from ../xpu_common.h
wire.o: wire.c wire.h scan_exec.h scan_prog.h gpu_vk.h gpuserv_proto.h ../xpu_common.h ../xpu_opcodes.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -c -o $@ $<

# one generator run produces both files
kernels_spv.c: kernels_spv.h
kernels_spv.h: $(SPV_FILES) gen_kspv.sh
	sh ./gen_kspv.sh kernels_spv.c kernels_spv.h $(SPV_FILES)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

gpu_vk.o devcaps.o probe_main.o: gpu_vk.h
devcaps.o probe_main.o: gpuserv_probe.h gpuserv_proto.h
devcaps.o kernels_spv.o: kernels_spv.h
gpuserv_main.o: gpuserv_main.c gpuserv_probe.h gpuserv_proto.h vm_selftest.h gpu_vk.h wire.h scan_exec.h
	$(CC) $(CFLAGS) -Wno-format -DCUDA_MAXTHREADS_PER_BLOCK=512 -I .. -c -o $@ $<

check: gpuserv-probe proto_check
	./gpuserv-probe > build/probe.out
	./proto_check build/probe.out
	@cat build/probe.out

clean:
	rm -rf build gen *.o libgpuserv.a gpuserv-probe pgstrom-gpuserv proto_check kernels_spv.c kernels_spv.h

.PHONY: all check clean
