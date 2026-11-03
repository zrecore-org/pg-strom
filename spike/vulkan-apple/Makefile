# Throwaway spike, not part of the PG-Strom build.
# Needs the LunarG Vulkan SDK (MoltenVK + glslangValidator) installed under /usr/local.
GLSLANG ?= glslangValidator
# Slang: official release tarball, or `brew install schrodinger/shader-slang/shader-slang`
SLANGC  ?= $(shell which slangc 2>/dev/null || echo $(HOME)/tools/slang-2026.20/bin/slangc)
SRC     ?= ../../src
HEAP_FLAGS ?=
SHADERS  = shaders/scan.spv shaders/sum64.spv shaders/bda.spv shaders/probe_i64_subgroup.spv shaders/scan_slang.spv

all: vkspike $(SHADERS) vmspike shaders/vm_main.spv shaders/fixed_main.spv

shaders/%.spv: shaders/%.comp
	$(GLSLANG) -V --target-env vulkan1.2 $< -o $@

shaders/%_slang.spv: shaders/%.slang
	$(SLANGC) $< -target spirv -entry main -stage compute -o $@

vkspike: vkspike.c
	$(CC) -O2 -std=gnu11 -I/usr/local/include -L/usr/local/lib -lvulkan \
	      -Wl,-rpath,/usr/local/lib -o $@ $<

run: all
	./vkspike shaders scan.spv
	./vkspike shaders scan_slang.spv

clean:
	rm -f vkspike vmspike heapspike shaders/*.spv; rm -rf gen

# ---- opcode-switch VM prototype ----
# enums extracted verbatim from the real xpu_common.h; xpu_opcodes.h used unchanged
gen/opcodes.inc gen/opcodes.slang: gen_opcodes.sh $(SRC)/xpu_common.h $(SRC)/xpu_opcodes.h
	sh gen_opcodes.sh $(SRC)/xpu_common.h gen

shaders/vm_main.spv shaders/fixed_main.spv: shaders/%.spv: shaders/vm.slang shaders/common.slang gen/opcodes.slang
	$(SLANGC) shaders/vm.slang -I gen -I shaders -I $(SRC) -target spirv -entry $* -stage compute -o $@

vmspike: vmspike.c vkutil.h gen/opcodes.inc
	$(CC) -O2 -std=gnu11 -I gen -I shaders -I $(SRC) -I/usr/local/include -L/usr/local/lib -lvulkan \
	      -Wl,-rpath,/usr/local/lib -o $@ vmspike.c

run-vm: vmspike shaders/vm_main.spv shaders/fixed_main.spv
	SLANGC=$(SLANGC) ./vmspike shaders

# ---- real PostgreSQL heap pages (needs the throwaway cluster: ./pg_setup.sh) ----
shaders/heap_fixed.spv: shaders/heap_fixed.slang shaders/heap_common.slang shaders/common_out.slang
	$(SLANGC) shaders/heap_fixed.slang -I shaders $(HEAP_FLAGS) -target spirv -entry heap_fixed -stage compute -o $@

shaders/heap_vm.spv: shaders/vm.slang shaders/heap_common.slang shaders/common_out.slang gen/opcodes.slang
	$(SLANGC) shaders/vm.slang -I gen -I shaders -I $(SRC) -DVM_HEAP $(HEAP_FLAGS) -target spirv -entry heap_vm -stage compute -o $@

heapspike: heapspike.c vkutil.h exprlib.h gen/opcodes.inc
	$(CC) -O2 -std=gnu11 -I gen -I $(SRC) -I/usr/local/include -L/usr/local/lib -lvulkan \
	      -Wl,-rpath,/usr/local/lib -o $@ heapspike.c

# oracle = SQL on the very same table: match count + sum of blkno*512+offnum
ORACLE_SUM = sum(((ctid::text::point)[0])::bigint * 512 + ((ctid::text::point)[1])::bigint)
E1_WHERE   = b + a::int8 > 1073741824 AND f > 500::float4 AND d < 25 AND a > 1000
E2_WHERE   = (a > 1000000000 OR b IS NULL) AND NOT (f < 100::float4 OR d IS NULL)

run-heap: heapspike shaders/heap_fixed.spv shaders/heap_vm.spv
	@REL=/tmp/pgscan/data/$$(./pgq.sh -c "SELECT pg_relation_filepath('t')"); \
	E1=$$(./pgq.sh -F ' ' -c "SELECT count(*), $(ORACLE_SUM) FROM t WHERE $(E1_WHERE)"); \
	E2=$$(./pgq.sh -F ' ' -c "SELECT count(*), $(ORACLE_SUM) FROM t WHERE $(E2_WHERE)"); \
	echo "oracle E1: $$E1   E2: $$E2"; \
	SLANGC=$(SLANGC) ./heapspike shaders $$REL $$E1 $$E2
