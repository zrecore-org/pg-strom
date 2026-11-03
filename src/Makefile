#
# PG-Strom Makefile
#
include ../Makefile.common

#
# Source of PG-Strom host code
#
STROM_OBJS = main.o githash.o extra.o codegen.o \
             executor.o fallback.o \
             gpu_device.o gpu_service.o gpu_cache.o \
             gpu_scan.o gpu_join.o gpu_preagg.o \
             relscan.o brin.o select_into.o \
             arrow_fdw.o arrow_meta.o \
             parquet_read.o parquet_cache.o \
             misc.o float2.o tinyint.o aggfuncs.o \
             pg_portable.o gpu_launcher.o gpuserv/gpuserv_proto.o
ifeq ($(WITH_POSTGIS),1)
STROM_OBJS += gist.o
endif
GENERATED-HEADERS = gpu_devattrs.h githash.c
STROM_HEADERS = arrow_defs.h float2.h

#
# Githash.c checker
#
GITHASH_CHECKS := $(shell grep -q \"$(PGSTROM_GITHASH)\" githash.c 2>/dev/null || rm -f githash.c)

#
# Source of NVIDIA GPU device code
#
include Makefile.cuda
ifdef PGSTROM_NO_CUDA
# "no GPU" build: the CUDA driver API resolves to the stub that reports no device
STROM_OBJS += nocuda/cuda_stub.o
endif

#
# Installation Scripts
#
__STROM_SQL = pg_strom--4.0--5.0.sql \
              pg_strom--5.0.sql \
              pg_strom--5.0--5.1.sql \
              pg_strom--5.1--6.0.sql \
              pg_strom--6.0--6.2.sql \
              pg_strom--6.2--7.0.sql
STROM_SQL = $(addprefix sql/,$(__STROM_SQL))

#
# Flags to build
#
PGSTROM_FLAGS += $(PGSTROM_FLAGS_CUSTOM)
PGSTROM_FLAGS += -D__PGSTROM_MODULE__=1
PGSTROM_FLAGS += -DPGSTROM_VERSION=\"$(PGSTROM_VERSION)\"

ifeq ($(PGSTROM_DEBUG),1)
PGSTROM_FLAGS += -g -O0 -DPGSTROM_DEBUG_BUILD=1
endif
# _Float16 on arm64. Older GCC on Linux/aarch64 needs -mfp16-format=ieee, GCC 13 and
# later rejects it (the type is native), clang (macOS) has no such option: add it only
# if the compiler accepts it.
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
PGSTROM_FLAGS += -DHAVE_FLOAT2
ifneq ($(shell echo 'int x;' | $(CC) -mfp16-format=ieee -x c -c -o /dev/null - 2>/dev/null && printf yes),)
PGSTROM_FLAGS += -mfp16-format=ieee
endif
endif
ifdef PGSTROM_NO_CUDA
PGSTROM_FLAGS += -DPGSTROM_NO_CUDA=1
endif
ifeq ($(WITH_POSTGIS),1)
PGSTROM_FLAGS += -DWITH_POSTGIS=1
endif
ifneq ($(PGSTROM_GITHASH),)
PGSTROM_FLAGS += -DPGSTROM_GITHASH=\"$(PGSTROM_GITHASH)\"
endif
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"   \
        -DPGINCLUDEDIR=\"$(shell $(PG_CONFIG) --includedir-server)\" \
        -DCUDA_MAXREGCOUNT=$(MAXREGCOUNT)                 \
        -DCUDA_MAXTHREADS_PER_BLOCK=$(MAXTHREADS_PER_BLOCK) \
        -DCUDA_CORE_FILES="\"$(__CUDA_CORE_FILES)\""      \
        -DCUDA_CORE_HEADERS="\"$(__CUDA_CORE_HEADERS)\""  \
        -DCUDA_TOOLKIT_BASEDIR="\"$(CUDA_PATH)\""
#
# Pre-built Fatbin (only debugging purpose)
#
ifeq ($(WITH_FATBIN),1)
DATA_built = $(CUDA_FATBIN)
PGSTROM_FLAGS += -DWITH_FATBIN=1
endif

#
# libarrow / libparquet
#
# (printf, not 'echo -n': macOS /bin/sh prints the "-n")
HAS_LIBARROW = $(shell $(PKGCONF) --exists arrow>/dev/null 2>&1   && printf yes)
HAS_PARQUET  = $(shell $(PKGCONF) --exists parquet>/dev/null 2>&1 && printf yes)
HAS_LIBNUMA  = $(shell $(PKGCONF) --exists numa>/dev/null 2>&1   && printf yes)
ifeq ($(HAS_LIBARROW),yes)
PGSTROM_FLAGS += -DHAS_LIBARROW=1
PG_CXXFLAGS += $(shell $(PKGCONF) --cflags arrow)
ifeq ($(HAS_PARQUET),yes)
PGSTROM_FLAGS += -DHAS_PARQUET=1
PG_CXXFLAGS += $(shell $(PKGCONF) --cflags parquet)
SHLIB_LINK += $(shell $(PKGCONF) --libs parquet)
PARQUET_LIBDIR = $(shell $(PKGCONF) --variable=libdir parquet)
ifneq ($(PARQUET_LIBDIR),)
SHLIB_LINK += -Wl,-rpath,$(PARQUET_LIBDIR)
endif
else
SHLIB_LINK += $(shell $(PKGCONF) --libs arrow)
ARROW_LIBDIR = $(shell $(PKGCONF) --variable=libdir arrow)
ifneq ($(ARROW_LIBDIR),)
SHLIB_LINK += -Wl,-rpath,$(ARROW_LIBDIR)
endif
endif
endif
ifeq ($(HAS_LIBNUMA),yes)
PGSTROM_FLAGS += -DHAS_LIBNUMA=1
PG_CXXFLAGS += $(shell $(PKGCONF) --cflags numa)
SHLIB_LINK += $(shell $(PKGCONF) --libs numa)
endif

#
# Definition of PG-Strom Extension
#
MODULE_big = pg_strom
MODULEDIR  = pg_strom
DATA = $(STROM_SQL) ../LICENSE Makefile.cuda \
       $(CUDA_SRCS) $(CUDA_HEADERS) $(STROM_HEADERS)
OBJS = $(STROM_OBJS)

EXTRA_CLEAN = $(CUDA_OBJS) $(GENERATED-HEADERS) \
              $(shell ls -d pgstrom-gpucode-V*-*.fatbin 2>/dev/null)
EXTENSION = pg_strom
PG_CPPFLAGS := $(PGSTROM_FLAGS) -I $(CUDA_IPATH)
# (+=, not :=: the pkgconf --cflags added above must survive; Homebrew's headers are
#  not in the default search path)
PG_CXXFLAGS += -std=c++20 -Wno-sign-compare
ifndef PGSTROM_NO_CUDA
SHLIB_LINK += -L $(CUDA_LPATH) -lcuda
endif
# the module links through the C driver; the C++ runtime (arrow_meta.cpp, parquet_read.cpp)
# has to be named explicitly on macOS
ifeq ($(shell uname -s),Darwin)
SHLIB_LINK += -lc++
endif

#
# The GPU service: a separate executable (pgstrom-gpuserv) with its own Makefile.
# WITH_GPUSERV=auto (default) builds it when slangc and the Vulkan headers are found and
# otherwise warns and carries on, so the module still builds and runs without a GPU;
# WITH_GPUSERV=1 makes a missing prerequisite an error, 0 skips it.
#
WITH_GPUSERV ?= auto
ifneq ($(WITH_GPUSERV),0)
all: gpuserv-build
install: gpuserv-install
uninstall: gpuserv-uninstall
clean: gpuserv-clean
endif

gpuserv-build:
	@$(MAKE) -C gpuserv pgstrom-gpuserv || { \
	  if [ "$(WITH_GPUSERV)" = 1 ]; then exit 1; fi; \
	  echo "WARNING: pgstrom-gpuserv was not built (needs slangc and the Vulkan headers); PG-Strom will run without GPU"; }

gpuserv-install: gpuserv-build
	@if [ -x gpuserv/pgstrom-gpuserv ]; then \
	  $(MKDIR_P) '$(DESTDIR)$(pkglibdir)' && \
	  $(INSTALL_PROGRAM) gpuserv/pgstrom-gpuserv '$(DESTDIR)$(pkglibdir)/pgstrom-gpuserv'; fi

gpuserv-uninstall:
	rm -f '$(DESTDIR)$(pkglibdir)/pgstrom-gpuserv'

gpuserv-clean:
	@$(MAKE) -C gpuserv clean

.PHONY: gpuserv-build gpuserv-install gpuserv-uninstall gpuserv-clean

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Our C++ code (and libarrow/libparquet) expects C++20 as baseline,
# however, some distribution has different defaults for clang.
# PostgreSQL's makefile has no particular label to give "-std=c++17"
# to clang to build JIT bytecode. So, we inject this option after
# the inclusion of buildchain.
#
BITCODE_CXXFLAGS += $(PG_CXXFLAGS)

#
# Device Attributes
#
submake-generated-headers: $(GENERATED-HEADERS)

ifdef PGSTROM_NO_CUDA
gpu_devattrs.h: nocuda/gpu_devattrs.h
	cp $< $@
else
gpu_devattrs.h: $(CUDA_IPATH)/cuda.h
	cat $(CUDA_IPATH)/cuda.h | \
	grep -E '^[ ]+CU_DEVICE_ATTRIBUTE_' | \
	grep -v -E 'CU_DEVICE_ATTRIBUTE_MAX$$' | \
	grep -v 'Deprecated[,\.]' | \
	sed -e 's|[ ]*CU_DEVICE_ATTRIBUTE_|DEV_ATTR(|g' \
	    -e 's| =.*/\*\*<[ ]*|, "|g' \
	    -e 's|[ ]*\*/|")|g' \
        -e 's| =.*$$|, NULL)|g' > $@
endif

githash.c:
	(echo "extern const char *pgstrom_githash_cstring;"; \
	 echo ""; \
	 echo "const char *pgstrom_githash_cstring = \"$(PGSTROM_GITHASH)\";") > githash.c

#
# GPU Device Code
#
ifndef PGSTROM_NO_CUDA
%.o: %.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_CFLAGS)  --device-c -o $@ $<

$(CUDA_FATBIN): $(CUDA_OBJS)
	$(NVCC) $(NVCC_LDFLAGS) --device-link --fatbin -o $@ $(CUDA_OBJS)

fatbin: $(CUDA_FATBIN)
endif
