#
# PG-Strom Makefile
#
include ../Makefile.common

#
# Source of PG-Strom host code
#
STROM_OBJS = main.o githash.o codegen.o \
             executor.o fallback.o \
             gpu_device.o gpu_client.o gpu_cache.o \
             gpu_scan.o gpu_join.o gpu_preagg.o gist.o \
             relscan.o brin.o \
             arrow_fdw.o arrow_meta.o \
             parquet_read.o parquet_cache.o \
             misc.o float2.o tinyint.o aggfuncs.o \
             pg_portable.o gpu_launcher.o svc_proto.o svc_caps.o svc_prog.o svc_area.o
GENERATED-HEADERS = githash.c
STROM_HEADERS = arrow_defs.h float2.h

#
# Githash.c checker
#
GITHASH_CHECKS := $(shell grep -q \"$(PGSTROM_GITHASH)\" githash.c 2>/dev/null || rm -f githash.c)

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
ifneq ($(PGSTROM_GITHASH),)
PGSTROM_FLAGS += -DPGSTROM_GITHASH=\"$(PGSTROM_GITHASH)\"
endif
PGSTROM_FLAGS += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\"   \
        -DPGINCLUDEDIR=\"$(shell $(PG_CONFIG) --includedir-server)\" \
        -DCUDA_MAXTHREADS_PER_BLOCK=512

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
DATA = $(STROM_SQL) ../LICENSE $(STROM_HEADERS)
OBJS = $(STROM_OBJS)

EXTRA_CLEAN = $(GENERATED-HEADERS) $(OBJS:.o=.d)
EXTENSION = pg_strom
PG_CPPFLAGS := $(PGSTROM_FLAGS)
# Header dependencies: PGXS tracks them only if PostgreSQL was configured with --enable-depend, and an object
# left over from before a change of a struct (scan_prog.h) crashed the server with a stack overflow.
PG_CPPFLAGS += -MMD -MP
# (+=, not :=: the pkgconf --cflags added above must survive; Homebrew's headers are
#  not in the default search path)
PG_CXXFLAGS += -std=c++20 -Wno-sign-compare
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

# A9 part 3: the GPU driver compiles each kernel for the GPU at its first use (minutes for the large ones; the result cannot be
# shipped). "make prewarm" (as the user the server runs as: the pipeline cache is in that user's $HOME/.cache/pgstrom or in
# pg_strom.gpu_kernel_cache_dir) compiles the common set now, when the administrator expects to wait; PGSTROM_PREWARM=1 makes
# "make install" do it (not for DESTDIR builds, which have no GPU); PREWARM_SET=lite|geometry|all picks the set.
PREWARM_SET ?= lite
prewarm:
	'$(DESTDIR)$(pkglibdir)/pgstrom-gpuserv' --prewarm-all --prewarm-set $(PREWARM_SET)
ifeq ($(PGSTROM_PREWARM),1)
ifeq ($(DESTDIR),)
install: prewarm
prewarm: gpuserv-install
endif
endif

gpuserv-uninstall:
	rm -f '$(DESTDIR)$(pkglibdir)/pgstrom-gpuserv'

gpuserv-clean:
	@$(MAKE) -C gpuserv clean

.PHONY: gpuserv-build gpuserv-install gpuserv-uninstall gpuserv-clean prewarm

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Our C++ code (and libarrow/libparquet) expects C++20 as baseline,
# however, some distribution has different defaults for clang.
# PostgreSQL's makefile has no particular label to give "-std=c++17"
# to clang to build JIT bytecode. So, we inject this option after
# the inclusion of buildchain.
#
BITCODE_CXXFLAGS += $(PG_CXXFLAGS)

submake-generated-headers: $(GENERATED-HEADERS)

githash.c:
	(echo "extern const char *pgstrom_githash_cstring;"; \
	 echo ""; \
	 echo "const char *pgstrom_githash_cstring = \"$(PGSTROM_GITHASH)\";") > githash.c

# the .d files written by -MMD (after the PGXS include, so that they do not become the default goal)
-include $(OBJS:.o=.d)
