# Throwaway PostgreSQL module: GPU scan of live shared_buffers. Not installed into the
# PostgreSQL tree; load it by absolute path (see ../live_test.sh).
MODULE_big = zcscan
OBJS = zcscan.o
PG_CPPFLAGS = -I/usr/local/include -I$(srcdir)/..
SHLIB_LINK = -L/usr/local/lib -lvulkan -Wl,-rpath,/usr/local/lib
PG_CONFIG ?= /opt/homebrew/opt/postgresql@17/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
