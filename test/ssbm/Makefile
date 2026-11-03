DBGEN = dbgen-ssbm
# (dists.dss.h is a prerequisite only: clang refuses a header on the compile line together with -o)
DBGEN_SOURCE = bcd2.c  build.c load_stub.c print.c text.c \
	bm_utils.c driver.c permute.c rnd.c speed_seed.c dists.dss.h
DBGEN_CFLAGS = -DDBNAME=\"dss\" -DLINUX -DDB2 -DSSBM -DTANDEM \
               -DSTATIC_DISTS=1 \
               -O2 -g -I. -I$(STROM_BUILD_ROOT)/utils/ssbm
PREFIX	?= /usr/local
BINDIR	?= $(PREFIX)/bin

all: dbgen-ssbm

$(DBGEN): $(DBGEN_SOURCE)
	$(CC) $(DBGEN_CFLAGS) $(filter %.c,$(DBGEN_SOURCE)) -o $(DBGEN) -lm

install: $(DBGEN)
	mkdir -p $(DESTDIR)$(BINDIR)
	install -m 0755 $(DBGEN) $(DESTDIR)$(BINDIR)

clean:
	rm -f $(DBGEN)
