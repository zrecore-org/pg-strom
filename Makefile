# ----------------------------------------
#
# Makefile for packaging
#
# ----------------------------------------
include Makefile.common

__PGSTROM_TGZ := pg_strom-$(PGSTROM_VERSION)
PGSTROM_TAR   := $(__PGSTROM_TGZ).tar
PGSTROM_TGZ   := $(__PGSTROM_TGZ).tar.gz
PGSTROM_TGZ_FULLPATH := $(shell realpath $(PGSTROM_TGZ))

SWDC ?= $(shell test -d ../swdc/.git && realpath ../swdc || echo /dev/null)
__SWDC_URL = git@github.com:heterodb/swdc.git

__ARCH = $(shell rpmbuild -E %{_arch})
__DIST = $(shell rpmbuild -E %{dist} | grep -o -E '\.el[0-9]+' | sed 's/\.el/rhel/g')
__SOURCEDIR = $(shell rpmbuild -E %{_sourcedir})
__SPECDIR = $(shell rpmbuild -E %{_specdir})
__RPMDIR = $(shell rpmbuild -E %{_rpmdir})/$(__ARCH)
__SRPMDIR = $(shell rpmbuild -E %{_srcrpmdir})
__SPECFILE = $(__SPECDIR)/pg_strom.spec
__RPMFILES = $(shell rpmspec --rpms -q $(__SPECFILE) | grep -vE 'debug(info|source)')
__RPMDEBUGS = $(shell rpmspec --rpms -q $(__SPECFILE) | grep -E 'debug(info|source)')
__SRPMFILE = $(shell rpmspec --srpm -q $(__SPECFILE) | sed s/\.$(__ARCH)/.src/g)
__SWDC_RPMS_DIR=docs/yum/$(__DIST)-$(__ARCH)
__SWDC_DEBUG_DIR=docs/yum/$(__DIST)-debuginfo
__SWDC_SRPM_DIR=docs/yum/$(__DIST)-source

rpm: rpm-pg_strom

all: rpm-pg_strom rpm-mysql2arrow

__precheck_swdc:
	(cd $(SWDC);	\
	 test "`git remote get-url origin`" = "$(__SWDC_URL)" || exit 1; \
	 git pull || exit 1)
	pinentry -h >&/dev/null || (echo "dnf install pinentry"; exit 1)

swdc: __precheck_swdc rpm-pg_strom
	(cd $(SWDC);						\
	mkdir -p $(__SWDC_RPMS_DIR);				\
	for __file in $(addsuffix .rpm,$(__RPMFILES)); do	\
	  rpmsign --addsign $(__RPMDIR)/$${__file} &&		\
	  install -m 644 $(__RPMDIR)/$${__file} $(__SWDC_RPMS_DIR) && \
	  git add $(__SWDC_RPMS_DIR)/$${__file};			\
	done;							\
	mkdir -p $(__SWDC_DEBUG_DIR);				\
	for __file in $(addsuffix .rpm,$(__RPMDEBUGS)); do	\
	  rpmsign --addsign $(__RPMDIR)/$${__file} &&		\
	  install -m 644 $(__RPMDIR)/$${__file} $(__SWDC_DEBUG_DIR) && \
	  git add $(__SWDC_DEBUG_DIR)/$${__file};	\
	done;							\
	mkdir -p $(__SWDC_SRPM_DIR);				\
	for __file in $(addsuffix .rpm,$(__SRPMFILE)); do	\
	  rpmsign --addsign $(__SRPMDIR)/$${__file} &&		\
	  install -m 644 $(__SRPMDIR)/$${__file} $(__SWDC_SRPM_DIR) && \
	  git add $(__SWDC_SRPM_DIR)/$${__file};		\
	done;							\
	install -m 644 $(PGSTROM_TGZ_FULLPATH) docs/tgz &&	\
	git add docs/tgz/$(PGSTROM_TGZ);			\
	./update-index.sh)

tarball:
	git archive --format=tar			\
	            --prefix=$(__PGSTROM_TGZ)/		\
	            -o $(PGSTROM_TAR) $(GITHASH)	\
	            LICENSE LICENSES README.md Makefile	\
	            doc src arrow-tools test/ssbm files/SOURCE-OFFER.txt
	TARFILE=`realpath $(PGSTROM_TAR)` && 		\
	TEMP=`mktemp -d` &&				\
	mkdir -p $${TEMP}/$(__PGSTROM_TGZ) &&		\
	git show --format=raw $(GITHASH):Makefile.common \
	| awk '/^GITHASH_IF_NOT_GIVEN/{ print "GITHASH_IF_NOT_GIVEN := $(GITHASH)"; next } { print }' \
	> $${TEMP}/$(__PGSTROM_TGZ)/Makefile.common &&	\
	pushd $${TEMP} &&				\
	echo $${TARFILE} &&				\
	tar rf $${TARFILE} $(__PGSTROM_TGZ)/Makefile.common && \
	popd &&						\
	gzip -f $(PGSTROM_TAR)

rpm-pg_strom: tarball
	cp -f $(PGSTROM_TGZ) $(__SOURCEDIR) || exit 1
	git show --format=raw $(GITHASH):files/systemd-pg_strom.conf > \
		$(__SOURCEDIR)/systemd-pg_strom.conf || exit 1
	git show --format=raw $(GITHASH):files/pg_strom.spec.in |	\
		sed -e "s/@@STROM_VERSION@@/$(VERSION)/g"		\
		    -e "s/@@STROM_RELEASE@@/$(RELEASE)/g"		\
		    -e "s/@@STROM_TARBALL@@/$(__PGSTROM_TGZ)/g"		\
		    -e "s/@@PGSTROM_GITHASH@@/$(GITHASH)/g" > $(__SPECFILE)
	# rpmbuild in RHEL10 checks 'rpath', so suggest to ignore the rpath that
	# src/Makefile adds for libarrow / libparquet (QA_RPATHS=7).
	# slangc is not an RPM: give its path as 'SLANGC=/path/to/slangc make rpm'
	# if it is neither in PATH nor in a place src/gpuserv/Makefile looks at.
	env QA_RPATHS=7 rpmbuild -ba $(__SPECFILE)

# Debian package of the module built for the PostgreSQL given by PG_CONFIG:
#   make deb PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config
# Needs slangc (SLANGC=/path/to/slangc, PATH, or see src/gpuserv/Makefile), the Vulkan
# headers (libvulkan-dev), libarrow-dev and libparquet-dev. Not built or tested on a
# Debian machine by the Vulkan port yet.
DEB_ARCH     = $(shell dpkg --print-architecture)
ARROW_MAJOR  = $(shell $(PKGCONF) --modversion arrow   | cut -d. -f1)
PARQUET_MAJOR= $(shell $(PKGCONF) --modversion parquet | cut -d. -f1)

deb:
	@which fakeroot >&/dev/null || (echo "Run, sudo dnf install fakeroot"; exit 1)
	@which dpkg-deb >&/dev/null || (echo "Run, sudo dnf install dpkg"; exit 1)
	WORK=`mktemp -d` && mkdir -p $${WORK}/DEBIAN &&		\
	(cat files/pg-strom.control.in |			\
	 sed -e 's/@@PG_MAJORVERSION@@/$(PG_MAJORVERSION)/g'	\
	     -e 's/@@PG_MINORVERSION@@/$(PG_MINORVERSION)/g'	\
	     -e 's/@@PGSTROM_VERSION@@/$(PGSTROM_VERSION)/g'	\
	     -e 's/@@DEB_ARCH@@/$(DEB_ARCH)/g'			\
	     -e 's/@@ARROW_MAJOR@@/$(ARROW_MAJOR)/g'		\
	     -e 's/@@PARQUET_MAJOR@@/$(PARQUET_MAJOR)/g'	\
	) > $${WORK}/DEBIAN/control &&				\
	make -C src PG_CONFIG=$(PG_CONFIG) DESTDIR=$${WORK} WITH_GPUSERV=1 clean && \
	make -C src PG_CONFIG=$(PG_CONFIG) DESTDIR=$${WORK} WITH_GPUSERV=1 -j 8  && \
	make -C src PG_CONFIG=$(PG_CONFIG) DESTDIR=$${WORK} WITH_GPUSERV=1 install && \
	install -Dm 644 files/pg-strom.copyright $${WORK}/usr/share/doc/pg-strom-pg$(PG_MAJORVERSION)/copyright && \
	install -Dm 644 files/SOURCE-OFFER.txt $${WORK}/usr/share/doc/pg-strom-pg$(PG_MAJORVERSION)/SOURCE-OFFER.txt && \
	install -Dm 644 LICENSES/GPL-2.0.txt $${WORK}/usr/share/doc/pg-strom-pg$(PG_MAJORVERSION)/GPL-2.0.txt && \
	fakeroot dpkg-deb --build $${WORK} .
