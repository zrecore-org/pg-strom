include ../Makefile.common

# 'contrib_regression_pg_strom' as test database
USE_MODULE_DB  := 1
MODULE_big     := pg_strom

CREATEDB_CMD   := $(shell $(PG_CONFIG) --bindir)/createdb
DROPDB_CMD     := $(shell $(PG_CONFIG) --bindir)/dropdb
PSQL           := $(shell $(PG_CONFIG) --bindir)/psql
PG2ARROW_CMD   := $(abspath ../arrow-tools/pg2arrow)
ARROW2CSV_CMD  := $(abspath ../arrow-tools/arrow2csv)
DBGEN_SSBM_CMD := $(abspath ./ssbm/dbgen-ssbm)
ARROW_TEST_DATA_DIR    := $(shell pwd)/data

REGRESS_DBNAME      := contrib_regression_$(MODULE_big)
REGRESS_REVISION    := 20260430
REGRESS_DB_REVISION := $(shell $(PSQL) $(REGRESS_DBNAME) -At -c 'SELECT pgstrom.regression_testdb_revision()')
REGRESS        := --schedule=./parallel_schedule
REGRESS_SETENV := 'PGDATABASE=$(REGRESS_DBNAME)'               \
                  'ARROW_TEST_DATA_DIR=$(ARROW_TEST_DATA_DIR)' \
                  'PG2ARROW_CMD=$(PG2ARROW_CMD)'               \
                  'ARROW2CSV_CMD=$(ARROW2CSV_CMD)'             \
                  'DBGEN_SSBM_CMD=$(DBGEN_SSBM_CMD)'           \
                  'REGRESS_DATASET_REVISION=$(REGRESS_REVISION)'
REGRESS_INPUTDIR  ?= ./$(PG_MAJORVERSION)
REGRESS_OUTPUTDIR ?= ./$(PG_MAJORVERSION)
REGRESS_OPTS   := --inputdir=$(REGRESS_INPUTDIR)               \
                  --outputdir=$(REGRESS_OUTPUTDIR)             \
                  --encoding=UTF-8                             \
                  --load-extension=pg_strom                    \
                  --launcher="env $(REGRESS_SETENV)"

ifeq ($(REGRESS_DB_REVISION),$(REGRESS_REVISION))
REGRESS_OPTS   += --use-existing
endif

REGRESS_PREP   := pg2arrow arrow2csv dbgen-ssbm parallel_schedule
EXTRA_CLEAN    := parallel_schedule

all: installcheck

# Vulkan build (branch vulkan-gpu): vulkan_schedule is the part of parallel_schedule that must pass,
# pending_schedule the tests that need a feature the Vulkan service does not have yet; see
# test/vulkan_known_failures.txt. test/vulkan_installcheck.sh runs these against a throwaway cluster.
#
#   vulkan-installcheck   vulkan_schedule (pgstrom_init sees an up-to-date dataset, see vulkan-initdb)
#   pending-installcheck  pending_schedule (expected to fail)
#   vulkan-oracle         vulkan_schedule once more with pg_strom.enabled forced off (results in
#                         <major>/oracle/results), the CPU oracle of vulkan_regen_expected.sh
VULKAN_PREP    := pg2arrow arrow2csv dbgen-ssbm
ORACLE_DIR     := $(abspath ./$(PG_MAJORVERSION)/oracle)

# Load the dataset outside of pg_regress: pgstrom_init then takes its "up to date" branch, whose output
# does not depend on how the database was created (the load branch prints the dbgen banner).
vulkan-initdb: $(VULKAN_PREP)
	@if [ "$(REGRESS_DB_REVISION)" != "$(REGRESS_REVISION)" ]; then \
	  echo "loading the regression dataset into $(REGRESS_DBNAME)"; \
	  $(DROPDB_CMD) --if-exists $(REGRESS_DBNAME) && \
	  $(CREATEDB_CMD) --template=template0 --encoding=UTF-8 $(REGRESS_DBNAME) && \
	  $(PSQL) -X -q -d $(REGRESS_DBNAME) -v ON_ERROR_STOP=1 \
	    -c "ALTER DATABASE $(REGRESS_DBNAME) SET lc_messages TO 'C'" \
	    -c "ALTER DATABASE $(REGRESS_DBNAME) SET lc_monetary TO 'C'" \
	    -c "ALTER DATABASE $(REGRESS_DBNAME) SET lc_numeric TO 'C'" \
	    -c "ALTER DATABASE $(REGRESS_DBNAME) SET lc_time TO 'C'" \
	    -c "ALTER DATABASE $(REGRESS_DBNAME) SET bytea_output TO 'hex'" \
	    -c "ALTER DATABASE $(REGRESS_DBNAME) SET timezone_abbreviations TO 'Default'" \
	    -c "CREATE EXTENSION pg_strom" && \
	  env $(REGRESS_SETENV) $(PSQL) -X -q -d $(REGRESS_DBNAME) -v ON_ERROR_STOP=1 \
	    -f sql/pgstrom_init.sql >/dev/null; \
	fi

vulkan-installcheck: vulkan-initdb
	$(MAKE) installcheck REGRESS='--schedule=./vulkan_schedule' REGRESS_PREP='$(VULKAN_PREP)'

pending-installcheck: vulkan-initdb
	$(MAKE) installcheck REGRESS='--schedule=./pending_schedule' REGRESS_PREP='$(VULKAN_PREP)'

vulkan-oracle: vulkan-initdb
	rm -rf $(ORACLE_DIR) && mkdir -p $(ORACLE_DIR)/sql $(ORACLE_DIR)/expected
	for f in sql/*.sql; do \
	  perl -pe 's/(pg_strom\.enabled\s*(=|TO)\s*)(on|true|1)\b/$${1}off/gi' $$f > $(ORACLE_DIR)/$$f; \
	  : > $(ORACLE_DIR)/expected/`basename $$f .sql`.out; \
	done
	PGOPTIONS='-c pg_strom.enabled=off' $(MAKE) installcheck REGRESS='--schedule=./vulkan_schedule' \
	  REGRESS_PREP='$(VULKAN_PREP)' REGRESS_INPUTDIR='$(ORACLE_DIR)' REGRESS_OUTPUTDIR='$(ORACLE_DIR)'; true

.PHONY: vulkan-initdb vulkan-installcheck pending-installcheck vulkan-oracle

pg2arrow:
	make -C ../arrow-tools PG_CONFIG=$(PG_CONFIG) pg2arrow

arrow2csv:
	make -C ../arrow-tools PG_CONFIG=$(PG_CONFIG) arrow2csv

dbgen-ssbm:
	make -C ssbm

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
