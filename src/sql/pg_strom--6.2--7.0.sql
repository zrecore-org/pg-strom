---
--- PG-Strom v6.2 -> v7.0 (Vulkan GPU service)
---

---
--- Real capabilities of each Vulkan device, as measured by the probe kernels
--- (caps) against what the API claimed (reported_caps)
---
---
--- The license query belonged to the closed-source heterodb-extra module, which is gone
---
DROP FUNCTION IF EXISTS pgstrom.license_query();

CREATE FUNCTION pgstrom.gpu_devcaps()
  RETURNS TABLE (dev_id int,
                 device_name text,
                 tier text,
                 tier_reason text,
                 device_type text,
                 driver text,
                 caps text,
                 reported_caps text,
                 subgroup_size_min int,
                 subgroup_size_max int,
                 subgroup_size_observed int,
                 max_shared_memory int,
                 max_storage_buffer_range bigint,
                 device_memory bigint,
                 selected bool)
  AS 'MODULE_PATHNAME','pgstrom_gpu_devcaps'
  LANGUAGE C STRICT;

---
--- State of the GPU service process
---
CREATE FUNCTION pgstrom.gpu_service_state(OUT state text,
                                          OUT pid int,
                                          OUT generation bigint,
                                          OUT restart_count int,
                                          OUT selected_device int,
                                          OUT socket_path text,
                                          OUT last_error text)
  RETURNS record
  AS 'MODULE_PATHNAME','pgstrom_gpu_service_state'
  LANGUAGE C STRICT;

---
--- Connect to the GPU service and read its handshake reply
---
CREATE FUNCTION pgstrom.gpu_service_ping()
  RETURNS text
  AS 'MODULE_PATHNAME','pgstrom_gpu_service_ping'
  LANGUAGE C STRICT;

---
--- bool_and / every / bool_or: the minimum and the maximum of the 0/1 values (A6)
---
CREATE FUNCTION pgstrom.pmin(bool)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(bool,bool)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(bool)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(bool,bool)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_int64'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_bool(bytea)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_bool'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fmin_bool(bytea)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_bool'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fmax_bool(bytea)
  RETURNS bool
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_bool'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE AGGREGATE pgstrom.min_bool(bytea)
(
  sfunc = pgstrom.fmin_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_bool,
  parallel = safe
);
CREATE AGGREGATE pgstrom.max_bool(bytea)
(
  sfunc = pgstrom.fmax_trans_int64,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_bool,
  parallel = safe
);

---
--- min / max of a numeric of a known scale: the partial state is the numeric in a bytea (A6)
---
CREATE FUNCTION pgstrom.pmin(numeric)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmin(numeric,bool)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(numeric)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_numeric'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.pmax(numeric,bool)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_partial_minmax_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fmin_trans_numeric(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fmin_trans_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fmax_trans_numeric(bytea,bytea)
  RETURNS bytea
  AS 'MODULE_PATHNAME','pgstrom_fmax_trans_numeric'
  LANGUAGE C CALLED ON NULL INPUT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fminmax_final_numx(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_numx'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fmin_numx(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_numx'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE FUNCTION pgstrom.fmax_numx(bytea)
  RETURNS numeric
  AS 'MODULE_PATHNAME','pgstrom_fminmax_final_numx'
  LANGUAGE C STRICT PARALLEL SAFE;
CREATE AGGREGATE pgstrom.min_numx(bytea)
(
  sfunc = pgstrom.fmin_trans_numeric,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_numx,
  parallel = safe
);
CREATE AGGREGATE pgstrom.max_numx(bytea)
(
  sfunc = pgstrom.fmax_trans_numeric,
  stype = bytea,
  finalfunc = pgstrom.fminmax_final_numx,
  parallel = safe
);
