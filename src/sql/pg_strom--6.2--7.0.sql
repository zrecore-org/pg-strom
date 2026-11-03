---
--- PG-Strom v6.2 -> v7.0 (Vulkan GPU service)
---

---
--- Real capabilities of each Vulkan device, as measured by the probe kernels
--- (caps) against what the API claimed (reported_caps)
---
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
