/*
 * gpuserv_proto.c
 *
 * Text serialisation of the probe result, shared by the service (writer) and the
 * PostgreSQL module (parser). libc only.
 */
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "gpuserv_proto.h"

static const char *const cap_names[GPUCAP_NUM] = {
	"int64", "storage8", "subgroup_ballot", "subgroup_arith32", "subgroup_arith64",
	"atomic_i64_add", "atomic_i64_minmax", "atomic_i64_cas", "fp64", "atomic_f32_add",
	"bda", "memory_model", "host_import", "unified_memory", "fp16"
};

const char *
gpuserv_cap_name(int bit)
{
	return (bit >= 0 && bit < GPUCAP_NUM) ? cap_names[bit] : "?";
}

const char *
gpuserv_tier_name(uint32_t tier)
{
	switch (tier)
	{
		case GPUSERV_TIER_BASE:		return "base";
		case GPUSERV_TIER_DESKTOP:	return "desktop";
		default:					return "none";
	}
}

static void
write_caps(FILE *out, int index, const char *key, uint64_t caps)
{
	int		i, n = 0;

	fprintf(out, "dev%d.%s=", index, key);
	for (i = 0; i < GPUCAP_NUM; i++)
		if (caps & (1ULL << i))
			fprintf(out, "%s%s", n++ ? "," : "", cap_names[i]);
	fputc('\n', out);
}

/* values are single-line; strip anything that would break the format */
static void
write_str(FILE *out, int index, const char *key, const char *val)
{
	fprintf(out, "dev%d.%s=", index, key);
	for (; *val; val++)
		fputc((*val == '\n' || *val == '\r') ? ' ' : *val, out);
	fputc('\n', out);
}

void
gpuserv_probe_write_header(FILE *out, int ndevices)
{
	fprintf(out, "proto=%d\nndevices=%d\n", GPUSERV_PROTO_VERSION, ndevices);
}

void
gpuserv_probe_write_footer(FILE *out)
{
	fprintf(out, "end\n");
	fflush(out);
}

void
gpuserv_devcaps_write(FILE *out, int i, const gpuserv_devcaps *dc)
{
	write_str(out, i, "name", dc->name);
	write_str(out, i, "driver_name", dc->driver_name);
	write_str(out, i, "driver_info", dc->driver_info);
	write_str(out, i, "uuid", dc->uuid);
	write_str(out, i, "tier_reason", dc->tier_reason);
	fprintf(out, "dev%d.vendor_id=%u\n", i, dc->vendor_id);
	fprintf(out, "dev%d.device_id=%u\n", i, dc->device_id);
	fprintf(out, "dev%d.device_type=%u\n", i, dc->device_type);
	fprintf(out, "dev%d.api_version=%u\n", i, dc->api_version);
	fprintf(out, "dev%d.driver_version=%u\n", i, dc->driver_version);
	fprintf(out, "dev%d.tier=%s\n", i, gpuserv_tier_name(dc->tier));
	write_caps(out, i, "caps", dc->caps);
	write_caps(out, i, "reported", dc->reported_caps);
	fprintf(out, "dev%d.subgroup_size_min=%u\n", i, dc->subgroup_size_min);
	fprintf(out, "dev%d.subgroup_size_max=%u\n", i, dc->subgroup_size_max);
	fprintf(out, "dev%d.subgroup_size_observed=%u\n", i, dc->subgroup_size_observed);
	fprintf(out, "dev%d.max_shared_memory=%u\n", i, dc->max_shared_memory);
	fprintf(out, "dev%d.max_wg_invocations=%u\n", i, dc->max_wg_invocations);
	fprintf(out, "dev%d.max_push_constants=%u\n", i, dc->max_push_constants);
	fprintf(out, "dev%d.host_import_alignment=%u\n", i, dc->host_import_alignment);
	fprintf(out, "dev%d.probes_ok=%u\n", i, dc->probes_ok);
	fprintf(out, "dev%d.probes_failed=%u\n", i, dc->probes_failed);
	fprintf(out, "dev%d.max_storage_buffer_range=%llu\n", i, (unsigned long long) dc->max_storage_buffer_range);
	fprintf(out, "dev%d.max_buffer_size=%llu\n", i, (unsigned long long) dc->max_buffer_size);
	fprintf(out, "dev%d.device_memory=%llu\n", i, (unsigned long long) dc->device_memory);
}

static uint64_t
parse_caps(const char *val)
{
	uint64_t	caps = 0;
	char		buf[512];
	char	   *tok, *save = NULL;
	int			i;

	strncpy(buf, val, sizeof(buf) - 1);
	buf[sizeof(buf) - 1] = '\0';
	for (tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save))
		for (i = 0; i < GPUCAP_NUM; i++)
			if (strcmp(tok, cap_names[i]) == 0)
				caps |= (1ULL << i);
	return caps;
}

static void
copy_str(char *dst, size_t dstsz, const char *src)
{
	strncpy(dst, src, dstsz - 1);
	dst[dstsz - 1] = '\0';
}

static int
apply_field(gpuserv_devcaps *dc, const char *key, const char *val)
{
	unsigned long long v = strtoull(val, NULL, 10);

	if (!strcmp(key, "name"))				copy_str(dc->name, sizeof(dc->name), val);
	else if (!strcmp(key, "driver_name"))	copy_str(dc->driver_name, sizeof(dc->driver_name), val);
	else if (!strcmp(key, "driver_info"))	copy_str(dc->driver_info, sizeof(dc->driver_info), val);
	else if (!strcmp(key, "uuid"))			copy_str(dc->uuid, sizeof(dc->uuid), val);
	else if (!strcmp(key, "tier_reason"))	copy_str(dc->tier_reason, sizeof(dc->tier_reason), val);
	else if (!strcmp(key, "vendor_id"))		dc->vendor_id = v;
	else if (!strcmp(key, "device_id"))		dc->device_id = v;
	else if (!strcmp(key, "device_type"))	dc->device_type = v;
	else if (!strcmp(key, "api_version"))	dc->api_version = v;
	else if (!strcmp(key, "driver_version"))	dc->driver_version = v;
	else if (!strcmp(key, "tier"))
		dc->tier = !strcmp(val, "desktop") ? GPUSERV_TIER_DESKTOP :
				   !strcmp(val, "base") ? GPUSERV_TIER_BASE : GPUSERV_TIER_NONE;
	else if (!strcmp(key, "caps"))			dc->caps = parse_caps(val);
	else if (!strcmp(key, "reported"))		dc->reported_caps = parse_caps(val);
	else if (!strcmp(key, "subgroup_size_min"))		dc->subgroup_size_min = v;
	else if (!strcmp(key, "subgroup_size_max"))		dc->subgroup_size_max = v;
	else if (!strcmp(key, "subgroup_size_observed"))	dc->subgroup_size_observed = v;
	else if (!strcmp(key, "max_shared_memory"))		dc->max_shared_memory = v;
	else if (!strcmp(key, "max_wg_invocations"))	dc->max_wg_invocations = v;
	else if (!strcmp(key, "max_push_constants"))	dc->max_push_constants = v;
	else if (!strcmp(key, "host_import_alignment"))	dc->host_import_alignment = v;
	else if (!strcmp(key, "probes_ok"))		dc->probes_ok = v;
	else if (!strcmp(key, "probes_failed"))	dc->probes_failed = v;
	else if (!strcmp(key, "max_storage_buffer_range"))	dc->max_storage_buffer_range = v;
	else if (!strcmp(key, "max_buffer_size"))	dc->max_buffer_size = v;
	else if (!strcmp(key, "device_memory"))	dc->device_memory = v;
	/* unknown keys are ignored: a newer service may report more */
	return 0;
}

int
gpuserv_probe_parse(const char *text, gpuserv_devcaps *devs, int max)
{
	const char *p = text;
	int			ndev = -1, proto = -1, seen_end = 0;

	for (int i = 0; i < max; i++)
		memset(&devs[i], 0, sizeof(gpuserv_devcaps));
	while (*p)
	{
		const char *eol = strchr(p, '\n');
		size_t		len = eol ? (size_t)(eol - p) : strlen(p);
		char		line[512];
		char	   *eq;

		if (len >= sizeof(line))
			len = sizeof(line) - 1;
		memcpy(line, p, len);
		line[len] = '\0';
		p += (eol ? len + 1 : len);
		if (len > 0 && line[len - 1] == '\r')
			line[len - 1] = '\0';
		if (line[0] == '\0')
			continue;
		if (!strcmp(line, "end"))
		{
			seen_end = 1;
			break;
		}
		eq = strchr(line, '=');
		if (!eq)
			continue;
		*eq++ = '\0';
		if (!strcmp(line, "proto"))
			proto = atoi(eq);
		else if (!strcmp(line, "ndevices"))
			ndev = atoi(eq);
		else if (!strncmp(line, "dev", 3) && isdigit((unsigned char) line[3]))
		{
			char   *dot;
			int		idx = (int) strtol(line + 3, &dot, 10);

			if (*dot == '.' && idx >= 0 && idx < max)
				apply_field(&devs[idx], dot + 1, eq);
		}
	}
	if (!seen_end || proto != GPUSERV_PROTO_VERSION || ndev < 0)
		return -1;
	return ndev < max ? ndev : max;
}
