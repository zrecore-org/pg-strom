/*
 * proto_check.c - sanity checks of the probe text format. With a file argument: parse it,
 * write it back and require an identical text (round trip). Always also round-trips a
 * synthetic record and checks that truncated text is rejected.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpuserv_proto.h"

static char *
slurp(FILE *f)
{
	size_t	cap = 4096, len = 0, r;
	char   *buf = malloc(cap);

	while (buf && (r = fread(buf + len, 1, cap - len - 1, f)) > 0)
	{
		len += r;
		if (cap - len < 2)
		{
			cap *= 2;
			buf = realloc(buf, cap);
		}
	}
	if (buf)
		buf[len] = '\0';
	return buf;
}

static char *
render(const gpuserv_devcaps *devs, int n)
{
	char   *buf = NULL;
	size_t	len = 0;
	FILE   *f = open_memstream(&buf, &len);

	gpuserv_probe_write_header(f, n);
	for (int i = 0; i < n; i++)
		gpuserv_devcaps_write(f, i, &devs[i]);
	gpuserv_probe_write_footer(f);
	fclose(f);
	return buf;
}

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "proto_check FAIL: %s\n", msg); return 1; } } while (0)

int
main(int argc, char **argv)
{
	static gpuserv_devcaps in[2], out[2];
	char   *t1, *t2;

	/* synthetic */
	snprintf(in[0].name, sizeof(in[0].name), "Test GPU");
	snprintf(in[0].uuid, sizeof(in[0].uuid), "00112233445566778899aabbccddeeff");
	snprintf(in[0].tier_reason, sizeof(in[0].tier_reason), "none");
	in[0].tier = GPUSERV_TIER_DESKTOP;
	in[0].caps = GPUCAP_INT64 | GPUCAP_FP16 | GPUCAP_BDA;
	in[0].reported_caps = in[0].caps | GPUCAP_SUBGROUP_ARITH64;
	in[0].max_storage_buffer_range = 1ULL << 33;
	in[0].device_memory = 17179869184ULL;
	in[0].subgroup_size_observed = 32;
	t1 = render(in, 1);
	CHECK(gpuserv_probe_parse(t1, out, 2) == 1, "synthetic parse");
	t2 = render(out, 1);
	CHECK(strcmp(t1, t2) == 0, "synthetic round trip differs");
	CHECK(out[0].caps == in[0].caps && out[0].reported_caps == in[0].reported_caps, "caps");
	CHECK(out[0].max_storage_buffer_range == (1ULL << 33), "u64 field");
	{
		char	cut[4096];
		char   *p = strstr(t1, "\nend\n");

		CHECK(p != NULL, "no end marker");
		memcpy(cut, t1, p - t1 + 1);
		cut[p - t1 + 1] = '\0';
		CHECK(gpuserv_probe_parse(cut, out, 2) == -1, "truncated text must be rejected");
	}
	free(t1);
	free(t2);

	if (argc > 1)
	{
		FILE   *f = fopen(argv[1], "r");
		int		n;

		CHECK(f != NULL, "cannot open probe file");
		t1 = slurp(f);
		fclose(f);
		n = gpuserv_probe_parse(t1, out, 2);
		CHECK(n >= 0, "probe text does not parse");
		t2 = render(out, n < 2 ? n : 2);
		if (n <= 2)
			CHECK(strcmp(t1, t2) == 0, "probe text round trip differs");
		printf("proto_check ok: %d device(s)\n", n);
		free(t1);
		free(t2);
	}
	else
		printf("proto_check ok (synthetic only)\n");
	return 0;
}
