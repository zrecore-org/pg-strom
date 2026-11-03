/*
 * probe_main.c - gpuserv-probe: print the machine-readable device probe on stdout.
 * Human log lines (--verbose) go to stderr; stdout carries only the probe text.
 */
#include <stdio.h>
#include <string.h>
#include "gpuserv_probe.h"

int
main(int argc, char **argv)
{
	static gpuserv_devcaps devs[GPUSERV_MAX_DEVICES];
	int		allow_cpu = 0, verbose = 0, n;

	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--allow-cpu"))
			allow_cpu = 1;
		else if (!strcmp(argv[i], "--verbose"))
			verbose = 1;
		else
		{
			fprintf(stderr, "usage: %s [--allow-cpu] [--verbose]\n", argv[0]);
			return strcmp(argv[i], "--help") ? 2 : 0;
		}
	}
	n = gpuserv_probe_devices(devs, GPUSERV_MAX_DEVICES, allow_cpu, verbose ? stderr : NULL);
	if (n < 0)
	{
		fprintf(stderr, "gpuserv-probe: %s\n", gpuserv_probe_last_error());
		return 1;
	}
	gpuserv_probe_write_header(stdout, n);
	for (int i = 0; i < n; i++)
		gpuserv_devcaps_write(stdout, i, &devs[i]);
	gpuserv_probe_write_footer(stdout);
	return 0;
}
