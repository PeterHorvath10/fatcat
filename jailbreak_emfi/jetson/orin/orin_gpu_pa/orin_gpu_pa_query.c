// SPDX-License-Identifier: GPL-2.0

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "orin_gpu_pa_uapi.h"

static unsigned long long parse_number(const char *text)
{
	char *end = NULL;
	unsigned long long value;

	errno = 0;
	value = strtoull(text, &end, 0);
	if (errno != 0 || end == text || *end != '\0') {
		fprintf(stderr, "invalid integer: %s\n", text);
		exit(2);
	}
	return value;
}

static int query_address(int fd, unsigned int pid, unsigned long long gpu_va,
			 int quiet)
{
	struct orin_gpu_pa_query query = {
		.pid = pid,
		.gpu_va = gpu_va,
	};

	if (ioctl(fd, ORIN_GPU_PA_IOC_QUERY, &query) != 0) {
		fprintf(stderr, "ioctl: %s\n", strerror(errno));
		return 1;
	}
	if (query.status != 0) {
		if (!quiet)
			printf("gpu_va=0x%016" PRIx64 " status=%d(%s)\n",
			       (uint64_t)query.gpu_va, query.status,
			       strerror(-query.status));
		return 1;
	}

	printf("gpu_va=0x%016" PRIx64
	       " page=%uK pte=%08x:%08x aperture=0x%x"
	       " gmmu=0x%010" PRIx64 " iova=0x%010" PRIx64
	       " pa=0x%010" PRIx64
	       " mapping=0x%" PRIx64 "+0x%" PRIx64 " map_pgsz_idx=%u"
	       " channel=%u candidate_package=%u\n",
	       (uint64_t)query.gpu_va, 1U << (query.page_shift - 10),
	       query.pte[1], query.pte[0], query.aperture,
	       (uint64_t)query.gmmu_address, (uint64_t)query.iova,
	       (uint64_t)query.physical_address, (uint64_t)query.mapping_start,
	       (uint64_t)query.mapping_size, query.mapping_page_size_index,
	       query.channel, query.candidate_package);
	return 0;
}

static void usage(const char *program)
{
	fprintf(stderr,
		"usage:\n"
		"  %s PID GPU_VA\n"
		"  %s scan PID START LENGTH [STEP]\n",
		program, program);
	exit(2);
}

int main(int argc, char **argv)
{
	unsigned long long start, length, step, offset;
	unsigned int pid;
	int fd;

	fd = open("/dev/orin_gpu_pa", O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr, "open /dev/orin_gpu_pa: %s\n", strerror(errno));
		return 1;
	}

	if (argc == 3 && strcmp(argv[1], "scan") != 0) {
		pid = (unsigned int)parse_number(argv[1]);
		start = parse_number(argv[2]);
		return query_address(fd, pid, start, 0);
	}
	if ((argc == 5 || argc == 6) && strcmp(argv[1], "scan") == 0) {
		pid = (unsigned int)parse_number(argv[2]);
		start = parse_number(argv[3]);
		length = parse_number(argv[4]);
		step = argc == 6 ? parse_number(argv[5]) : 2ULL * 1024 * 1024;
		if (length == 0 || step == 0)
			usage(argv[0]);
		for (offset = 0; offset < length; offset += step)
			query_address(fd, pid, start + offset, 1);
		close(fd);
		return 0;
	}

	close(fd);
	usage(argv[0]);
}
