#ifndef ORIN_GPU_PA_UAPI_H
#define ORIN_GPU_PA_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define ORIN_GPU_PA_IOC_MAGIC 0xB7

struct orin_gpu_pa_query {
	__u32 pid;
	__u32 page_shift;
	__u64 gpu_va;
	__u32 pte[2];
	__u32 aperture;
	__u64 gmmu_address;
	__u64 iova;
	__u64 physical_address;
	__u64 mapping_start;
	__u64 mapping_size;
	__u32 channel;
	__u32 candidate_package;
	__u32 mapping_page_size_index;
	__s32 status;
};

#define ORIN_GPU_PA_IOC_QUERY \
	_IOWR(ORIN_GPU_PA_IOC_MAGIC, 1, struct orin_gpu_pa_query)

#endif
