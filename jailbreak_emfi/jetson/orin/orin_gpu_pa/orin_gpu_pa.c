// SPDX-License-Identifier: GPL-2.0
/*
 * Read-only Jetson Orin nvgpu CUDA-VA to system-PA diagnostic.
 *
 * The stock R36.5 nvgpu driver does not export its page-table lookup helper.
 * Root supplies the live addresses of nvgpu_get_pte and gk20a_as_ops from
 * /proc/kallsyms when loading this temporary module.  No kernel memory or GPU
 * mapping is modified.
 */

#include <linux/device.h>
#include <linux/fdtable.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/iommu.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pid.h>
#include <linux/platform_device.h>
#include <linux/sched/task.h>
#include <linux/string.h>
#include <linux/uaccess.h>

#include "orin_gpu_pa_uapi.h"

#define ORIN_IOMMU_MARKER_BIT 37
#define ORIN_SMALL_PAGE_SHIFT 12
#define ORIN_BIG_PAGE_SHIFT 16

static unsigned long get_pte_addr;
module_param(get_pte_addr, ulong, 0400);
MODULE_PARM_DESC(get_pte_addr,
	"runtime address of the nvgpu_get_pte symbol from /proc/kallsyms");

static unsigned long as_ops_addr;
module_param(as_ops_addr, ulong, 0400);
MODULE_PARM_DESC(as_ops_addr,
	"runtime address of the gk20a_as_ops symbol from /proc/kallsyms");

static unsigned long find_mapped_addr;
module_param(find_mapped_addr, ulong, 0400);
MODULE_PARM_DESC(find_mapped_addr,
	"runtime address of nvgpu_vm_find_mapped_buf_range from /proc/kallsyms");

/* The first fields are stable in NVIDIA R36.5's struct gk20a_as_share. */
struct orin_as_share_prefix {
	void *g;
	void *vm;
	int id;
};

typedef int (*orin_get_pte_fn)(void *g, void *vm, u64 vaddr, u32 *pte);
typedef void *(*orin_find_mapped_fn)(void *vm, u64 vaddr);

/* Prefix of R36.5 struct nvgpu_mapped_buf.  Only addr, size, and pgsz_idx
 * are read.  The explicit intermediate fields preserve their verified ABI
 * offsets without depending on nvgpu's private headers at module build time.
 */
struct orin_nvgpu_rbtree_node {
	u64 key_start;
	u64 key_end;
	bool is_red;
	void *parent;
	void *left;
	void *right;
};

struct orin_mapped_buf_prefix {
	void *vm;
	void *vm_area;
	atomic_t refcount;
	struct orin_nvgpu_rbtree_node node;
	struct {
		void *prev;
		void *next;
	} buffer_list;
	u64 addr;
	u64 size;
	u32 pgsz_idx;
};

struct orin_find_as_ctx {
	const struct file_operations *as_fops;
	struct file *file;
	unsigned int fd;
};

struct orin_find_dev_ctx {
	struct device *dev;
};

static struct device *orin_gpu_dev;
static struct iommu_domain *orin_gpu_domain;

static int orin_find_gpu_device(struct device *dev, void *data)
{
	struct orin_find_dev_ctx *ctx = data;

	if (strcmp(dev_name(dev), "17000000.gpu") != 0)
		return 0;

	ctx->dev = get_device(dev);
	return 1;
}

static int orin_find_as_file(const void *data, struct file *file,
			     unsigned int fd)
{
	struct orin_find_as_ctx *ctx = (struct orin_find_as_ctx *)data;

	if (file->f_op != ctx->as_fops)
		return 0;

	get_file(file);
	ctx->file = file;
	ctx->fd = fd;
	return 1;
}

static struct file *orin_get_process_as_file(pid_t nr)
{
	struct orin_find_as_ctx ctx = {
		.as_fops = (const struct file_operations *)as_ops_addr,
	};
	struct task_struct *task;
	struct pid *pid;

	pid = find_get_pid(nr);
	if (pid == NULL)
		return ERR_PTR(-ESRCH);

	task = get_pid_task(pid, PIDTYPE_PID);
	put_pid(pid);
	if (task == NULL)
		return ERR_PTR(-ESRCH);

	/* task_lock keeps task->files stable; iterate_fd locks the fd table. */
	task_lock(task);
	if (task->files != NULL)
		iterate_fd(task->files, 0, orin_find_as_file, &ctx);
	task_unlock(task);
	put_task_struct(task);

	if (ctx.file == NULL)
		return ERR_PTR(-ENOENT);
	return ctx.file;
}

static bool orin_same_pte(const u32 left[2], const u32 right[2])
{
	return left[0] == right[0] && left[1] == right[1];
}

static u32 orin_page_shift(orin_get_pte_fn get_pte,
			   struct orin_as_share_prefix *share, u64 gpu_va,
			   const u32 current_pte[2])
{
	u64 page_va = gpu_va & ~((1ULL << ORIN_SMALL_PAGE_SHIFT) - 1ULL);
	u32 adjacent[2] = { 0, 0 };

	/* All GA10B user mappings are 4 KiB or 64 KiB.  A big-page PTE is
	 * unchanged for neighboring 4 KiB addresses within that 64 KiB page.
	 */
	if (get_pte(share->g, share->vm,
		    page_va + (1ULL << ORIN_SMALL_PAGE_SHIFT), adjacent) == 0 &&
	    orin_same_pte(current_pte, adjacent))
		return ORIN_BIG_PAGE_SHIFT;

	if (page_va >= (1ULL << ORIN_SMALL_PAGE_SHIFT) &&
	    get_pte(share->g, share->vm,
		    page_va - (1ULL << ORIN_SMALL_PAGE_SHIFT), adjacent) == 0 &&
	    orin_same_pte(current_pte, adjacent))
		return ORIN_BIG_PAGE_SHIFT;

	return ORIN_SMALL_PAGE_SHIFT;
}

static u64 orin_decode_gmmu_address(const u32 pte[2])
{
	/* GA10B: PTE word 0 bits 31:8 contain address bits 35:12;
	 * word 1 bits 3:0 contain address bits 39:36.  Bits 31:24 in word 1
	 * are the memory kind and bits 23:4 are compression-tag metadata.
	 */
	return ((u64)((pte[0] >> 8) & 0x00ffffffU) << 12) |
	       ((u64)(pte[1] & 0x0000000fU) << 36);
}

static u32 orin_dram_channel(u64 physical_address)
{
	static const u64 masks[3] = {
		0xb94f5200ULL,
		0xcae4a400ULL,
		0x2e764800ULL,
	};
	u32 channel = 0;
	unsigned int bit;

	for (bit = 0; bit < ARRAY_SIZE(masks); bit++)
		channel |= (hweight64(physical_address & masks[bit]) & 1U) << bit;
	return channel;
}

static int orin_translate(struct orin_gpu_pa_query *query)
{
	struct orin_as_share_prefix *share;
	orin_get_pte_fn get_pte = (orin_get_pte_fn)get_pte_addr;
	orin_find_mapped_fn find_mapped = (orin_find_mapped_fn)find_mapped_addr;
	struct orin_mapped_buf_prefix *mapping;
	struct file *as_file;
	u64 page_offset;
	u64 translated;
	u32 channel;
	int err;

	as_file = orin_get_process_as_file((pid_t)query->pid);
	if (IS_ERR(as_file))
		return PTR_ERR(as_file);

	share = (struct orin_as_share_prefix *)as_file->private_data;
	if (share == NULL || share->g == NULL || share->vm == NULL) {
		err = -ENODEV;
		goto out_fput;
	}

	err = get_pte(share->g, share->vm, query->gpu_va, query->pte);
	if (err != 0)
		goto out_fput;
	if ((query->pte[0] & 1U) == 0U) {
		err = -ENOENT;
		goto out_fput;
	}

	mapping = find_mapped(share->vm, query->gpu_va);
	if (mapping == NULL || mapping->addr > query->gpu_va ||
	    mapping->size == 0ULL ||
	    query->gpu_va - mapping->addr >= mapping->size) {
		err = -EPROTO;
		goto out_fput;
	}
	query->mapping_start = mapping->addr;
	query->mapping_size = mapping->size;
	query->mapping_page_size_index = mapping->pgsz_idx;

	query->aperture = query->pte[0] & 0x6U;
	query->gmmu_address = orin_decode_gmmu_address(query->pte);
	query->page_shift = orin_page_shift(get_pte, share, query->gpu_va,
					    query->pte);
	page_offset = query->gpu_va & ((1ULL << query->page_shift) - 1ULL);

	if ((query->gmmu_address & BIT_ULL(ORIN_IOMMU_MARKER_BIT)) != 0ULL) {
		if (orin_gpu_domain == NULL) {
			err = -ENODEV;
			goto out_fput;
		}
		query->iova = query->gmmu_address &
			      ~BIT_ULL(ORIN_IOMMU_MARKER_BIT);
		query->iova += page_offset;
		translated = iommu_iova_to_phys(orin_gpu_domain, query->iova);
		if (translated == 0ULL) {
			err = -ENOENT;
			goto out_fput;
		}
		query->physical_address = translated;
	} else {
		/* Non-IOMMU mappings contain the system address directly. */
		query->iova = 0ULL;
		query->physical_address = query->gmmu_address + page_offset;
	}

	channel = orin_dram_channel(query->physical_address);
	query->channel = channel;
	/* EM fault localisation places channels 0,1,4,5 and 2,3,6,7 on
	 * separate packages, so channel bit 1 selects the package.
	 */
	query->candidate_package = (channel >> 1) & 1U;
	err = 0;

out_fput:
	fput(as_file);
	return err;
}

static long orin_gpu_pa_ioctl(struct file *file, unsigned int cmd,
			      unsigned long arg)
{
	struct orin_gpu_pa_query query;
	int err;

	if (cmd != ORIN_GPU_PA_IOC_QUERY)
		return -ENOTTY;
	if (copy_from_user(&query, (void __user *)arg, sizeof(query)) != 0)
		return -EFAULT;

	memset(query.pte, 0, sizeof(query.pte));
	query.page_shift = 0;
	query.aperture = 0;
	query.gmmu_address = 0;
	query.iova = 0;
	query.physical_address = 0;
	query.mapping_start = 0;
	query.mapping_size = 0;
	query.channel = 0;
	query.candidate_package = 0;
	query.mapping_page_size_index = 0;
	err = orin_translate(&query);
	query.status = err;

	if (copy_to_user((void __user *)arg, &query, sizeof(query)) != 0)
		return -EFAULT;
	return 0;
}

static const struct file_operations orin_gpu_pa_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = orin_gpu_pa_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = orin_gpu_pa_ioctl,
#endif
};

static struct miscdevice orin_gpu_pa_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "orin_gpu_pa",
	.fops = &orin_gpu_pa_fops,
	.mode = 0660,
};

static int __init orin_gpu_pa_init(void)
{
	struct orin_find_dev_ctx ctx = { };
	int err;

	if (get_pte_addr == 0UL || as_ops_addr == 0UL ||
	    find_mapped_addr == 0UL) {
		pr_err("orin_gpu_pa: all three nvgpu symbol addresses are required\n");
		return -EINVAL;
	}

	bus_for_each_dev(&platform_bus_type, NULL, &ctx, orin_find_gpu_device);
	if (ctx.dev == NULL) {
		pr_err("orin_gpu_pa: platform device 17000000.gpu not found\n");
		return -ENODEV;
	}

	orin_gpu_dev = ctx.dev;
	orin_gpu_domain = iommu_get_domain_for_dev(orin_gpu_dev);
	if (orin_gpu_domain == NULL)
		pr_info("orin_gpu_pa: GPU uses direct physical GMMU addresses\n");

	err = misc_register(&orin_gpu_pa_misc);
	if (err != 0)
		goto out_put_device;

	pr_info("orin_gpu_pa: read-only diagnostic loaded\n");
	return 0;

out_put_device:
	put_device(orin_gpu_dev);
	orin_gpu_dev = NULL;
	orin_gpu_domain = NULL;
	return err;
}

static void __exit orin_gpu_pa_exit(void)
{
	misc_deregister(&orin_gpu_pa_misc);
	if (orin_gpu_dev != NULL)
		put_device(orin_gpu_dev);
	orin_gpu_dev = NULL;
	orin_gpu_domain = NULL;
	pr_info("orin_gpu_pa: diagnostic unloaded\n");
}

module_init(orin_gpu_pa_init);
module_exit(orin_gpu_pa_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Anonymous");
MODULE_DESCRIPTION("Read-only Jetson Orin nvgpu CUDA VA to system PA mapper");
