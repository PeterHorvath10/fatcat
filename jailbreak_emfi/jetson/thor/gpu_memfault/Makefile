# Build on Thor:  make NVCC=<cuda>/bin/nvcc ARCH=<gpu arch>   (values: [memfault] nvcc / cuda_arch in config/thor.toml)
NVCC ?= nvcc
ifndef ARCH
$(error set ARCH to the GPU architecture, e.g. make ARCH=sm_XXX (memfault.cuda_arch in config/thor.toml))
endif

gpu_memfault: gpu_memfault.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $< -lcuda

clean:
	rm -f gpu_memfault
