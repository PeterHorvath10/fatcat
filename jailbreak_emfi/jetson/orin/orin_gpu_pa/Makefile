ifneq ($(KERNELRELEASE),)
obj-m := orin_gpu_pa.o
else
KDIR ?= /lib/modules/$(shell uname -r)/build

.PHONY: all clean

all: orin_gpu_pa_query
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

orin_gpu_pa_query: orin_gpu_pa_query.c orin_gpu_pa_uapi.h
	$(CC) -O2 -Wall -Wextra -o $@ orin_gpu_pa_query.c

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	$(RM) orin_gpu_pa_query
endif
