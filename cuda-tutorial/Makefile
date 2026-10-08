# Build every example next to its source:   make        (on a GPU node: -arch=native)
# One example:                               make 1-first-kernel/hello
# Another GPU target:                        make ARCH=-arch=sm_75
NVCC      ?= nvcc
ARCH      ?= -arch=native
NVCCFLAGS ?= -O2 -std=c++17 $(ARCH)

SRCS := $(wildcard */*.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

# Compile inside each lesson directory, so error messages name just the file
# (e.g. "CUDA error at pointer_mistake1.cu:21", as on the slides).
# thrust and atomics examples pass device lambdas to kernels.
6-optional-thrust-atomics/%: 6-optional-thrust-atomics/%.cu
	cd $(dir $<) && $(NVCC) $(NVCCFLAGS) --extended-lambda $(notdir $<) -o $(notdir $@)

%: %.cu common/check.h
	cd $(dir $<) && $(NVCC) $(NVCCFLAGS) $(notdir $<) -o $(notdir $@)

clean:
	rm -f $(BINS)

.PHONY: all clean
