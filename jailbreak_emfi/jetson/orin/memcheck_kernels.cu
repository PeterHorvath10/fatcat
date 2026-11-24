// Memory-check kernels of the Orin workload (jetson/orin/memcheck.py), compiled at run time with NVRTC.
//
// init_values fills the buffer with a hash of the word index; load_compute_reduce reads every word and
// folds it into one checksum per block (the glitched pass); compare_block_sums flags blocks whose
// checksum differs from the warm-up pass; scan_data_values then finds the exact words that differ from
// the expected pattern.
static __device__ __forceinline__ unsigned int mix32(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static __device__ __forceinline__ unsigned int rotl32(unsigned int x, int r)
{
    return (x << r) | (x >> (32 - r));
}

extern "C" __global__ void init_values(
    unsigned int *data,
    unsigned long long words,
    unsigned int seed)
{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long i = tid; i < words; i += stride) {
        unsigned int lo = (unsigned int)i;
        unsigned int hi = (unsigned int)(i >> 32);
        data[i] = mix32(lo ^ rotl32(hi + seed, 13) ^ 0x9e3779b9U);
    }
}

extern "C" __global__ void load_compute_reduce(
    const unsigned int *data,
    unsigned long long *block_sums,
    unsigned long long words)
{
    extern __shared__ unsigned long long scratch[];

    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long acc = 0xcbf29ce484222325ULL ^ tid;
    unsigned int pass_key = mix32(0xa5a5a5a5U);

    for (unsigned long long i = tid; i < words; i += stride) {
        unsigned int x = data[i];
        unsigned int y = mix32(x + pass_key + (unsigned int)i);
        unsigned int z = rotl32(y ^ x, (int)(i & 15U) + 1);

        acc += (unsigned long long)(y ^ z) + ((unsigned long long)x << 1);
        acc ^= acc << 7;
        acc ^= acc >> 9;
    }

    scratch[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] ^= scratch[threadIdx.x + offset] + 0x9e3779b97f4a7c15ULL;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = scratch[0];
    }
}

extern "C" __global__ void compare_block_sums(
    const unsigned long long *expected,
    const unsigned long long *actual,
    unsigned int *mismatch_flags,
    unsigned long long *expected_mismatch_addresses,
    unsigned long long *actual_mismatch_addresses,
    unsigned long long count)
{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long i = tid; i < count; i += stride) {
        unsigned int mismatch = expected[i] == actual[i] ? 0U : 1U;
        mismatch_flags[i] = mismatch;
        expected_mismatch_addresses[i] = mismatch ? (unsigned long long)(expected + i) : 0ULL;
        actual_mismatch_addresses[i] = mismatch ? (unsigned long long)(actual + i) : 0ULL;
    }
}

struct DataMismatchRecord
{
    unsigned long long word_index;
    unsigned int expected;
    unsigned int actual;
};

extern "C" __global__ void scan_data_values(
    const unsigned int *data,
    unsigned long long words,
    unsigned int seed,
    unsigned long long *mismatch_count,
    DataMismatchRecord *records,
    unsigned long long record_capacity)
{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long i = tid; i < words; i += stride) {
        unsigned int lo = (unsigned int)i;
        unsigned int hi = (unsigned int)(i >> 32);
        unsigned int expected = mix32(lo ^ rotl32(hi + seed, 13) ^ 0x9e3779b9U);
        unsigned int actual = data[i];
        if (actual != expected) {
            unsigned long long slot = atomicAdd(mismatch_count, 1ULL);
            if (slot < record_capacity) {
                records[slot].word_index = i;
                records[slot].expected = expected;
                records[slot].actual = actual;
            }
        }
    }
}
