// gpumem.cu: EMFI target on the Jetson's GPU. Every shot streams a buffer of known content through
// global-memory loads and checks each loaded word.
//
// The buffer (--mb, default 256 MB; the Orin's GPU L2 is 2 MB, so every pass reads DRAM) is filled
// once. Each 32-bit word's expected value is computed from its index and never loaded, so a wrong
// value cannot come from the reference. A shot is one kernel of about --ms (passes over the buffer,
// calibrated at startup). A word that loads wrong is recorded with the GPU time and re-read 20 us
// later, after the pulse (ld.global.cv): the right value then means only that load was faulted.
// After the kernel, L2 is flushed and one more pass checks the buffer: whatever is still wrong is
// wrong in DRAM, and is rewritten before the next shot.
//
// Words checked are counted as executed (not derived from the loop bounds): fewer than expected
// means the kernel skipped work, a control-flow fault rather than a data fault.
//
// --mode write: the shot is one kernel that stores every word of the buffer exactly once, with a new
// hash seed each shot (so a lost write leaves the previous shot's value). Then L2 is flushed and the
// buffer is read back from DRAM, ~30 ms after the pulse, so those reads are not hit. Nothing is
// overwritten before that check: a write the pulse corrupted is still there. Lane 0 of each warp
// logs the time of each step, which places every bad word at the moment it was stored. The shot
// lasts as long as one pass takes: ~20 ms for 2 GB.
//
// --mode sm: no memory under test. Every thread runs a chain in its own registers for about --ms
// (8 integer lanes that feed each other, and a chaotic float map y = 3.99 y (1 - y), so any flipped
// bit spreads and never heals) and stores a hash of its state at N_CKPT checkpoints; those stores
// are the only memory traffic. Each checkpoint is compared with a fault-free run from startup. A
// state that is still wrong at every later checkpoint was corrupted inside the SM (register file,
// ALU, FPU, scheduler); a single wrong checkpoint means only that store was hit. Records:
// [us since the kernel start of the first wrong checkpoint, thread * 2 + (0 int, 1 float),
// expected, got, wrong checkpoints, first wrong checkpoint, SM].
//
// stdin, one JSON reply line each on stdout (diagnostics on stderr):
//   go ID [IDLE_MS]     sleep IDLE_MS (CPU asleep, GPU idle), then one shot
//   poke WORD VALUE     overwrite one buffer word in DRAM (tests the detection path)
//   skip WORD           write mode: the next shot does not store the 16 bytes holding WORD (tests it)
//   quit
// A CUDA error is replied as {"id", "error"} and ends the process (the context is gone).
//
// build: nvcc -O3 -arch=sm_87 -o gpumem gpumem.cu
// usage: gpumem [--mode read|write|sm] [--mb 256] [--ms 100] [--pattern hash|zeros|ones|alt]
//               [--load cg|ca|cv|nc] [--gap-ms 30] [--selftest 5]      (write: hash only, --ms unused)

#include <cuda_runtime.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// hash pattern: word w holds w * K ^ seed. K is odd, so this is a bijection on 32 bits: a wrong
// value that is another word's pattern names that word (gpumem_analyze.py inverts it). The seed is
// SEED, and in write mode SEED ^ shot * C, a new one every shot.
constexpr unsigned K = 0x9E3779B1u, SEED = 0x5A17C3E1u, C = 0x85EBCA6Bu;
constexpr int THREADS = 256, U = 4;          // U loads in flight per thread before any compare
constexpr int MAX_RECS = 4096, MAX_PRINT = 512;

enum Pattern { HASH, ZEROS, ONES, ALT };
enum Load { CG, CA, CV, NC };
enum Mode { READ, WRITE, SM };
static const char* PATTERNS[] = {"hash", "zeros", "ones", "alt"};
static const char* LOADS[] = {"cg", "ca", "cv", "nc"};
static const char* MODES[] = {"read", "write", "sm"};
constexpr int N_CKPT = 100, LANES = 8;       // SM mode: checkpoints per shot, integer lanes per thread

struct Rec { unsigned long long t; unsigned word, want, got, again, pass, sm; };
struct Head { unsigned long long t_start, t_end, checked, n_bad; };
struct Out { Head h; Rec rec[MAX_RECS]; };

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

template <int P> __device__ __forceinline__ unsigned expect(unsigned w, unsigned seed) {
  if (P == HASH) return w * K ^ seed;
  if (P == ZEROS) return 0u;
  if (P == ONES) return ~0u;
  return w & 1 ? 0xAAAAAAAAu : 0x55555555u;
}

// asm volatile: the compiler may neither merge the identical loads of successive passes nor drop
// the unused ones of flush(). cg: L2 only, ca: L1 and L2, cv: fetch again, nc: read-only path.
template <int L> __device__ __forceinline__ uint4 load(const uint4* p) {
  uint4 v;
#define LD(op) asm volatile("ld.global." op ".v4.u32 {%0,%1,%2,%3}, [%4];" \
                            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p))
  if (L == CG) LD("cg");
  else if (L == CA) LD("ca");
  else if (L == CV) LD("cv");
  else LD("nc");
#undef LD
  return v;
}

// Slow path, only on a mismatch. got is the register as compared: got == want in a record means
// the compare or branch was faulted, not the load.
__device__ __noinline__ void report(Out* out, const uint4* buf, unsigned w, unsigned want, unsigned got,
                                    unsigned pass) {
  const unsigned long long t = gtime();
  __nanosleep(20000);                        // re-read well after the pulse (80-960 ns)
  unsigned again, sm;
  asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(again) : "l"(reinterpret_cast<const unsigned*>(buf) + w));
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const unsigned long long k = atomicAdd(&out->h.n_bad, 1ull);
  if (k < MAX_RECS) out->rec[k] = {t, w, want, got, again, pass, sm};
}

template <int P>
__device__ __forceinline__ void check4(Out* out, const uint4* buf, unsigned i, uint4 x, unsigned pass, unsigned seed) {
  const unsigned w = 4 * i;
  if (x.x != expect<P>(w, seed)) report(out, buf, w, expect<P>(w, seed), x.x, pass);
  if (x.y != expect<P>(w + 1, seed)) report(out, buf, w + 1, expect<P>(w + 1, seed), x.y, pass);
  if (x.z != expect<P>(w + 2, seed)) report(out, buf, w + 2, expect<P>(w + 2, seed), x.z, pass);
  if (x.w != expect<P>(w + 3, seed)) report(out, buf, w + 3, expect<P>(w + 3, seed), x.w, pass);
}

// nvec is a multiple of U * the grid, so there are no bounds checks in the loop.
template <int P, int L>
__global__ void __launch_bounds__(THREADS) check(const uint4* buf, unsigned nvec, unsigned passes, unsigned seed,
                                                 Out* out) {
  if (threadIdx.x == 0) atomicMin(&out->h.t_start, gtime());
  const unsigned stride = gridDim.x * THREADS;
  unsigned n = 0;
  for (unsigned pass = 0; pass < passes; ++pass)
    for (unsigned i = blockIdx.x * THREADS + threadIdx.x; i < nvec; i += U * stride) {
      uint4 x[U];
#pragma unroll
      for (int u = 0; u < U; ++u) x[u] = load<L>(buf + i + u * stride);
#pragma unroll
      for (int u = 0; u < U; ++u) check4<P>(out, buf, i + u * stride, x[u], pass, seed);
      asm volatile("add.u32 %0, %0, %1;" : "+r"(n) : "n"(4 * U));
    }
  atomicAdd(&out->h.checked, (unsigned long long)n);
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&out->h.t_end, gtime());
}

template <int P> __global__ void fill(uint4* buf, unsigned nvec, unsigned seed) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += gridDim.x * blockDim.x) {
    const unsigned w = 4 * i;
    buf[i] = make_uint4(expect<P>(w, seed), expect<P>(w + 1, seed), expect<P>(w + 2, seed), expect<P>(w + 3, seed));
  }
}

// Write mode: every vector stored once, in the order of check() (U per thread per step), and
// lane 0 of each warp logs when it started each step. That is when the stores left the SM; L2
// writes a line to DRAM when it evicts it, ~20 us later at full rate.
__global__ void __launch_bounds__(THREADS) store(uint4* buf, unsigned nvec, unsigned seed, unsigned skip,
                                                 unsigned long long* ts, Out* out) {
  if (threadIdx.x == 0) atomicMin(&out->h.t_start, gtime());
  const unsigned stride = gridDim.x * THREADS, warps = stride / 32;
  const unsigned warp = (blockIdx.x * THREADS + threadIdx.x) / 32;
  unsigned n = 0, step = 0;
  for (unsigned i = blockIdx.x * THREADS + threadIdx.x; i < nvec; i += U * stride, ++step) {
    if ((threadIdx.x & 31) == 0) ts[(size_t)step * warps + warp] = gtime();
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const unsigned w = 4 * (i + u * stride);
      if (i + u * stride != skip) buf[i + u * stride] = make_uint4(expect<HASH>(w, seed), expect<HASH>(w + 1, seed), expect<HASH>(w + 2, seed),
                                       expect<HASH>(w + 3, seed));
    }
    asm volatile("add.u32 %0, %0, %1;" : "+r"(n) : "n"(4 * U));
  }
  atomicAdd(&out->h.checked, (unsigned long long)n);
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&out->h.t_end, gtime());
}

// SM mode: register-only chains, state stored at every checkpoint (see the header). Lane 0 of each
// warp logs when it stored each checkpoint, and thread 0 of each block its SM.
__global__ void __launch_bounds__(THREADS) alu(unsigned iters, unsigned seed, unsigned* ck, unsigned long long* ts,
                                               unsigned* smid, Out* out) {
  if (threadIdx.x == 0) {
    atomicMin(&out->h.t_start, gtime());
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smid[blockIdx.x] = sm;
  }
  const unsigned tid = blockIdx.x * THREADS + threadIdx.x, nthr = gridDim.x * THREADS, warps = nthr / 32;
  unsigned r[LANES];
#pragma unroll
  for (int k = 0; k < LANES; ++k) r[k] = (tid * LANES + k) * K ^ seed;
  float y = 0.1f + 0.8f * (float)(tid & 0xFFFF) / 65536.f;
  unsigned n = 0;
  for (int c = 0; c < N_CKPT; ++c) {
    for (unsigned it = 0; it < iters; ++it) {
#pragma unroll
      for (int k = 0; k < LANES; ++k) r[k] = r[k] * K + (r[(k + 1) % LANES] >> 7);
      y = 3.99f * y * (1.f - y);
    }
    unsigned h = 0;
#pragma unroll
    for (int k = 0; k < LANES; ++k) h = h * 0x01000193u ^ r[k];
    if ((threadIdx.x & 31) == 0) ts[(size_t)c * warps + tid / 32] = gtime();
    ck[((size_t)c * nthr + tid) * 2] = h;
    ck[((size_t)c * nthr + tid) * 2 + 1] = __float_as_uint(y);
    asm volatile("add.u32 %0, %0, 2;" : "+r"(n));
  }
  atomicAdd(&out->h.checked, (unsigned long long)n);
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&out->h.t_end, gtime());
}

// Reads a buffer 16x the L2 size: afterwards nothing of the checked buffer is left in L2.
__global__ void flush(const uint4* p, unsigned nvec) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += gridDim.x * blockDim.x) load<CG>(p + i);
}

using Check = void (*)(const uint4*, unsigned, unsigned, unsigned, Out*);
#define ROW(P) {check<P, CG>, check<P, CA>, check<P, CV>, check<P, NC>}
static const Check CHECKS[4][4] = {ROW(HASH), ROW(ZEROS), ROW(ONES), ROW(ALT)};
#undef ROW
static void (*const FILLS[4])(uint4*, unsigned, unsigned) = {fill<HASH>, fill<ZEROS>, fill<ONES>, fill<ALT>};

static int g_pattern = HASH, g_load = CG, g_mode = READ, g_blocks;
static unsigned g_nvec, g_nflush, g_passes, g_warps, g_seed = SEED, g_shots = 0, g_skip = ~0u;
static unsigned long long* d_ts;              // write mode: [step][warp] start times
static std::vector<unsigned long long> h_ts;
static double g_gap_ms = 30;
static unsigned *d_ck, *d_smid, g_iters, g_sm_blocks, g_nthr;   // SM mode
static unsigned long long* d_ck_ts;
static std::vector<unsigned> h_ck, g_gold, h_smid;
static std::vector<unsigned long long> h_ck_ts;
static uint4 *g_buf, *g_flush;
static Out* d_out;
static Out h_out, h_ver, h_tmp;               // host copies, 128 KB each
static cudaEvent_t g_e0, g_e1;
static std::string g_id = "init";

static void die(const char* what, cudaError_t e) {
  printf("{\"id\": \"%s\", \"error\": \"%s: %s\"}\n", g_id.c_str(), what, cudaGetErrorString(e));
  fflush(stdout);
  exit(3);
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) die(#x, e_); } while (0)

static double now_ms() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void sleep_ms(double ms) {
  if (ms > 0) std::this_thread::sleep_for(std::chrono::microseconds((long long)(ms * 1e3)));
}

// One kernel (check or store) into d_out, results copied to h. Returns its time in ms (CUDA events).
template <class Launch> static float timed(Launch launch, Out& h) {
  const Head init = {~0ull, 0, 0, 0};
  CK(cudaMemcpy(&d_out->h, &init, sizeof init, cudaMemcpyHostToDevice));
  CK(cudaEventRecord(g_e0));
  launch();
  CK(cudaGetLastError());
  CK(cudaEventRecord(g_e1));
  CK(cudaEventSynchronize(g_e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, g_e0, g_e1));
  CK(cudaMemcpy(&h.h, &d_out->h, sizeof h.h, cudaMemcpyDeviceToHost));
  if (h.h.n_bad)
    CK(cudaMemcpy(h.rec, d_out->rec, std::min<unsigned long long>(h.h.n_bad, MAX_RECS) * sizeof(Rec),
                  cudaMemcpyDeviceToHost));
  return ms;
}

static float run(Check k, unsigned passes, Out& h) {
  return timed([&] { k<<<g_blocks, THREADS>>>(g_buf, g_nvec, passes, g_seed, d_out); }, h);
}

static void evict() {
  flush<<<g_blocks, THREADS>>>(g_flush, g_nflush);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
}

static void refill() {
  FILLS[g_pattern]<<<g_blocks, THREADS>>>(g_buf, g_nvec, g_seed);
  CK(cudaGetLastError());
  evict();                                    // also writes the filled lines back to DRAM
}

// L2 flushed, then one pass with ld.global.cv: whatever reads wrong now is wrong in DRAM.
static unsigned long long verify(Out& h) {
  evict();
  run(CHECKS[g_pattern][CV], 1, h);
  return h.h.n_bad + (h.h.checked != 4ull * g_nvec);
}

// Write mode: when the word was stored, us after the store kernel started (h_ts copied back).
static double stored_us(unsigned word) {
  const unsigned i = word / 4, stride = g_blocks * THREADS;
  const size_t k = (size_t)(i / (U * stride)) * g_warps + (i % stride) / 32;
  return (double)(long long)(h_ts[k] - h_out.h.t_start) / 1e3;
}

// [t_us, word, expected, loaded, re-read, pass, SM]; t_us since the start of h's kernel, or with
// at_store, since the store kernel started, when the word was stored.
static void print_recs(const Out& h, bool at_store = false) {
  const unsigned n = (unsigned)std::min<unsigned long long>(h.h.n_bad, MAX_PRINT);
  printf("[");
  for (unsigned r = 0; r < n; ++r) {
    const Rec& x = h.rec[r];
    printf("%s[%.3f, %u, %u, %u, %u, %u, %u]", r ? ", " : "",
           at_store ? stored_us(x.word) : (double)(long long)(x.t - h.h.t_start) / 1e3, x.word, x.want, x.got,
           x.again, x.pass, x.sm);
  }
  printf("]");
}

// Write mode without the reply: a new seed stored, L2 flushed, read back into h_ver. Returns the
// store kernel's time; h_out holds its head, h_ts its step times when anything read back wrong.
static float write_shot(double gap_ms) {
  g_seed = SEED ^ ++g_shots * C;
  const float ms = timed([] { store<<<g_blocks, THREADS>>>(g_buf, g_nvec, g_seed, g_skip, d_ts, d_out); }, h_out);
  g_skip = ~0u;
  sleep_ms(gap_ms);
  if (verify(h_ver))
    CK(cudaMemcpy(h_ts.data(), d_ts, h_ts.size() * sizeof h_ts[0], cudaMemcpyDeviceToHost));
  return ms;
}

// SM mode: one alu() kernel, its checkpoints copied to h_ck. Returns the kernel's time.
static float sm_shot() {
  const float ms = timed([] { alu<<<g_sm_blocks, THREADS>>>(g_iters, SEED, d_ck, d_ck_ts, d_smid, d_out); }, h_out);
  CK(cudaMemcpy(h_ck.data(), d_ck, h_ck.size() * sizeof h_ck[0], cudaMemcpyDeviceToHost));
  return ms;
}

// SM mode: thread states that differ from the fault-free run (g_gold) into h_out (n_bad, rec), as
// described in the header. Returns n_bad.
static unsigned long long sm_compare() {
  unsigned long long nbad = 0;
  bool copied = false;
  for (unsigned t = 0; t < g_nthr; ++t)
    for (unsigned w = 0; w < 2; ++w) {
      int first = -1, nwrong = 0;
      for (int c = 0; c < N_CKPT; ++c) {
        const size_t i = ((size_t)c * g_nthr + t) * 2 + w;
        if (h_ck[i] != g_gold[i]) {
          if (first < 0) first = c;
          ++nwrong;
        }
      }
      if (first < 0) continue;
      if (!copied) {                         // the step times and SMs, only when needed
        CK(cudaMemcpy(h_ck_ts.data(), d_ck_ts, h_ck_ts.size() * sizeof h_ck_ts[0], cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_smid.data(), d_smid, h_smid.size() * sizeof h_smid[0], cudaMemcpyDeviceToHost));
        copied = true;
      }
      if (nbad < MAX_RECS) {
        const size_t i = ((size_t)first * g_nthr + t) * 2 + w;
        h_out.rec[nbad] = {h_ck_ts[(size_t)first * (g_nthr / 32) + t / 32], t * 2 + w, g_gold[i], h_ck[i],
                           (unsigned)nwrong, (unsigned)first, h_smid[t / THREADS]};
      }
      ++nbad;
    }
  h_out.h.n_bad = nbad;
  return nbad;
}

static unsigned long long words_expected() {
  return g_mode == SM ? 2ull * N_CKPT * g_nthr : 4ull * g_nvec * g_passes;
}

// The gap keeps the verify pass apart from the shot in the probe trace (and, in write mode, its
// reads well after the pulse). Write mode: checked = words stored, no loads to report (recs []);
// verify_recs are the words read back wrong, timed when they were stored.
static void shot(double idle_ms) {
  const double t_line = now_ms();
  sleep_ms(idle_ms);
  const double t_wait = now_ms() - t_line;
  const unsigned seed_prev = g_seed;
  float kernel_ms;
  unsigned long long vbad;
  int restored = -1;
  if (g_mode == WRITE) {
    kernel_ms = write_shot(g_gap_ms);
    vbad = h_ver.h.n_bad + (h_ver.h.checked != 4ull * g_nvec);
  } else if (g_mode == SM) {
    kernel_ms = sm_shot();
    sm_compare();
    h_ver.h.n_bad = 0;                        // nothing to verify: no buffer under test
    vbad = 0;
  } else {
    kernel_ms = run(CHECKS[g_pattern][g_load], g_passes, h_out);
    sleep_ms(g_gap_ms);
    vbad = verify(h_ver);
    if (vbad) {
      refill();
      restored = verify(h_tmp) == 0;
    }
  }
  const Head& h = h_out.h;
  printf("{\"id\": \"%s\", \"t_wait_ms\": %.3f, \"kernel_ms\": %.3f, \"gpu_ms\": %.3f, \"passes\": %u, "
         "\"checked\": %llu, \"expected\": %llu, \"n_bad\": %llu, \"seed\": %u, \"seed_prev\": %u, \"recs\": ",
         g_id.c_str(), t_wait, kernel_ms, (double)(long long)(h.t_end - h.t_start) / 1e6,
         g_mode == SM ? g_iters : g_passes, h.checked, words_expected(), g_mode == WRITE ? 0ull : h.n_bad, g_seed,
         seed_prev);
  if (g_mode == WRITE) printf("[]");
  else print_recs(h_out);
  printf(", \"verify_bad\": %llu, \"verify_recs\": ", vbad);
  print_recs(h_ver, g_mode == WRITE);
  printf(", \"restored\": %s, \"error\": null}\n", restored < 0 ? "null" : restored ? "true" : "false");
  fflush(stdout);
}

static int index_of(const char* const* names, int n, const std::string& s) {
  for (int k = 0; k < n; ++k)
    if (s == names[k]) return k;
  return -1;
}

int main(int argc, char** argv) {
  double mb = 256, ms = 100;
  int selftest = 5;
  for (int a = 1; a < argc; a += 2) {
    const std::string k = argv[a], v = a + 1 < argc ? argv[a + 1] : "";
    if (k == "--mb") mb = atof(v.c_str());
    else if (k == "--ms") ms = atof(v.c_str());
    else if (k == "--gap-ms") g_gap_ms = atof(v.c_str());
    else if (k == "--selftest") selftest = atoi(v.c_str());
    else if (k == "--pattern") g_pattern = index_of(PATTERNS, 4, v);
    else if (k == "--load") g_load = index_of(LOADS, 4, v);
    else if (k == "--mode") g_mode = index_of(MODES, 3, v);
    else g_pattern = -1;
    if (g_pattern < 0 || g_load < 0 || g_mode < 0 || (g_mode == WRITE && g_pattern != HASH) || v.empty() || mb <= 0 ||
        mb > 4096 || ms <= 0) {
      fprintf(stderr, "bad option %s %s\n", k.c_str(), v.c_str());
      return 2;
    }
  }

  CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));   // CPU asleep during the kernel: a quieter probe
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  const Check main_check = CHECKS[g_pattern][g_load];
  int per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, main_check, THREADS, 0));
  g_blocks = per_sm * prop.multiProcessorCount;
  const unsigned long long step = (unsigned long long)U * g_blocks * THREADS;   // vectors per grid step
  g_nvec = (unsigned)std::max(step, (unsigned long long)(mb * (1 << 20)) / 16 / step * step);
  const unsigned long long flush_bytes = std::max(32ull << 20, 16ull * prop.l2CacheSize);
  g_nflush = (unsigned)(flush_bytes / 16);
  CK(cudaMalloc(&g_buf, (size_t)g_nvec * 16));
  CK(cudaMalloc(&g_flush, flush_bytes));
  CK(cudaMalloc(&d_out, sizeof(Out)));
  g_warps = g_blocks * THREADS / 32;
  if (g_mode == WRITE) {
    h_ts.resize(g_nvec / step * g_warps);
    CK(cudaMalloc(&d_ts, h_ts.size() * sizeof h_ts[0]));
  }
  CK(cudaMemset(g_flush, 0, flush_bytes));
  CK(cudaEventCreate(&g_e0));
  CK(cudaEventCreate(&g_e1));
  refill();
  if (verify(h_ver)) {
    fprintf(stderr, "freshly filled buffer reads wrong (%llu words)\n", h_ver.h.n_bad);
    return 4;
  }

  // Passes per shot, timed over >= 5 ms: one pass of a buffer inside L2 takes ~3 us, less than
  // the launch itself (a single-pass calibration gave 11 ms kernels instead of 100).
  double pass_ms = 0;
  g_passes = 1;
  if (g_mode == READ) {
    unsigned cal = 1;
    while (run(main_check, cal, h_out) < 5 && cal < (1u << 24)) cal *= 4;
    std::vector<float> t1;
    for (int k = 0; k < 3; ++k) t1.push_back(run(main_check, cal, h_out));
    std::sort(t1.begin(), t1.end());
    pass_ms = t1[1] / cal;
    g_passes = (unsigned)std::max(1L, lround(ms / pass_ms));
  }

  // SM mode: its own grid (alu's occupancy), iterations per checkpoint timed over >= 5 ms, then the
  // fault-free reference run that every shot is compared with.
  if (g_mode == SM) {
    int per = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, alu, THREADS, 0));
    g_sm_blocks = per * prop.multiProcessorCount;
    g_nthr = g_sm_blocks * THREADS;
    h_ck.resize((size_t)N_CKPT * g_nthr * 2);
    h_ck_ts.resize((size_t)N_CKPT * (g_nthr / 32));
    h_smid.resize(g_sm_blocks);
    CK(cudaMalloc(&d_ck, h_ck.size() * sizeof h_ck[0]));
    CK(cudaMalloc(&d_ck_ts, h_ck_ts.size() * sizeof h_ck_ts[0]));
    CK(cudaMalloc(&d_smid, h_smid.size() * sizeof h_smid[0]));
    g_iters = 16;
    while (sm_shot() < 5 && g_iters < (1u << 24)) g_iters *= 4;
    std::vector<float> t1;
    for (int k = 0; k < 3; ++k) t1.push_back(sm_shot());
    std::sort(t1.begin(), t1.end());
    g_iters = (unsigned)std::max(1L, lround(ms / (t1[1] / g_iters)));
    pass_ms = sm_shot();
    g_gold = h_ck;
  }

  // Self-test: shots without any pulse must be clean, or nothing this program reports means anything.
  unsigned long long bad = 0;
  double ksum = 0;
  for (int s = 0; s < selftest; ++s) {
    if (g_mode == WRITE) {
      ksum += write_shot(0);                   // read back into h_ver
      bad += h_out.h.checked != 4ull * g_nvec;
    } else if (g_mode == SM) {
      ksum += sm_shot();
      bad += sm_compare() + (h_out.h.checked != words_expected());
      continue;                                // no buffer to verify
    } else {
      ksum += run(main_check, g_passes, h_out);
      bad += h_out.h.n_bad + (h_out.h.checked != 4ull * g_nvec * g_passes);
      verify(h_ver);
    }
    bad += h_ver.h.n_bad + (h_ver.h.checked != 4ull * g_nvec);
  }
  if (g_mode == WRITE) pass_ms = selftest ? ksum / selftest : 0;
  printf("{\"ready\": true, \"mode\": \"%s\", \"device\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"l2_bytes\": %d, \"blocks\": %d, "
         "\"threads\": %d, \"mb\": %.3f, \"words\": %llu, \"pattern\": \"%s\", \"load\": \"%s\", \"seed\": %u, "
         "\"k\": %u, \"pass_ms\": %.6f, \"passes\": %u, \"kernel_ms\": %.3f, \"gap_ms\": %.1f, \"selftest_shots\": %d, "
         "\"selftest_bad\": %llu}\n",
         MODES[g_mode], prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize, g_blocks, THREADS,
         g_nvec * 16.0 / (1 << 20), 4ull * g_nvec, PATTERNS[g_pattern], LOADS[g_load], SEED, K, pass_ms, g_passes,
         selftest ? ksum / selftest : 0.0, g_gap_ms, selftest, bad);
  fflush(stdout);

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd, id = "?";
    in >> cmd;
    if (cmd == "quit") break;
    if (cmd == "go") {
      double idle = 0;
      in >> id >> idle;
      g_id = id;
      shot(idle);
    } else if (cmd == "poke") {
      unsigned long long w = ~0ull;
      unsigned v = 0, was = 0;
      in >> w >> v;
      if (w < 4ull * g_nvec) {
        unsigned* p = reinterpret_cast<unsigned*>(g_buf) + w;
        CK(cudaMemcpy(&was, p, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(p, &v, 4, cudaMemcpyHostToDevice));
        evict();
      }
      printf("{\"poke\": %llu, \"was\": %u, \"value\": %u, \"ok\": %s}\n", w, was, v,
             w < 4ull * g_nvec ? "true" : "false");
      fflush(stdout);
    } else if (cmd == "skip") {
      unsigned long long w = ~0ull;
      in >> w;
      const bool ok = g_mode == WRITE && w < 4ull * g_nvec;
      if (ok) g_skip = (unsigned)(w / 4);
      printf("{\"skip\": %llu, \"ok\": %s}\n", w, ok ? "true" : "false");
      fflush(stdout);
    }
  }
  return 0;
}
