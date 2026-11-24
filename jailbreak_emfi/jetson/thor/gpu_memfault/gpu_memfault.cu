// GPU memfault with CORRECT physical addresses.
//
// Uses cuMemHostAlloc(DEVICEMAP) memory: a GPU kernel torture-reads/verifies it
// (so GPU-path faults during the glitch are caught), while the TRUE DRAM PA of
// each page comes from /proc/self/pagemap (0..128 GiB) -- not the 0xff.. GPU
// aperture the RM page table reports. Channel = validated full-range masks.
//
// Acts as the collection TCP SERVER; the EMFI controller connects as client and
// sends attempt_index per probe position (same protocol as the Python memfault).
// Root required (pagemap PFNs).
// usage: gpu_memfault --host H --port P --log PATH [--gib G] [--window S] [--free-run GAP_S]
//                     [--pattern const|index] [--mem host|device] [--bar0 RESOURCE_FILE | --no-ecc]
// (the service starts it through jetson/thor/gpu_memfault/launch.py with the values of [memfault] in
// config/thor.toml; --host/--port/--log have no defaults)
// --free-run: no controller; repeat the verify window every GAP_S idle seconds
// (attempt_index -1, collection_id "freerun"), e.g. to set up the scope trigger.
// --pattern index: word i = (i*PM)^CONSTV instead of CONSTV everywhere, so a read served from
// another word of the buffer decodes to its source index: extra columns src_word (decoded,
// -1 if not a buffer word), src_pa, src_channel.
// --mem device: buffer in cuMemAlloc device memory (large GPU pages) instead of host-pinned 4 KiB
// pages; no pagemap PA there, so va = GPU VA and pa/channel/bank/src_pa = NA/-1.
// --no-ecc: don't poll the GPU ECC counters. By default every window snapshots the gb10b ECC
// status/corrected/uncorrected registers of all units (BAR0 = --bar0, the GPU's PCI resource0 file)
// before and after, and logs each unit that changed to <log>_ecc.tsv. The register offsets below are
// facts of the gb10b GPU, not settings.
#include <cuda.h>
#include <sys/mman.h>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unordered_map>
#include <string>

#define CKU(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*n=0;cuGetErrorName(r,&n);fprintf(stderr,"%s -> %s @%d\n",#x,n?n:"?",__LINE__);exit(1);} }while(0)
#define CKR(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"%s -> %s @%d\n",#x,cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

// Thor's integrated GPU on PCI 0000:01:00.0: BAR0 register window (hardware fact; --bar0 overrides).
static const char* DEFAULT_BAR0="/sys/bus/pci/devices/0000:01:00.0/resource0";
static const uint64_t CONSTV = 0x5555555555555555ULL;
static const uint64_t PM = 0x9E3779B97F4A7C15ULL;   // odd: i -> i*PM is invertible mod 2^64
__host__ __device__ static inline uint64_t pat(bool index, uint64_t i){ return index ? (i*PM)^CONSTV : CONSTV; }
// DRAM hash = live MC McEmemAdrCfg{Channel,Bank}Mask registers, applied to the DRAM
// offset PA-0x80000000 (verified against the live MC registers; same masks as the analysis' Thor DRAM map)
static const uint64_t DRAM_BASE=0x80000000ULL;
static const uint64_t CH[4]={0x68b62e3800ULL,0x4b5a527e00ULL,0x7fa9a6cc00ULL,0xe694cd5a00ULL};
static const uint64_t BK[4]={0x023331c9ULL,0x04bc922dULL,0x057bd5beULL,0x07ad6ad5ULL};
static inline int par(uint64_t x){return __builtin_parityll(x);}
static int channel_of(uint64_t pa){uint64_t o=pa-DRAM_BASE;int c=0;for(int i=0;i<4;i++)c|=par(o&CH[i])<<i;return c;}
static int bank_of(uint64_t pa){uint64_t b=(pa-DRAM_BASE)>>13;int k=0;for(int i=0;i<4;i++)k|=par(b&BK[i])<<i;return k;}

static int pmfd=-1;
static uint64_t va2pa(uint64_t va){uint64_t e; if(pread(pmfd,&e,8,(va/4096)*8)!=8)return 0; if(!((e>>63)&1))return 0; return (e&((1ULL<<55)-1))*4096+(va%4096);}

// ---- GPU ECC counters via BAR0 (gb10b register layout) ----
struct EccUnit{ std::string name; uint32_t st,c,u; };   // c/u = 0: no such counter
static volatile uint32_t* bar0=nullptr;
static inline uint32_t rd(uint32_t o){ return o?bar0[o/4]:0; }
static inline bool badf(uint32_t v){ return (v>>20)==0xbad; }
static std::vector<EccUnit> ecc_units(){
    std::vector<EccUnit> u; char n[64];
    for(int l=0;l<4;l++)for(int s=0;s<4;s++){ uint32_t b=0x140000+l*0x2000+s*0x200;
        snprintf(n,sizeof n,"l2_ltc%d_lts%d",l,s); u.push_back({n,b+0x4f0,b+0x4f4,b+0x4f8}); }
    const struct{const char*n;uint32_t st,c,u;} TPC[]={{"sm_lrf",0x358,0x35c,0x360},{"sm_rams",0x388,0x38c,0x390},
        {"sm_l1_tag",0x624,0x628,0x62c},{"sm_cbu",0x638,0x63c,0x640},{"sm_icache",0x64c,0x650,0x654}};
    const struct{const char*n;uint32_t st,c,u;} GPC[]={{"gcc_l15",0x1048,0x104c,0x1054},
        {"mmu_l1tlb",0x1414,0x1418,0x141c},{"gpccs_falcon",0x2678,0x267c,0x2680}};
    int ngpc=rd(0x22430);
    for(int g=0;g<ngpc&&g<8;g++){ uint32_t go=0x500000+g*0x10000;
        for(auto&x:GPC){ snprintf(n,sizeof n,"gpc%d_%s",g,x.n); u.push_back({n,go+x.st,go+x.c,go+x.u}); }
        for(int t=0;t<4;t++){ uint32_t to=go+0x5000+t*0x1000;
            if(badf(rd(to+0x358)))continue;            // TPC not present (logical numbering)
            for(auto&x:TPC){ snprintf(n,sizeof n,"gpc%d_tpc%d_%s",g,t,x.n); u.push_back({n,to+x.st,to+x.c,to+x.u}); }
            // error capture, status only: TPC exception, per SM (stride 0x80) warp/global error status
            snprintf(n,sizeof n,"gpc%d_tpc%d_tpc_exception",g,t); u.push_back({n,to+0x508,0,0});
            for(int s=0;s<2;s++){
                snprintf(n,sizeof n,"gpc%d_tpc%d_sm%d_warp_esr",g,t,s); u.push_back({n,to+0x730+s*0x80,0,0});
                snprintf(n,sizeof n,"gpc%d_tpc%d_sm%d_global_esr",g,t,s); u.push_back({n,to+0x734+s*0x80,0,0}); } }
        snprintf(n,sizeof n,"gpc%d_gpc_exception",g); u.push_back({n,go+0x2c90,0,0}); }
    const EccUnit HUB[]={{"mmu_l2tlb",0x1fa570,0x1fa574,0x1fa578},{"mmu_hubtlb",0x1fa584,0x1fa588,0x1fa58c},
        {"mmu_fillunit",0x1fa598,0x1fa59c,0x1fa5a0},{"fecs_falcon",0x409678,0x40967c,0x409680},
        {"mme",0x404514,0,0x40451c},{"mme_fe1",0x40a814,0,0x40a81c},{"pmu_falcon",0x8f66b0,0x8f66b8,0x8f66bc},
        {"xal_prireorder",0x10fca4,0x10fca8,0x10fcac},{"xbar_gpc0_gnic0",0xb2004c,0,0xb20054},{"lrc",0x8c0584,0,0x8c0590},
        // not ECC units; status = the register, logged when it changes
        {"ltc_ltcs_intr",0x1c620c,0,0},{"mmu_fault_status",0x1fa560,0,0},{"mmu_fault_info",0x1fa55c,0,0},
        {"gr_intr",0x400100,0,0},{"gr_exception",0x400108,0,0},{"gr_exception1",0x400118,0,0},
        {"gr_trapped_addr",0x400704,0,0},{"fe_hww_esr",0x404000,0,0},{"fecs_host_int_status",0x409c18,0,0},
        {"priv_error_adr",0x122120,0,0},{"priv_error_wrdat",0x122124,0,0},{"priv_error_info",0x122128,0,0},
        {"priv_error_code",0x12212c,0,0},{"ringmaster_intr0",0x120058,0,0},{"ringmaster_intr1",0x12005c,0,0}};
    for(auto&x:HUB) u.push_back(x);
    return u;
}
// GR busy state sampled while the verify kernels run: gr_status, gr_engine_status, gr_activity_0..4
static const uint32_t ACT[7]={0x400700,0x40060c,0x400380,0x400384,0x400388,0x40038c,0x400390};
struct ActAcc{ long samples=0,idle=0; uint32_t orv[7]={}; uint64_t t0=0,t1=0; };
static inline uint64_t ptimer(){ uint32_t hi,lo; do{ hi=rd(0x9410); lo=rd(0x9400); }while(hi!=rd(0x9410)); return (uint64_t)hi<<32|lo; }
static inline void act_sample(ActAcc&a){
    uint32_t s=rd(ACT[0]); a.samples++; if(!s)a.idle++; if(!badf(s))a.orv[0]|=s;
    for(int i=1;i<7;i++){ uint32_t v=rd(ACT[i]); if(!badf(v))a.orv[i]|=v; }   // 0xbadf: gated, skip
}
// Channel RAM: one word per channel per runlist (bit 5 = eng_faulted). Runlist blocks at 0xd00000 +
// rl*0x400; their channel_config (+0x4) = chram BAR0 offset | log2(#channels).
struct Chram{ int rl; uint32_t base; int n; };
static std::vector<Chram> chram_find(){
    std::vector<Chram> c;
    for(int rl=0;rl<16;rl++){ uint32_t v=rd(0xd00000+rl*0x400+4);
        if(!v||badf(v))continue;
        uint32_t b=(v>>4)<<4; int n=1<<(v&0xf); if(n>4096||b+4*n>(1u<<24))continue;
        bool dup=false; for(auto&x:c) dup|=x.base==b; if(!dup) c.push_back({rl,b,n}); }
    return c;
}
// faulted channels: key rl<<16|ch -> chram word
static void chram_faulted(const std::vector<Chram>&c,std::unordered_map<uint32_t,uint32_t>&f){
    f.clear();
    for(auto&x:c) for(int ch=0;ch<x.n;ch++){ uint32_t v=rd(x.base+4*ch); if(!badf(v)&&(v>>5&1)) f[(uint32_t)x.rl<<16|ch]=v; }
}
static void ecc_snap(const std::vector<EccUnit>&u,std::vector<uint32_t>&v){
    v.resize(u.size()*3); for(size_t i=0;i<u.size();i++){ v[3*i]=rd(u[i].st); v[3*i+1]=rd(u[i].c); v[3*i+2]=rd(u[i].u); }
}

// ---- GPU verify: read every word, record mismatches (idx+val) ----
__global__ void verify(const unsigned long long* buf, unsigned long long n,
                       bool index, unsigned* mcount,
                       unsigned long long* midx, unsigned long long* mval, unsigned cap){
    unsigned long long t=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
    for(unsigned long long i=t;i<n;i+=stride){
        unsigned long long v=buf[i];
        if(v!=pat(index,i)){unsigned k=atomicAdd(mcount,1u); if(k<cap){midx[k]=i; mval[k]=v;}}
    }
}

__global__ void fill(unsigned long long* buf, unsigned long long n, bool index){
    unsigned long long t=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
    for(unsigned long long i=t;i<n;i+=stride) buf[i]=pat(index,i);
}

// ---- minimal collect handshake (server) ----
static const unsigned char COLLECT[4]={0x1a,0x0d,0x00,0x00}; // 3354 LE
static bool recv_exact(int fd,void*p,size_t n){unsigned char*b=(unsigned char*)p;size_t g=0;while(g<n){ssize_t r=recv(fd,b+g,n-g,0);if(r<=0)return false;g+=r;}return true;}
// returns attempt_index (and hex uuid) or -1 on disconnect
static long collect_handshake(int fd,char uuid_hex[33]){
    if(send(fd,COLLECT,4,0)!=4)return -1;
    unsigned char ack[4];
    do{ if(!recv_exact(fd,ack,4))return -1; }while(memcmp(ack,COLLECT,4)!=0);
    unsigned char att[8],uu[16];
    if(!recv_exact(fd,att,8))return -1;
    if(!recv_exact(fd,uu,16))return -1;
    uint64_t a=0; for(int i=0;i<8;i++)a|=(uint64_t)att[i]<<(8*i);
    for(int i=0;i<16;i++)sprintf(uuid_hex+2*i,"%02x",uu[i]); uuid_hex[32]=0;
    // drain optional prompt fields (non-blocking)
    unsigned char junk[64]; int fl=fcntl(fd,F_GETFL,0); fcntl(fd,F_SETFL,fl|O_NONBLOCK);
    while(recv(fd,junk,sizeof junk,0)>0){} fcntl(fd,F_SETFL,fl);
    return (long)a;
}

int main(int argc,char**argv){
    double gib=16; const char*host=nullptr; int port=-1; double window=6.0;
    const char*logpath=nullptr; const char*bar0path=DEFAULT_BAR0; double freerun=-1; bool index=false, device=false, ecc=true;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--gib"))gib=atof(argv[++i]);
        else if(!strcmp(argv[i],"--host"))host=argv[++i];
        else if(!strcmp(argv[i],"--port"))port=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--window"))window=atof(argv[++i]);
        else if(!strcmp(argv[i],"--log"))logpath=argv[++i];
        else if(!strcmp(argv[i],"--free-run"))freerun=atof(argv[++i]);
        else if(!strcmp(argv[i],"--pattern"))index=!strcmp(argv[++i],"index");
        else if(!strcmp(argv[i],"--mem"))device=!strcmp(argv[++i],"device");
        else if(!strcmp(argv[i],"--no-ecc"))ecc=false;
        else if(!strcmp(argv[i],"--bar0"))bar0path=argv[++i];
        else{ fprintf(stderr,"unknown argument %s\n",argv[i]); return 2; }
    }
    if(!logpath||(freerun<0&&(!host||port<=0))){
        fprintf(stderr,"usage: gpu_memfault --host H --port P --log PATH [--gib G] [--window S] [--free-run GAP_S]\n"
                       "       [--pattern const|index] [--mem host|device] [--bar0 RESOURCE_FILE | --no-ecc]\n");
        return 2;
    }
    pmfd=open("/proc/self/pagemap",O_RDONLY);
    size_t nbytes=((size_t)(gib*(1ULL<<30)))&~63ULL; size_t nwords=nbytes/8;
    CKU(cuInit(0)); CUdevice dev; CKU(cuDeviceGet(&dev,0));
    CUcontext ctx; CKU(cuDevicePrimaryCtxRetain(&ctx,dev)); CKU(cuCtxSetCurrent(ctx));
    void*hp=nullptr; CUdeviceptr dp=0;
    if(device){
        CKU(cuMemAlloc(&dp,nbytes));
        fprintf(stderr,"alloc %.1f GiB device memory; GPU VA=0x%llx\n",nbytes/1073741824.0,(unsigned long long)dp);
    }else{
        CKU(cuMemHostAlloc(&hp,nbytes,CU_MEMHOSTALLOC_DEVICEMAP));
        CKU(cuMemHostGetDevicePointer(&dp,hp,0));
        fprintf(stderr,"alloc %.1f GiB host-pinned; CPU VA=%p base PA=0x%lx\n",
                nbytes/1073741824.0,hp,va2pa((uint64_t)hp));
    }
    if(index){ fill<<<4096,256>>>((unsigned long long*)dp,nwords,true); CKR(cudaDeviceSynchronize()); }
    else if(device) CKU(cuMemsetD32(dp,0x55555555u,nbytes/4));
    else memset(hp,0x55,nbytes); // pattern 0x5555...
    uint64_t inv=PM; for(int i=0;i<6;i++) inv*=2-PM*inv;   // PM^-1 mod 2^64
    fprintf(stderr,"pattern: %s\n",index?"index (i*PM)^0x55..":"const 0x55..");
    const unsigned CAP=1u<<20;
    unsigned *d_cnt; unsigned long long *d_idx,*d_val; CKR(cudaMalloc(&d_cnt,4));
    CKR(cudaMalloc(&d_idx,(size_t)CAP*8)); CKR(cudaMalloc(&d_val,(size_t)CAP*8));
    unsigned long long *h_idx=(unsigned long long*)malloc((size_t)CAP*8),*h_val=(unsigned long long*)malloc((size_t)CAP*8);
    unsigned long long *buf=(unsigned long long*)hp;

    // mkdir -p logdir
    { std::string d=logpath; auto s=d.find_last_of('/'); if(s!=std::string::npos){std::string cmd="mkdir -p '"+d.substr(0,s)+"'"; if(system(cmd.c_str())){} } }
    FILE*log=fopen(logpath,"a");
    fseek(log,0,SEEK_END);
    if(ftell(log)==0) fprintf(log,"timestamp\tattempt_index\tcollection_id\tword_index\tva\tpa\tchannel\tbank\texpected\tactual\txor%s\n",
                              index?"\tsrc_word\tsrc_pa\tsrc_channel":"");
    fflush(log);

    // ECC polling (after cuInit, so the GPU is powered when BAR0 is read)
    std::vector<EccUnit> eunits; std::vector<uint32_t> e0,e1; FILE*elog=nullptr,*wlog=nullptr;
    std::vector<Chram> chrams; std::unordered_map<uint32_t,uint32_t> cf0,cf1;
    if(ecc){
        int fd=open(bar0path,O_RDONLY|O_SYNC);
        void*m=fd<0?MAP_FAILED:mmap(nullptr,1<<24,PROT_READ,MAP_SHARED,fd,0);
        if(m==MAP_FAILED){ fprintf(stderr,"ECC polling off: cannot map GPU BAR0\n"); ecc=false; }
        else{
            bar0=(volatile uint32_t*)m; eunits=ecc_units(); chrams=chram_find();
            { size_t nch=0; for(auto&x:chrams) nch+=x.n; fprintf(stderr,"channel RAM: %zu runlists, %zu channels\n",chrams.size(),nch); }
            std::string ep=logpath; auto d=ep.rfind(".tsv"); ep=(d==std::string::npos?ep:ep.substr(0,d))+"_ecc.tsv";
            elog=fopen(ep.c_str(),"a"); fseek(elog,0,SEEK_END);
            if(ftell(elog)==0) fprintf(elog,"timestamp\tattempt_index\tcollection_id\tunit\tstatus_before\tstatus_after\t"
                                       "corrected_before\tcorrected_after\tuncorrected_before\tuncorrected_after\n");
            fflush(elog);
            fprintf(stderr,"ECC polling: %zu units (boot0 0x%x) -> %s\n",eunits.size(),bar0[0],ep.c_str());
            ep=ep.substr(0,ep.size()-8)+"_win.tsv";         // <log>_win.tsv: one row per window
            wlog=fopen(ep.c_str(),"a"); fseek(wlog,0,SEEK_END);
            if(ftell(wlog)==0) fprintf(wlog,"timestamp\tattempt_index\tcollection_id\tlaunches\tsamples\tidle_samples\t"
                                       "gr_status_or\tgr_engine_status_or\tgr_activity0_or\tgr_activity1_or\tgr_activity2_or\t"
                                       "gr_activity3_or\tgr_activity4_or\tptimer_start_ns\tptimer_end_ns\n");
            fflush(wlog);
        }
    }

    int threads=256, blocks=4096; long attempts=0;
    // one window: active GPU access, accumulating mismatches; then log true PA + channel, repair
    auto window_once=[&](long att,const char*uuid){
        // active GPU access during the glitch window
        CKR(cudaMemset(d_cnt,0,4));
        ActAcc act; long launches=0;
        if(ecc){ ecc_snap(eunits,e0); chram_faulted(chrams,cf0); act.t0=ptimer(); }
        double t_end=(double)clock()/CLOCKS_PER_SEC; // fallback; use wall time below
        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        do{
            verify<<<blocks,threads>>>((unsigned long long*)dp,nwords,index,d_cnt,d_idx,d_val,CAP);
            launches++;
            if(ecc){ cudaError_t q; while((q=cudaStreamQuery(0))==cudaErrorNotReady) act_sample(act); CKR(q); }
            else CKR(cudaDeviceSynchronize());
            clock_gettime(CLOCK_MONOTONIC,&t1);
        }while((t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9 < window);
        if(ecc) act.t1=ptimer();
        unsigned cnt=0; CKR(cudaMemcpy(&cnt,d_cnt,4,cudaMemcpyDeviceToHost));
        unsigned got=cnt<CAP?cnt:CAP;
        if(got){CKR(cudaMemcpy(h_idx,d_idx,(size_t)got*8,cudaMemcpyDeviceToHost));
                CKR(cudaMemcpy(h_val,d_val,(size_t)got*8,cudaMemcpyDeviceToHost));}
        // dedup by word index, log true PA + channel, repair
        char ts[40]; time_t tt=time(0); strftime(ts,sizeof ts,"%Y-%m-%dT%H:%M:%S%z",localtime(&tt));
        std::unordered_map<unsigned long long,unsigned long long> seen;
        for(unsigned k=0;k<got;k++) seen[h_idx[k]]=h_val[k];
        for(auto&kv:seen){
            unsigned long long wi=kv.first, actual=kv.second;
            uint64_t va=(device?(uint64_t)dp:(uint64_t)hp)+wi*8; uint64_t pa=device?0:va2pa(va);
            int ch=pa?channel_of(pa):-1, bk=pa?bank_of(pa):-1;
            uint64_t exp=pat(index,wi);
            char pas[24]; if(pa) snprintf(pas,sizeof pas,"0x%lx",pa); else strcpy(pas,"NA");
            fprintf(log,"%s\t%ld\t%s\t%llu\t0x%lx\t%s\t%d\t%d\t0x%llx\t0x%llx\t0x%llx",
                    ts,att,uuid,wi,va,pas,ch,bk,
                    (unsigned long long)exp,actual,actual^exp);
            if(index){
                uint64_t src=(actual^CONSTV)*inv;               // word the value belongs to
                uint64_t spa=src<nwords&&!device?va2pa((uint64_t)hp+src*8):0;
                if(src<nwords){ char sps[24]; if(spa) snprintf(sps,sizeof sps,"0x%lx",spa); else strcpy(sps,"NA");
                                fprintf(log,"\t%llu\t%s\t%d",(unsigned long long)src,sps,spa?channel_of(spa):-1); }
                else fprintf(log,"\t-1\tNA\t-1");
            }
            fputc('\n',log);
            if(device) CKU(cuMemcpyHtoD(dp+wi*8,&exp,8)); else buf[wi]=exp; // repair
        }
        fflush(log);
        int echanged=0;
        if(ecc){
            ecc_snap(eunits,e1);
            for(size_t i=0;i<eunits.size();i++){
                const uint32_t*a=&e0[3*i],*b=&e1[3*i];
                if(a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2])continue;
                // a gated unit reads 0xbadf...; a change only from/to that is not an event
                if(a[1]==b[1]&&a[2]==b[2]&&(badf(a[0])||badf(b[0]))&&(badf(a[0])?b[0]:a[0])==0)continue;
                fprintf(elog,"%s\t%ld\t%s\t%s\t0x%x\t0x%x\t%u\t%u\t%u\t%u\n",ts,att,uuid,eunits[i].name.c_str(),
                        a[0],b[0],a[1],b[1],a[2],b[2]);
                echanged++;
            }
            // channels whose engine-faulted bit appeared (or whose word changed while faulted)
            chram_faulted(chrams,cf1);
            for(auto&kv:cf1){ auto it=cf0.find(kv.first); if(it!=cf0.end()&&it->second==kv.second)continue;
                fprintf(elog,"%s\t%ld\t%s\tchram_rl%u_ch%u_eng_faulted\t0x%x\t0x%x\t0\t0\t0\t0\n",ts,att,uuid,
                        kv.first>>16,kv.first&0xffff,it==cf0.end()?0u:it->second,kv.second);
                echanged++; }
            fflush(elog);
            fprintf(wlog,"%s\t%ld\t%s\t%ld\t%ld\t%ld",ts,att,uuid,launches,act.samples,act.idle);
            for(int i=0;i<7;i++) fprintf(wlog,"\t0x%x",act.orv[i]);
            fprintf(wlog,"\t%llu\t%llu\n",(unsigned long long)act.t0,(unsigned long long)act.t1);
            fflush(wlog);
        }
        attempts++;
        fprintf(stderr,"attempt %ld: %u raw mismatches, %zu distinct words, %d ECC units changed (total %ld)\n",
                att,cnt,seen.size(),echanged,attempts);
    };
    if(freerun>=0){
        fprintf(stderr,"free-run: %.1f s window every %.1f s idle\n",window,freerun);
        while(true){ window_once(-1,"freerun"); usleep((useconds_t)(freerun*1e6)); }
    }

    // TCP server
    int srv=socket(AF_INET,SOCK_STREAM,0); int one=1; setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,&one,4);
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(port); inet_pton(AF_INET,host,&sa.sin_addr);
    if(bind(srv,(sockaddr*)&sa,sizeof sa)){perror("bind");return 1;} listen(srv,1);
    fprintf(stderr,"serving on %s:%d (GPU memfault, true-PA)\n",host,port);
    while(true){
        fprintf(stderr,"waiting for controller...\n");
        sockaddr_in ca; socklen_t cl=sizeof ca; int c=accept(srv,(sockaddr*)&ca,&cl);
        if(c<0)continue; setsockopt(c,IPPROTO_TCP,TCP_NODELAY,&one,4);
        fprintf(stderr,"controller connected\n");
        char uuid[33];
        while(true){
            long att=collect_handshake(c,uuid);
            if(att<0){fprintf(stderr,"controller disconnected; re-waiting\n");break;}
            window_once(att,uuid);
        }
        close(c);
    }
    return 0;
}
