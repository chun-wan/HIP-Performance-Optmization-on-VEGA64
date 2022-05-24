Contents

[1前言...  2](#_Toc15369680)

[1.1 HIP基礎...  2](#_Toc15369681)

[1.2預習資料...  3](#_Toc15369682)

[1.3匹配硬件...  3](#_Toc15369683)

[2基於Vega10嘅硬件相關優化實例...  3](#_Toc15369684)

[2.1  塊與線程： Blocks &Threads. 3](#_Toc15369685)

[2.1.1最高綫程速率...  3](#_Toc15369686)

[2.1.2 1D形狀Block嘅線程速率曲線...  6](#_Toc15369687)

[2.1.3 2D形狀Block線程速率...  8](#_Toc15369688)

[2.1.3 3D形狀Block嘅線程生成速率...  9](#_Toc15369689)

[2.2 Compute Resources計算資源...  10](#_Toc15369690)

[2.2.1 Execute 1，000，000 of FMA：簡單循環100万次...  10](#_Toc15369691)

[2.2.2 Specified Loop Unroll：指定循環展開大小...  13](#_Toc15369692)

[2.2.3 Double Loop：雙層循環... 13](#_Toc15369693)

[2.2.4 Increasing Threads In Parallel ：增加並行線程... 14](#_Toc15369694)

[2.2.5 Enough Parallel Threads：足夠多線程充滿64个計算單位...  14](#_Toc15369695)

[2.3 VGPR：矢量通用寄存器...  16](#_Toc15369696)

[2.4 SGPR：標量通用寄存器...  17](#_Toc15369697)

[2.5 Divergence：Wave分歧...  20](#_Toc15369698)

[2.6 Memory Read Latency：顯存讀寫延遲...  22](#_Toc15369699)

[2.6.1 L2 Cache Miss：直接由顯存讀寫...  22](#_Toc15369700)

[2.6.2 CacheLine Length：緩存行長度...  23](#_Toc15369701)

[2.6.3 L1/L2 Cacheline Hit Latency：一/二級緩存命中延時...  24](#_Toc15369702)

[2.7 Alternative Method to measure CacheLine Size：另一組測試Cacheline長度...  25](#_Toc15369703)

[2.7.1測試CacheLine大小...  25](#_Toc15369704)

[2.7.2 Divergence for Memory Read/Write：顯存訪問分歧...  25](#_Toc15369705)

[2.8 NCHW-4D Index Generation：4D數組索引生成...  25](#_Toc15369706)

[2.9 Local Data Share：本地數據共享...  26](#_Toc15369707)

[2.9.1 LDS Latency. 26](#_Toc15369708)

[2.9.2 LDS bank Conflicts. 27](#_Toc15369709) 

[2.10 Memory Channel Conflicts：存儲通道衝突...  28](#_Toc15369710)

[2.11 Math Functions：數學函數...  29](#_Toc15369711)

[2.12 Reduction：歸約...  29](#_Toc15369712)

[2.13  Padding Before Convolution. 31](#_Toc15369713)

[2.13.1 1st Padding Kernel 31](#_Toc15369714)

[2.13.2 Optimize Kernel to Remove Scratch Memory. 33](#_Toc15369715) 

[2.14 BatchNorm..  33](#_Toc15369716)

[3其他...  34](#_Toc15369717)

1     前言
========

1.1 HIP基礎
---------

請參考HIP官方發佈。  [https://github.com/ROCm-Developer-Tools/HIP](https://github.com/ROCm-Developer-Tools/HIP)

HIP允許並行程序開發者無縫移植CUDA C++代碼。 HIP源代碼（包括由CUDA移植嘅HIP代碼）可以被CUDA編譯執行喺NVIDIA GPU抑或被HIPCC 編譯執行喺AMD GPU上。 HIP包括以下關鍵特性：

* HIP是一個輕量級的，它幾乎不會對CUDA（或 hcc “HC”）代碼造成性能影響，
* HIP允許使用C++程序設計語言版本的多種特性編程，例如模板，C++11 Lambdas表達式，類，名字空間等。 
* HIP允許開發者使用基於目標平台嘅最佳開發環境和工具鏈。 
*   “hipify”工具能夠自動把CUDA源代碼移植到HIP.
*開發者可以指定平台（ CUDA或hcc ）進行性能調試抑或處理棘手問題。 

1.2 預習資料
--------

在閱讀第二章前，請確定已完成對以下材料的學習。

*   [HIP Kernel Language](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md)
*   [HIP Runtime API (Doxygen)](http://rocm-developer-tools.github.io/HIP)
*   [HIP Porting Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md)
*   [HIP Porting Driver Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_driver_api.md)
*   [HIP Programming Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_programming_guide.md)
*   Samples: [https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples)
*   Examples: [https://github.com/ROCm-Developer-Tools/HIP-Examples](https://github.com/ROCm-Developer-Tools/HIP-Examples)

1.3 匹配硬件
--------

本講座中所有測試均基於AMD Radeon MI25抑或硬件。 如果改為其他硬件，需要修改計算核心的頻率，Mi25對應的核心頻率為1.536 Ghz。 

2       基於Vega10嘅硬件相關優化實例
=========================

2.1  蚊與線程： Blocks &Threads
---------------------------

### 2.1.1 最高線程速率

AMD GCN硬件約定64 Threads 一個 wave，一個block可以有1-16個wave。 硬件生成Threads嘅速率將直接影響最終程序嘅效率， 例如GPU顯存嘅讀寫速度。  為咗測試Vega10嘅Threads 速率， 我哋可以寫一個最簡單嘅設備空函數，
```
__global__ void

null_kernel(hipLaunchParm lp,

       float* __restrict__ a)

{

}
```
執行rocm-smi，獲得MI25嘅額定頻率設置為1.536GHz。 
<table><tr><td bgcolor=“#707070”>

========================        ROCm System Management Interface        ========================

================================================================================================

GPU   Temp   AvgPwr   SCLK    MCLK    PCLK           Fan     Perf    PwrCap   SCLK OD   MCLK OD  GPU%

0     69.0c  19.0W    1536Mhz 945Mhz  8.0GT/s, x16   12.94%  manual  220.0W   0%        0%       0%

================================================================================================

========================               End of ROCm SMI Log              ========================
</td></tr></table>
因此程序設置總嘅Threads 數量為 1024*1204*1024， 已獲得接近秒級嘅GPU執行時間。 

Threads 速率是否與Block速率相關？ 仍然係一個謎。 因此測試程序暫時把每個 Block嘅Threads設置為最大值 1024。 

為咗獲得準備嘅時間，使用hipEventCreate函數產生兩個事件start，stop，透過hipEventRecord記錄兩個事件，並調用hipEventSynchronize 確保stop係同步事件並被正確執行，hipEventElapsedTime （ &eventMs，start，stop ）函數將獲得start，stop 兩個event嘅時間長度，單位係毫秒。 代碼如下：
```
  hipEvent_t start, stop; 

  hipEventCreate(&start); 

  hipEventCreate(&stop); 

  hipEventRecord(start, NULL); 

  hipLaunchKernel(null_kernel,

                               dim3(1024*1024, 1),

                               dim3(1024, 1, 1), 

                              0, 0,

                               deviceA); 

  hipEventRecord(stop, NULL); 

  hipEventSynchronize(stop); 

  hipEventElapsedTime(&eventMs, start, stop); 
```
完整的代碼如下：
```
example-1a.cpp

#include <assert.h>

#include <stdio.h>

#include <algorithm>

#include <stdlib.h>

#include<iostream>

#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define TOTAL_THREADS  (1024*1024*1024)

#define NUM  1

#define THREADS_PER_BLOCK_X  1024

#define THREADS_PER_BLOCK_Y  1

#define THREADS_PER_BLOCK_Z  1

__global__ void

null_kernel(hipLaunchParm lp,

       float* __restrict__ a)

{

}

using namespace std; 

int main() {

  float* hostA; 

  float* deviceA; 

  hipDeviceProp_t devProp; 

  hipGetDeviceProperties(&devProp, 0); 

  cout <<“System minor” << devProp.minor << endl; 

  cout <<“System major” << devProp.major << endl; 

  cout <<“agent prop name”<< devProp.name << endl; 

  cout <<óhip Device prop succeeded≤<< endl ; 

  hipEvent_t start, stop; 

  hipEventCreate(&start); 

  hipEventCreate(&stop); 

float eventMs = 1. 0f; 

  int i; 

  int errors; 

  hostA = (float*)malloc(NUM * sizeof(float)); 

  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float))); 

  hipLaunchKernel(null_kernel,

                  dim3(1, 1),

                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z),

                     0, 0,

                  deviceA); 

  hipEventRecord(start, NULL); 

  hipLaunchKernel(null_kernel,

                               dim3(TOTAL_THREADS/THREADS_PER_BLOCK_X, 1),

                               dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z),     

                     0, 0,

                               deviceA); 

  hipEventRecord(stop, NULL); 

  hipEventSynchronize(stop); 

  hipEventElapsedTime(&eventMs, start, stop); 

  printf("kernel_time (hipEventElapsedTime) =%6.3fms\\n", eventMs); 

  printf("Threads_per_cycle for Vega10 - 1.536GHz = % 3d\\n", int(TOTAL_THREADS / eventMs / 1.536 / 1e6)); 

  HIP_ASSERT(hipFree(deviceA)); 

  free(hostA); 

  return errors; 

}
```
使用如下指令編譯  example-1a.cpp

* <table ><tr><td bgcolor=“#707070”> hipcc example-1a .cpp-o example-1a .exe </td ></tr></table > 

本人假定隨後章節採用相同嘅方法進行編譯。

執行example-1a.exe，得到如下結果：
<table><tr><td bgcolor=“#707070”>
 System minor 0

 System major 3

 agent prop name Device 687f

hip Device prop succeeded

kernel_time (hipEventElapsedTime) =10.890ms

Threads_per_cycle for Vega10 - 1.536GHz =  64
</td></tr></table>
結果說明Mi25獲得64 threads/Cycle的極限性能。 

### 2.1.2 1D形狀 Block嘅綫程速率曲線

     頭炮簡單測試獲得MI25嘅線程速率為 64 threads/cycle，咁係咪所有1D 形狀塊均可獲得極限速率呢？ 

      Example2.cpp 將測試 自小而大不同的BlockDim，Dim3（1，1，1），Dim3（2，1，1），Dim3（4，1，1），Dim3（8，1，1），...，（1024，1，1）。 獲得如下結果：
<table><tr><td bgcolor=“#707070”>
 System minor 0

 System major 3

 agent prop name Device 687f

hip Device prop succeeded

kernel_time (hipEventElapsedTime) =2789.162ms

threads_per_block = 1,Threads_per_cycle for Vega10 - 1.536GHz =   0

kernel_time (hipEventElapsedTime) =1395.156ms

threads_per_block = 2,Threads_per_cycle for Vega10 - 1.536GHz =   1

kernel_time (hipEventElapsedTime) =697.689ms

threads_per_block = 4,Threads_per_cycle for Vega10 - 1.536GHz =   1

kernel_time (hipEventElapsedTime) =348.875ms

threads_per_block = 8,Threads_per_cycle for Vega10 - 1.536GHz =   2

kernel_time (hipEventElapsedTime) =174.456ms

threads_per_block = 16,Threads_per_cycle for Vega10 - 1.536GHz =   4

kernel_time (hipEventElapsedTime) =87.238ms

threads_per_block = 32,Threads_per_cycle for Vega10 - 1.536GHz =   8

kernel_time (hipEventElapsedTime) =43.629ms

threads_per_block = 64,Threads_per_cycle for Vega10 - 1.536GHz =  16

kernel_time (hipEventElapsedTime) =21.828ms

threads_per_block = 128,Threads_per_cycle for Vega10 - 1.536GHz =  32

kernel_time (hipEventElapsedTime) =10.929ms

threads_per_block = 256,Threads_per_cycle for Vega10 - 1.536GHz =  64

kernel_time (hipEventElapsedTime) =10.914ms

threads_per_block = 512,Threads_per_cycle for Vega10 - 1.536GHz =  64

kernel_time (hipEventElapsedTime) =10.909ms

threads_per_block = 1024,Threads_per_cycle for Vega10 - 1.536GHz =  64
</td></tr></table>
仔細觀察，僅僅当 BlockDim = 256， 512，1024时， 線程產生速度達到峰值。 呢個信息有咩含義， 抑或對GPU程序優化有何指導意義？ 

舉例， 在深度學習中有大量的簡單操作， 例如Copy，  激活函數，如果程序使用了比256小的BlockDim， 那麼程序將很難達到理論值，   例如 64，咁理論極限好有可能係64/256。 深度學習經常使用Padding Copy， 如果 H x W = 7x7，Padding= 3， 咁理論極限將會係13*13/256 = 66%。 

以上兩種情況， 如果程序能夠把原來4 threads的工作合併到一個thread，每個線程處理的事務隨之提高到4倍，例如讀寫操作，將極大地提高理論極限。 
<table><tr><td bgcolor=“#707070”>
Case1 :  min ( 64 *4, 256 )        = 256

Case 2:  min ( 13 * 13 *4, 256) = 256
</td></tr></table>
呢個測試結果是否有值得懷疑嘅地方？  呢個測試結果證明只有BlockDim =256才能達到理論極限，同AMD GCN嘅圖形像素渲染能力唔匹配，顏色渲染能力達到了64 Pixels/Cycle。 GCN架構嘅Pixel Shader都係64个像素一個Wave，換而言之HIP都應該能夠達到 64 Threads/Cycle。 而測試結果只有Pixel Shader嘅1/4，有兩種可能：1 ） ROCm使用咗特別寄存器設置使得線程產生速度降低到了1/4 ;2 ）硬件嘅計算線程生成速度係像素住色器嘅1/4速度。 第二個原因的可能性比較小，GCN統一化的著色器架構設計應保證不同類型的著色器（幾何，像素，計算）線程速度相同，否則對應硬件資源將被浪費。 

### 2.1.3 2D 形狀Block線程速率

本節將測試2D 形狀Block 嘅線程速率，前兩節已知1D最大線程數為1024，咁對應最大嘅 BlockDim應該為 Dim3 （ 32，32，1 ），  最小為Dim3 （ 1，1，1），這樣可以組成32個不同的測試組合。 

編譯執行eaxaple-1c.cpp，得到如下結果。 
<table><tr><td bgcolor=“#707070”>
threads_per_block = [1,1,1],Threads_per_cycle for Vega10 - 1.536GHz =   0

threads_per_block = [2,2,1],Threads_per_cycle for Vega10 - 1.536GHz =   1

threads_per_block = [3,3,1],Threads_per_cycle for Vega10 - 1.536GHz =   2

threads_per_block = [4,4,1],Threads_per_cycle for Vega10 - 1.536GHz =   4

threads_per_block = [5,5,1],Threads_per_cycle for Vega10 - 1.536GHz =   6

threads_per_block = [6,6,1],Threads_per_cycle for Vega10 - 1.536GHz =   9

threads_per_block = [7,7,1],Threads_per_cycle for Vega10 - 1.536GHz =  12

threads_per_block = [8,8,1],Threads_per_cycle for Vega10 - 1.536GHz =  16

threads_per_block = [9,9,1],Threads_per_cycle for Vega10 - 1.536GHz =  20

threads_per_block = [10,10,1],Threads_per_cycle for Vega10 - 1.536GHz =  25

threads_per_block = [11,11,1],Threads_per_cycle for Vega10 - 1.536GHz =  30

threads_per_block = [12,12,1],Threads_per_cycle for Vega10 - 1.536GHz =  36

threads_per_block = [13,13,1],Threads_per_cycle for Vega10 - 1.536GHz =  42

threads_per_block = [14,14,1],Threads_per_cycle for Vega10 - 1.536GHz =  49

threads_per_block = [15,15,1],Threads_per_cycle for Vega10 - 1.536GHz =  56

threads_per_block = [16,16,1],Threads_per_cycle for Vega10 - 1.536GHz =  64

threads_per_block = [17,17,1],Threads_per_cycle for Vega10 - 1.536GHz =  58

threads_per_block = [18,18,1],Threads_per_cycle for Vega10 - 1.536GHz =  54

threads_per_block = [19,19,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [20,20,1],Threads_per_cycle for Vega10 - 1.536GHz =  57

threads_per_block = [21,21,1],Threads_per_cycle for Vega10 - 1.536GHz =  63

threads_per_block = [22,22,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [23,23,1],Threads_per_cycle for Vega10 - 1.536GHz =  59

threads_per_block = [24,24,1],Threads_per_cycle for Vega10 - 1.536GHz =  64

threads_per_block = [25,25,1],Threads_per_cycle for Vega10 - 1.536GHz =  62

threads_per_block = [26,26,1],Threads_per_cycle for Vega10 - 1.536GHz =  61

threads_per_block = [27,27,1],Threads_per_cycle for Vega10 - 1.536GHz =  61

threads_per_block = [28,28,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [29,29,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [30,30,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [31,31,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [32,32,1],Threads_per_cycle for Vega10 - 1.536GHz =  64
</td></tr></table>
結果清晰第顯示，只有当BlockDim嘅總線程數量係256嘅倍數，Dim3 （ 16，16，1 ），Dim3 （ 24，24，1 ），Dim3 （ 32，32，1 ），才能獲得極限綫程生成速率。 Dim3（32，16，1）讀者有興趣可以自己測試。 

對於HIP程序開發者，對於簡單嘅顯存讀寫類，建議使用256倍數嘅BlockDim以獲取最高線程生成速率。 計算異常密集的任務，它的性能主要瓶頸和線程生成速率無關時，建議使用64倍數的BlockDim。 

### 2.1.3 3D 形狀Block的線程生成速率

HIP也提供3D 形狀的Block，1024最大線程數轉化為三維形狀，可以為Dim（16，16，4），Dim（32，16，2），Dim（8，8，64）等。 下面我們選擇一些特殊形狀， 測試其性能變化，Dim3（1，1，1），Dim3（2，2，2），Dim3（3，3，3），Dim3（4，4，4），Dim3（5，5，5） ，Dim3（6，6，6），Dim3（7，7，7），Dim3（8，8，8），Dim3（9，9，9）和Dim3（10，10，10）。 

編譯執行example-1d.cpp。 得到如下結果。 
<table><tr><td bgcolor=“#707070”>
threads_per_block = [1,1,1],Threads_per_cycle for Vega10 - 1.536GHz =   0

threads_per_block = [2,2,2],Threads_per_cycle for Vega10 - 1.536GHz =   2

threads_per_block = [3,3,3],Threads_per_cycle for Vega10 - 1.536GHz =   7

threads_per_block = [4,4,4],Threads_per_cycle for Vega10 - 1.536GHz =  16

threads_per_block = [5,5,5],Threads_per_cycle for Vega10 - 1.536GHz =  31

threads_per_block = [6,6,6],Threads_per_cycle for Vega10 - 1.536GHz =  54

threads_per_block = [7,7,7],Threads_per_cycle for Vega10 - 1.536GHz =  57

threads_per_block = [8,8,8],Threads_per_cycle for Vega10 - 1.536GHz =  64

threads_per_block = [9,9,9],Threads_per_cycle for Vega10 - 1.536GHz =  61

threads_per_block = [10,10,10],Threads_per_cycle for Vega10 - 1.536GHz =  62
</td></tr></table>
呢個實例嘅結論同前兩個測試相同， 只用線程數為256嘅成倍數才能獲得最佳性能。 

2.2 Compute Resources 計算資源
--------------------------

Vega64有64個計算單位（compute unit），每個計算單位有64個乘加器。 咁每個計算單位能夠64 FMAs/Cycle，64个計算單位嘅能力為4096 cycles/ cycle，個個乘法包含一個乘法同加灋，算做兩個浮點運算，乘以頻率1.536Ghz  =  15.6T Flops/s。 我們下面將研究HIPCC如何在單個計算單位獲得64 FMAs /cycle

### 2.2.1 Execute 1，000，000 of FMA： 簡單循環100萬次

 256 threads執行100万次FMA，只有64个搭加器，咁個個搭加器需要執行400万条指令，咁執行時間最短時間為 4/1.536 = 2.6 毫秒。 編譯器通常帶有好多有優化技術，它會優化掉對最終結果無貢獻嘅大量計算，因此程序必須迷惑編譯器，詐假意程序一定會產生輸出。 
```
#define FMA_PER_THREADS       1000000

__global__ void

test_kernel(hipLaunchParm lp,

       float* __restrict__ a)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; 

       float t0 = (float)x / (float) (x + 1); 

       float t1 = float(y + 1) / (float)(y + 100000000); 

       float sum=0.0; 

       for(int i =0; I < FMA_PER_THREADS; i++)

       {

              sum = t0 *sum + t1; 

       }

迷惑編譯器，防止編譯器優化把上面一百万条指令全部移除

       if （（float（x）+sum） < -1.0f）

       {

              a[0] = sum; 

       }

}
```
完整的程序參考example-2a.cpp。 使用如下命令行編譯：

<table><tr><td bgcolor=“#707070”>
hipcc example-2a.cpp -o example-2a.exe 
</td></tr></table>

hcc 提供了一個反匯編工具 /opt/rocm/hcc/bin/extractkernel。 用如下命令獲得上述test_kernel的GCN匯編代碼：

<table><tr><td bgcolor=“#707070”>
extractkernel -i  ./example-2a.exe
</td></tr></table>
執行命令得到嘅輸出：
<table><tr><td bgcolor=“#707070”>
Generated GCN ISA for gfx900 at: ./example-2a.exe-gfx900.isa
</td></tr></table>
打開example-2a.exe-gfx900.isa，可以發現如下代碼段：
<table><tr><td bgcolor=“#707070”>
000000000000124c BB0_1:

       v_mad_f32 v3, v1, v3, v2                                   // 00000000124C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001254: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000125C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001264: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000126C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001274: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000127C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001284: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000128C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001294: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000129C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012A4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012AC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012B4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012BC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012C4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012CC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012D4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012DC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012E4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012EC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012F4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012FC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001304: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000130C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001314: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000131C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001324: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000132C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001334: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000133C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001344: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000134C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001354: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000135C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001364: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000136C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001374: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000137C: D1C10003 040A0701

       s_sub_i32 s2, s2, 40                                       // 000000001384: 8182A802

       s_cmp_lg_u32 s2, 0                                         // 000000001388: BF078002

       v_mad_f32 v3, v1, v3, v2                                   // 00000000138C: D1C10003 040A0701

       s_cbranch_scc1 BB0_1                                       // 000000001394: BF85FFAD
</td></tr></table>
該段GCN 匯編代碼係對應test_kernel嘅100万次循環，包含：

* 40个v_mad_f32指令，編譯器做咗默認40次循環展開，
*   兩條SALU，s_sub_i32，s_cmp_lg_u32
*   一條跳轉指令 s_cbranch_scc1

那麼對應FMA指令的有效率為， 40/43 = 93%，乘以每個計算單位的64個乘加器，理論上可以獲得59個FMA /Cycle

現時執行example-2a.exe獲得測試性能。 
<table><tr><td bgcolor=LightGray>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     44
</td></tr></table>
實際上測試程序使用256 threads僅僅獲得了44个FMA/Cycle，遠遠低於理論預期。 咁呢度存在一些我哋仲未發現嘅性能陷阱。 可以有兩個方向進行測試，例如採用兩層循環，控制循環展開嘅指令數目， 增加threads數目以提高並行性，並減少因指令緩存（ instruction Cache ）讀取失敗嘅機率。 

### 2.2.2 Specified Loop Unroll： 指定循環展開大小

指定循環展開塊的大小可以減少SVALU的比例，提高程序整體效率，我們來嘗試指定循環展開數量為100。 代碼如下：
```
#pragma unroll 100

       for(int i =0; I < FMA_PER_THREADS; i++)

       {

              sum = t0 *sum + t1; 

       }
```

編譯example-2b .cpp並執行獲得如下結果。 
<table><tr><td bgcolor=“#707070”>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     48
</td></tr></table>
成績由44 FMA/Cycle/CU 提高到了48 FMA/Cycle/CU。 繼續使用extractkernels嚟檢查GCN匯編代碼，我哋發現主體循環代碼包含：

* 100個v_mad_f32指令，完全匹配指定的循環展開次數100次
*   兩條SALU，s_addk_i32，s_cmp_lg_u32
*   一條跳轉指令

此時example-2b能獲得理論性能為100/103 * 64 = 62 FMA/cycle/CU， example-2a高3 FMA/Cycle/CU，實際獲得4 FMA/Cycle/CU 的提升。 實際效果良好。 但係距離我哋期待嘅 64 FMA/Cycle/CU仍然有比較大嘅差距。 

### 2.2.3 Double Loop：雙層循環

  Example-2c將嘗試多層循環，內存循環體使用100次循環，外層循環體10000次循環。 
```
       for(int i =0; i < FMA_PER_THREADS/100; i++)

       {

              for(int j=0; j < 100; j++)

              sum = t0 *sum + t1; 

       }
```
編譯執行example-2c.cpp得到如下輸出結果：
<table><tr><td bgcolor=“#707070”>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     59 
</td></tr></table>
性能得到了很大提升，以慣例繼續使用extractkernel查看主要循環體：

* 100個v_mad_f32指令，完全匹配內層循環體100次
*   兩條SALU，s_add_i32，s_cmp_lg_u32
    *   s_add_i32 s2, s2, -1
*   一條跳轉指令s_cbranch_scc1

這個結果很難解釋為何example-3c.cpp比example-3b.cpp獲得大幅度的性能提升。 仔細檢查example-2b同example-2c嘅GCN匯編代碼，另外一個微小區別係成個Kernel代碼段嘅長度差咗4个字節。 一個可能測猜測係Instruction Cache有特定嘅尺寸，對於性能影響好大，如果成個循環體代碼長度係Instruction Cache嘅完整倍數，咁將獲得最優性能，否則最終嘅性能為實際指令編碼嘅字節數與對應 Cacheline之比。 例如Instruction Cache為8個DWORD，咁成個循環體最多損失14 DWORDs，103 條指令編碼總共203个DWORDs，最少26条Cachelines，最多27条 Cachelines，如果多一個不對齊的Cahceline，那麼最多損失8%的性能，或者5-6條FMA/Cycle/CU 。 如果Instruction Cache Line有兩條不對齊的Cachelines，最大性能差距會達到11條FMA/Cycle/CU。 

### 2.2.4 Increasing Threads In Parallel ：增加並行線程

256 threads意味住每個乘加器只有一個線程， 如果把每個乘加器嘅線程數量增加到2个，咁個個搭加器可以乒乓綫程以隱藏延遲，是否能夠提高計算單位嘅效率？ 

編譯並執行Example-2d.cpp，獲得如下結果。 
<table><tr><td bgcolor=“#707070”>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     59

Total Threads = 1 * 512, FMA_per_cycle for Vega10 - 1.536GHz =     62

Total Threads = 1 * 768, FMA_per_cycle for Vega10 - 1.536GHz =     63

Total Threads = 1 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =     63
</td></tr></table>
 結果顯示，当我哋增加1个計算單位嘅並行線程數，能夠有效增加SIMD嘅效率。 

### 2.2.5 Enough Parallel Threads： 足夠多線程充滿64个計算單位

前面四節討論了如何獲取單個計算單位的峰值性能，如果想要達到最佳性能，一個可能的辦法是手寫GCN assembly，然後仔細調整循環體Cacheline的長度，使得Assembly Shader無限接近理論最高性能。 

節我哋將探究不同 Block數量對於性能嘅影響。 下面這段程序使用雙重循環測試峰值計算性能，Block從1，2，3，...，128，BlockDim可選取 Dim3（256，1，1 ），Dim3 （512，1，1），Dim3（768，1，1）和 Dim3（1024，1，1）。 
```
  for (int i = 1; i < 5; i = i + 1) {

     for (int j = 0; j < 129; j++)

     {

          hipEventRecord(start, NULL); 

          hipLaunchKernel(null_kernel,

                                 dim3(j, 1, 1),

                                 dim3(THREADS_PER_BLOCK_X * i, 1, 1),

                                 0, 0,

                                 deviceA); 

          hipEventRecord(stop, NULL); 

          hipEventSynchronize(stop); 

          hipEventElapsedTime(&eventMs, start, stop); 

          printf("kernel_time (hipEventElapsedTime) =%6.3fms\\n", eventMs); 

          double FMA_per_cycle = double(THREADS_PER_BLOCK_X) * i *j * double(FMA_PER_THREDS) / eventMs / (1.536 * 1e6) + 0.5; 

            printf("Total Threads = %d * %d, FMA_per_cycle for Vega10 - 1.536GHz = %6d\\n", j, THREADS_PER_BLOCK_X * i,    

                     (int) FMA_per_cycle); 

        }

  }
```
編譯執行example-2e.cpp將得到4x128=512不同嘅性能組合， 我哋選取其中嘅10个組合。 
<table><tr><td bgcolor=“#707070”>
kernel_time (hipEventElapsedTime) =10.630ms

Total Threads = 1 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =     63

kernel_time (hipEventElapsedTime) =10.639ms

Total Threads = 2 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    125

kernel_time (hipEventElapsedTime) =10.641ms

Total Threads = 3 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    188

Total Threads = 8 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    499

kernel_time (hipEventElapsedTime) =10.720ms

Total Threads = 16 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    995

kernel_time (hipEventElapsedTime) =10.803ms

Total Threads = 32 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   1975

kernel_time (hipEventElapsedTime) =10.963ms

Total Threads = 64 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   3892

kernel_time (hipEventElapsedTime) =21.376ms

Total Threads = 65 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   2027

kernel_time (hipEventElapsedTime) =21.383ms

Total Threads = 66 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   2058

kernel_time (hipEventElapsedTime) =21.386ms
</td></tr></table>
我們觀察到Block數量從1到64，程序執行時間幾乎不變，GPU的FMA速率線性增長， 而Block 數量增加到65，GPU執行時間增加一倍，表明Vega10 GPU總共有64个計算單位。 我哋喺做程序優化嘅時候，程序需要儘可能保證Block嘅總數量係64嘅整倍數，這樣能夠保證減少因為計算單位空閒造成嘅性能下降， 例如總共65个Block ，咁它的最大理論效率只有64/128 = 50.8%。 性能基準測試程序期望壓榨每一个百分点嘅性能，Block總數將會成為成為性能優化嘅一個不可忽視手段。 

2.3 VGPR： 矢量通用寄存器
-----------------

上節我們討論了計算單元和並行線程數的關係，並且分析了Instruction Cacheline對於性能的影響。 每個計算線程仲有非常重要嘅資源-VPGRs。 當Kernel使用的VGPR資源過多，就會造成只有一個Thread運行在對應的MAC，或者單一wave（ 64 threads ）運行喺一個SIMD，咁會造成嚴重嘅性能下降。 如果線程使用嘅VGPR超過了硬件最大資源，編譯器將會開闢一塊內存，將超出部分暫時緩存到GPU顯存，性能可能會下降到峰值性能嘅5%以下。 

測試最大VGPR有好多方法， 例如構造一個VPGR嘅二叉樹，防止編譯器優化減少VGPR嘅數量，每次增加二叉樹葉子節點嘅數量，指導性能劇烈突然下降為止。 我呢度採用另外一個簡單方法，rocm 提供咗一個內嵌匯編嘅方式，下面嘅呢個 Kernel測試最大VGPR是否為V255，如果能夠編譯成功，咁可以 VGPR總數為256。 然後逐漸增大VGPR索引，睇下是否編譯無法通過，抑或執行失敗，咁上一個成功嘅索引值就係最大VGPR。 

下面係一個測試VGPR嘅簡單實例。 
```
__global__ void

test_kernel_255(hipLaunchParm lp,

       float* __restrict__ a)

{

       asm volatile("v_mov_b32 v0, 0"); 

       asm volatile("v_mov_b32 v255, 0" ); 

}
```
 我哋嘗試編譯並執行example-3a.cpp。 編譯和執行都順利完成。 然後再次用神器extractkernel查看 GCN assembly shader。 發現程序只有如下三行代碼：
```
              v_mov_b32_e32 v0, 0                                        // 000000001100: 7E000280

              v_mov_b32_e32 v255, 0                                    // 000000001104: 7FFE0280

              s_endpgm                                                            // 000000001108: BF810000
```
呢個結果非常符合我哋嘅預期。 我們可以增加下面一個Kernel到example-3b.cpp
```
__global__ void

test_kernel_256(hipLaunchParm lp,

       float* __restrict__ a)

{

       asm volatile("v_mov_b32 v0, 0"); 

       asm volatile("v_mov_b32 v256, 0"); 

}
```
老槼矩，調用 hipcc嘗試編譯example-3b.cpp。 編譯失敗並獲得下面錯誤信息：
<table><tr><td bgcolor=“#707070”>
<inline asm>：1：16：error： unknown token in expression

        v_mov_b32 v256, 0

                      ^

note: ! srcloc = 833

<inline asm>：1：18：error： not a valid operand

        v_mov_b32 v256, 0

                        ^

note: ! srcloc = 833

Generating AMD GCN kernel failed in llc for target: gfx900

clang-8: error: linker command failed with exit code 1 (use -v to see invocation)
</td></tr></table>
呢個kernel有兩個不同嘅內嵌匯編，第一条成功而第二条衰，表明Vega10能夠支持嘅最大VGPR為256 （由V0開始計數為V255 ）。 

2.4 SGPR： 標量通用寄存器
-----------------

SGPR喺AMD GCN體系結構係非常重要嘅一項特性。 SGPR第一個用途係讀GPU顯存常量到計算單位，例如圖形渲染中嘅投影矩阵，紋理對象描述，紋理採樣描述等。 SGPR係可讀可寫， 它可以作為用于程序流程控制，例如循環變量， 從而減低SIMD VGPR嘅需求，同時也降低大部分循環控制嘅功耗。 

同VGPR一樣，SGPR資源都係有限嘅， 我哋都可以採用內聯匯編嘅方法測試最大SGPR。 VGPR越界編譯緊嘅時候直接出錯，理論SGPR都有同樣嘅性質。 Example-4a.cpp使用下面的Kernel尋找最大SGPR。 
<table><tr><td bgcolor=“#707070”>
__global__ void

test_kernel_255(hipLaunchParm lp,

       float* __restrict__ a)

{

   asm volatile("s_mov_b32 s0, 0"); 

   asm volatile("s_mov_b32 s95, 0" ); 

   asm volatile("s_mov_b32 s96, 0" ); 

   asm volatile("s_mov_b32 s97, 0" ); 

   asm volatile("s_mov_b32 s98, 0" ); 

   asm volatile("s_mov_b32 s99, 0" ); 

   asm volatile("s_mov_b32 s100, 0" ); 

   asm volatile("s_mov_b32 s101, 0" ); 

   asm volatile("s_mov_b32 s102, 0" ); 

   asm volatile("s_mov_b32 s103, 0" ); 

   asm volatile("s_mov_b32 s104, 0" ); 

   asm volatile("s_mov_b32 s105, 0" ); 

   asm volatile("s_mov_b32 s106, 0" ); 

   asm volatile("s_mov_b32 s107, 0" ); 

   asm volatile("s_mov_b32 s108, 0" ); 

   asm volatile("s_mov_b32 s109, 0" ); 

}
</td></tr></table>
老規矩，使用糸hipcc  example-4a .cpp-o example-4a .exe 糸嘗試編譯。  得到如下錯誤：
<table><tr><td bgcolor=“#707070”>
<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s102, 0

                      ^

note: ! srcloc = 950

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s102, 0

                        ^

note: ! srcloc = 950

<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s103, 0

                      ^

note: ! srcloc = 990

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s103, 0

                        ^

note: ! srcloc = 990

<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s104, 0

                      ^

note: ! srcloc = 1030

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s104, 0

                        ^

note: ! srcloc = 1030

<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s105, 0

                      ^

note: ! srcloc = 1070

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s105, 0

                        ^

note: ! srcloc = 1070

<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s106, 0

                      ^

note: ! srcloc = 1110

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s106, 0

                        ^

note: ! srcloc = 1110

<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s107, 0

                      ^

note: ! srcloc = 1150

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s107, 0

                        ^

note: ! srcloc = 1150

<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s108, 0

                      ^

note: ! srcloc = 1190

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s108, 0

                        ^

note: ! srcloc = 1190

<inline asm>：1：16：error： unknown token in expression

        s_mov_b32 s109, 0

                      ^

note: ! srcloc = 1230

<inline asm>：1：18：error： not a valid operand

        s_mov_b32 s109, 0

                        ^

note: ! srcloc = 1230

Generating AMD GCN kernel failed in llc for target: gfx900

clang-8: error: linker command failed with exit code 1 (use -v to see invocation)
</td></tr></table>
SGPR S102之前能夠被編譯器正確識別，我哋就搵到咗最大程序SGPR為S101 （由S0開始計數）。 在GCN體系結構設計中，SGPR資源始終可以用到SGPR 101，讀者可以用BlockDim=Dim3 （ 1024，1，1 ） 進行驗證，而VGPR喺BlockDim=Dim3 （ 1024，1，1 ）則下降到V63。 

2.5 Divergence：Wave 分歧
------------------------

在SIMD結構中， 有一種特殊的情況， 如果一個wave只有1個Thread和其他63個Threads 執行路徑不同，那麼對性能有何影響，例如我們把2.2.1的代碼修改如下：
```
       if (hipThreadIdx_x == 0) {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t0 * sum + t1; 

              }

       }

       else {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t1 * sum + t0; 

              }

       }
```
SIMD嘅特點係所有Threads必須執行相同嘅指令，由於Thread0同其他代碼路徑不同，咁編譯器必須先生成Thread0嘅代碼，然後生成剩餘63 個Threads嘅代碼。 那麼SIMD則順序Thread0的代碼，然後Thread1-63的代碼。 咁性能將下降到2.2.1實例代碼嘅50%。 

是否可以改進呢種分歧？ 將2.2.1嘅實例中循環體部分看作一個函數foo，咁Thread0可以當作foo （ t0，t1 ），thread1-63 看做是foo（t1，t0），通過對參數的交換，實現所有線程調用同樣參數，那麼可以大大降低Divergence帶來的性能下降。  參考下面test_kernel_optimize.
```
__global__ void

test_kernel_divergence(hipLaunchParm lp,

       float* __restrict__ a)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; 

       float t0 = (float)x / (float)(x + 1); 

       float t1 = float(y + 1) / (float)(y + 100000000); 

       float sum = 0.0; 

       if (hipThreadIdx_x == 0) {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t0 * sum + t1; 

              }

       }

       else {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t1 * sum + t0; 

              }

       }

       if ((float(x) + sum) < -1.0f)

       {

              a[0] = sum; 

       }

}

__global__ void

test_kernel_optimize(hipLaunchParm lp,

       float* __restrict__ a)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y; 

       float t0 = (float)x / (float)(x + 1); 

       float t1 = float(y + 1) / (float)(y + 100000000); 

       float sum = 0.0; 

       if (hipThreadIdx_x == 0) {

              float t = t0; 

              t1 = t0; 

              t0 = t; 

       }

       for (int i = 0; I < FMA_PER_THREDS;  i++)

       {

              sum = t0 * sum + t1; 

       }

       if ((float(x) + sum) < -1.0f)

       {

              a[0] = sum; 

       }

}
```
編譯並執行程序example-5a.cpp得到如下結果，上述理論得到了驗證。 
<table><tr><td bgcolor=“#707070”>
execute test kernel

kernel_time (hipEventElapsedTime) = 3.774ms

Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     44

execute divergence kernel

kernel_time (hipEventElapsedTime) = 8.119ms

Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     21

execute optimized kernel

kernel_time (hipEventElapsedTime) = 3.838ms

Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     43
</td></tr></table>
2.6 Memory Read Latency： 顯存讀寫延遲
------------------------------

### 2.6.1 L2 Cache Miss： 直接由顯存讀寫

讀顯存的延遲可以連續讀不同的Cacheline，下一次讀操作用前一次讀操作的返回值，連續執行1，000，000次的有依賴關係的讀操作，取平均即可獲得讀操作的延遲。 我哋目前仲唔知如何Cacheline大小，而依據經驗值，條cacheline長度 可能為 16，32，64， 128字節，因此我們程序讀下一個值的地址比上一個地址大256DWORDs（1024字節），這樣可以保證整個程序不會讀兩個相同的Cacheline。 程序中buf嘅所有值為256。 
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

int t = buf[x]; 

       dependency reads

       for( int i=1; i < MAX_MEM_READS; i++)

       {

          t = buf[t * i ]; 

       }            

       if( t > 0x3fffffff)

       {

              buf[x] = t; 

       }

}
```
編譯執行example-6a.cpp得到如下結果。 
<table><tr><td bgcolor=“#707070”>
kernel_time (hipEventElapsedTime) =442.050ms

mem_read_latency_cycle =   647 cycles for Vega10--1.536GHz
</td></tr></table>
使用extractkernel工具產生GCN assembly得到以下指令序列做一次顯存讀操作，總計5條VALU和1條SALU 指令，六條指令需要至少24个時鐘周期，v_lshlrev_b64可能需要16个始終周期，咁可以得出顯存讀操作嘅延時為610个始終周期。 
<table><tr><td bgcolor=“#707070”>
              v_mul_lo_u32 v2, v2, s3                                    // 000000001504: D2850002 00000702

              s_add_i32 s3, s2, -2                                       // 00000000150C: 8103C202

              v_ashrrev_i32_e32 v3, 31, v2                               // 000000001510: 2206049F

              v_lshlrev_b64 v[2:3], 2, v[2:3]                            // 000000001514: D28F0002 00020482

              v_add_co_u32_e32 v2, vcc, s0, v2                           // 00000000151C: 32040400

              v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 000000001520: 38060704

              global_load_dword v2, v[2:3], off                          // 000000001524: DC508000 027F0002

              s_waitcnt vmcnt(0)  
</td></tr></table>
2.6.2 CacheLine Length： 緩存行長度
-----------------------------

本節畀出一個唔太準確嘅測量緩存行長度嘅辦法。 參考下面嘅程序，buf中所有嘅值都為固定值1，而卻只有一個thread，所有嘅讀取地址都依賴於上一個地址，如果多個連續嘅讀喺同一個地址內，緩存產生命中，咁它的平均單筆延遲遠小於由讀顯存延遲，否則非常接近讀顯存延遲。 
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int rangesize, int totalreads)

{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

    int t = buf[x]; 

    dependency reads

    for( int i=1; I < totalreads; i++)

    {

       int address = i * t * rangesize; 

       address = address - 1; 

address = (address & (rangesize - 1)) | (address & (~(rangesize-1)) ); 

       t = buf[address]; 

    }               

     if( t > 0x3fffffff)

     {

         buf[x] = t; 

     }

}
```
編譯執行example-6b.cpp得到如下輸出結果，可以得出結論 Cacheline長度為64字節。 
<table><tr><td bgcolor=“#707070”>
RangeSize[      16], kernel_time (hipEventElapsedTime) =4639.969ms

RangeSize[      16], mem_read_latency_cycle =   361 cycles for Vega10--1.536GHz

RangeSize[      32], kernel_time (hipEventElapsedTime) =3060.621ms

RangeSize[      32], mem_read_latency_cycle =   476 cycles for Vega10--1.536GHz

RangeSize[      64], kernel_time (hipEventElapsedTime) =2192.251ms

RangeSize[      64], mem_read_latency_cycle =   682 cycles for Vega10--1.536GHz

RangeSize[     128], kernel_time (hipEventElapsedTime) =1093.262ms

RangeSize[     128], mem_read_latency_cycle =   681 cycles for Vega10--1.536GHz

RangeSize[     256], kernel_time (hipEventElapsedTime) =566.791ms

RangeSize[     256], mem_read_latency_cycle =   706 cycles for Vega10--1.536GHz
</td></tr></table>
### 2.6.3 L1/L2 Cacheline Hit Latency：一/二級緩存命中延時

Example-6c.cpp展示一個簡單嘅Kernel測量一級緩存命中嘅延時。 設置rangesize = 1024，4096字節遠小於16KB L2 Cache，咁L1 Cache嘅命中率接近99%。  將步長設置為Cacheline大小16DWORDs==64字節，那麼每次讀取指令都會指向一個新的Cacheline。 
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int rangesize, int totalreads)

{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

    int t = buf[x]; 

    dependency reads

    for( int i=1; I < totalreads; i++)

    {

        int address = i * t * rangesize; 

        address = address - 1; 

        address = (address & (rangesize - 1)); 

        t = buf[address]; 

    }               

       if( t > 0x3fffffff)

       {

              buf[x] = t; 

       }

}
```
編譯執行example-6c.cpp 得到如下結果：
<table><tr><td bgcolor=“#707070”>
RangeSize[    4096], kernel_time (hipEventElapsedTime) =48.065ms

RangeSize[    4096], mem_read_latency_cycle =   239 cycles for Vega10--1.536GHz
</td></tr></table>
咁可以猜測L1 Cache命中延時小於239个時鐘周期，用xextractkernel -i example-6c .exe唻查看GCN Assembly  代碼，獲得主循環體代碼如下：
<table><tr><td bgcolor=“#707070”>
0000000000001170 BB0_2:

        s_waitcnt vmcnt(0)                                        

        v_mul_lo_u32 v2, v2, s2                                   

        v_mov_b32_e32 v4, s1                                      

        v_mul_lo_u32 v2, v2, s5                                   

        s_add_i32 s5, s5, 1                                       

        s_cmp_lg_u32 s3, s5                                       

        v_add_u32_e32 v2, -1, v2                                  

        v_and_b32_e32 v2, s4, v2                                  

        v_ashrrev_i32_e32 v3, 31, v2                             

        v_lshlrev_b64 v[2:3], 2, v[2:3]                          

        v_add_co_u32_e32 v2, vcc, s0, v2                         

        v_addc_co_u32_e32 v3, vcc, v4, v3, vcc

        global_load_dword v2, v[2:3], off                        

        s_cbranch_scc1 BB0_2                                     
</td></tr></table>
GCN Assembly代碼總計9条VALU指令， 4条Scalar指令，呢啲指令嘅延時需要64时鐘周期，考慮到由於 Cacheline不對齊會損失32-60個始終周期，L1 Cache命中的延時最低100個時鐘周期，最高130個時鐘周期。 

Example-6d.cpp把rangesize修改為32768 （ 128KB ），編譯執行獲得如下結果。 根據example-6c的分析，L2 CacheLIne命中的延時介乎270-300個時鐘周期之間。 
<table><tr><td bgcolor=“#707070”>
RangeSize[  131072], kernel_time (hipEventElapsedTime) =75.581ms

RangeSize[  131072], mem_read_latency_cycle =   376 cycles for Vega10--1.536GHz
</td></tr></table>
2.7 Alternative Method to measure CacheLine Size： 另一組測試Cacheline長度
-----------------------------------------------------------------

### 2.7.1 測試 CacheLine 大小

Example-7a.cpp和example-7b.cpp嘗試不斷增加讀寫步長來Cacheline大小，該組測試已經被2.6.2代替。 

### 2.7.2 Divergence for Memory Read/Write：顯存訪問分歧

Example-7c.cpp專門設計一個非常簡單的方法產生顯存讀寫分歧而導致的性能下降一半。 畀Thread0嘅顯存地址計算同其他64个地址計算不同，咁編譯器是否會產生兩個不同global_store_dword指令，編譯後檢查Extractkernel產生嘅 GCN assembly 代碼，發現只有一條global_store_dword，對於呢個簡單嘅代碼，HIPCC編譯器表現良好。 
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int divergence )

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       if ((hipThreadIdx_x & divergence) == divergence)

       {

               buf[x] = x; 

       }

       else  

       {     

              buf[x&(NUM-1)] = x; 

       }   

}
```
2.8 NCHW-4D Index Generation：4D數組索引生成
----------------------------------------

在優化CNN卷積運算中，需要實時生成索引進行加速。 假設我哋需要生成NCHW對應Channel=0時候NHW個元素嘅索引。 下面係簡單代碼實現，BlockDim = Dim3 （ 256，1，1 ）， Grim = Dim3 （ H * W/256，N，1 ）。 
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int h, int w, int c)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       int n = hipBlockIdx_y; 

       if (x < (h * w))

       {

              int nchw_offset = x + n * c * h * w; 

              int nhw_offset = x + n * h * w; 

              buf[nhw_offset] = nchw_offset; 

       }

}
```
編譯example-8a.cpp執行獲得309GB/s嘅速度。 考慮到hipLaunchKernel需要7微秒嘅額外使用，達到378GB/s嘅速度。 考慮到數量比較細，相對于480GB/s嘅峰值性能，已經係好好嘅就成績。 
<table><tr><td bgcolor=“#707070”>
N*H*W=[1024,56,56], hipEventElapsedTime =38.715 microseconds, 309.001966 GB/s
</td></tr></table>
2.9 Local Data Share：本地數據共享
---------------------------

### 2.9.1 LDS Latency

GCN架構中LDS訪問都係異步指令， 同顯存讀寫指令一樣，我哋首先要獲得LDS指令嘅延時。 衕理，使用一個線程，使用循環不斷訪問同一個地址，咁我哋就可以獲得LDS Latency。 Mask防止訪問越界， Thread0嘅Temp始終等於0， 該Mask并無特殊必要。 
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int mask, int outerLloops)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       __shared__ int ldsData[4096]; 

       ldsData[hipThreadIdx_x] = buf[x]; 

       int temp = hipThreadIdx_x; 

       for(int i = 0; I < outerLloops; i++){

              for(int j = 0; j < INNER_LOOPS; j++)

              {

                      temp = ldsData[temp] & mask; 

              }

       }

       if (temp > 0)

       {

              buf[x] = temp; 

       }

}
```
編譯後example.cpp並使用extractkernel發現LDS read由如下序列指令：
<table><tr><td bgcolor=“#707070”>
              v_and_b32_e32 v0, s0, v0                    

              v_lshlrev_b32_e32 v0, 2, v0                 

              ds_read_b32 v0, v0                               

              s_waitcnt lgkmcnt(0)                             
</td></tr></table>
2条VALU指令需要20个時鐘周期。 執行example-9a獲得如下結果，我們可以斷定LDS 延時最好情況低於44個時鐘周期：
<table><tr><td bgcolor=“#707070”>
latency for Vega10(1.536Ghz):  63 cycles
</td></tr></table>
### 2.9.2 LDS bank Conflicts

有32个Bank，如果每32threads中兩個以上訪問同一Bank，咁將造成Bank衝突，需要增加一個時鐘周期嚟訪問相同Bank 嘅數據。 下面的實例Buf的數據被初始化為和每個線程的hipThreadIdx_x相同，透過Stride來控制是否發生衝突，例如stride=1那麼就是沒有 Bank衝突發生，否則有可能發生不同的Bank 衝突。 

該實例只使用了64个threads即一個Wave，需要透過一個循環對4096个LDS單元做初始化。 然後透過mask保證訪問地址唔越界。 
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int stride, int mask, int outerLloops)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       __shared__ int ldsData[4096]; 

       for (int i = 0; i < NUM; i += 64)

       {

              ldsData[hipThreadIdx_x + i] = buf[hipThreadIdx_x + i]; 

       }

       int temp = (hipThreadIdx_x * stride) & mask; 

       for(int i = 0; I < outerLloops; i++)

       {

              for(int j = 0; j < INNER_LOOPS; j++)

              {

                      temp = ((ldsData[temp] + hipThreadIdx_x)*stride ) & mask; 

              }

       }

       if (temp > 0)

       {

              buf[x] = temp; 

       }

}
```
循例編譯並執行example-9b.cpp，截取部分輸出結果如下：
<table><tr><td bgcolor=“#707070”>
strdie = [1], latency for Vega10(1.536Ghz):  87 cycles

strdie = [2], latency for Vega10(1.536Ghz):  90 cycles

strdie = [3], latency for Vega10(1.536Ghz):  87 cycles

strdie = [4], latency for Vega10(1.536Ghz):  93 cycles

strdie = [5], latency for Vega10(1.536Ghz):  87 cycles

strdie = [6], latency for Vega10(1.536Ghz):  87 cycles

strdie = [7], latency for Vega10(1.536Ghz):  85 cycles

strdie = [8], latency for Vega10(1.536Ghz):  99 cycles

strdie = [9], latency for Vega10(1.536Ghz):  85 cycles

strdie = [10], latency for Vega10(1.536Ghz):  87 cycles

strdie = [11], latency for Vega10(1.536Ghz):  87 cycles

strdie = [12], latency for Vega10(1.536Ghz):  91 cycles

strdie = [13], latency for Vega10(1.536Ghz):  87 cycles

strdie = [14], latency for Vega10(1.536Ghz):  89 cycles

strdie = [15], latency for Vega10(1.536Ghz):  87 cycles

strdie = [16], latency for Vega10(1.536Ghz):  115 cycles
</td></tr></table>
結果非常有趣，Stride為奇數的延遲都為87Cycles以下， Stride=2，4，8，16的延遲急劇增加，stride為偶數的延遲大部分超過87 cycles， 這和我們在其他文章中看到的一致，Stride為奇數能夠消除Bank Conflicts，最糟糕的情況是Sttride= 2^N。 

可以採用另外一個方法證明呢個問題，做一個Excel表格，第一列依次為Thread ID 0-255，第二列為對應Stride=1嘅地址== ThreadID * Stride，  第三列為對應嘅Bank ID = （ ThreadID * Stride ） % 32，變換Stride，看看是否Bank ID能夠均勻分布在 0-31，如不能，則發生Bank Conflicts。 

2.10 Memory Channel Conflicts： 存儲通道衝突
------------------------------------

高端GPU都係基於通道內存嚟提高多帶寬，咁每個通道嘅內存只能讀寫特定嘅地址空間。 假設一個多通道顯存設計，每4KB內存空間，分配畀16个顯存通道，咁個個顯存通道只能讀寫其中嘅256字節嘅連續地址段。 

下面嘅實例程序使用Proctectbits將保持高於16KB嘅地址不變，ShrinkBits把低位地址空間而家一個抑或多個顯存通道，咁將產生衝突，從而導致性能下降。 
```
#define PROTECT_BITS  (0xFFFF0000)

__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int protectBits, int shrinkBits)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       int address; 

address = (x & protectBits) | (x &  shrinkBits); 

       buf[address] = x; 

}
```
我哋編譯執行example-10a.cpp獲得下面結果，可以清楚睇到最壞情況只有25%左右嘅性能。 
<table><tr><td bgcolor=“#707070”>
Shrink Size in Bytes[128], bandwidth 181 (GB/S)

Shrink Size in Bytes[256], bandwidth 90 (GB/S)

Shrink Size in Bytes[512], bandwidth 181 (GB/S)

Shrink Size in Bytes[1024], bandwidth 360 (GB/S)

Shrink Size in Bytes[2048], bandwidth 359 (GB/S)

Shrink Size in Bytes[4096], bandwidth 359 (GB/S)

Shrink Size in Bytes[8192], bandwidth 359 (GB/S)

Shrink Size in Bytes[16384], bandwidth 359 (GB/S)

Shrink Size in Bytes[32768], bandwidth 359 (GB/S)

Shrink Size in Bytes[65536], bandwidth 359 (GB/S)

Shrink Size in Bytes[131072], bandwidth 358 (GB/S)
</td></tr></table>
例如SGEMM（單精度浮點矩陣乘法），如果矩陣A 的尺寸為 [4096，4096]，矩陣B的尺寸也為[4096，4096]，那麼讀取矩陣A 同矩阵B就會遇到存儲通道讀寫衝突。 

如果大範圍測試M=N=K情況下嘅性能，由128開始，步長為16，會睇到好多性能下降嘅組合，其中一個重要原因就係存儲通道讀寫衝突引起。 

SGEMM避免讀寫衝突的一個簡單方法是使用Padding，例如K=4096，修改行的長度為4096+16，每行最後16個數據無效，可以有效提高性能。 

2.11 Math Functions：數學函數
------------------------

如果對CPU嘅數學函數做過測試，都應該知道每條數學函數需要数十到数百条指令完成。 數學函數在電腦中使用最低六次泰勒級數展開，加上額外的一些操作，數十條指令是非常正常的。 每下面一個實例用雙精度（ Double Precision ）三角函數嚟測試數學。 
```
#define INNER_LOOPS  100

#define OUTER_LOOPS  10000

__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int outerLoops)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

       double f = sin(x / 256.0); 

       for (int i = 0; I < outerLoops; i++)

              for (int j = 0; j< INNER_LOOPS; j++)

                      f = sin(f); 

       if (f > 0.999)

       {

              buf[x] = f; 

       }

}
```
編譯執行example-11a.cpp得到如下結果：
<table><tr><td bgcolor=“#707070”>
sin --double needs 2339 cycles
</td></tr></table>
該結果符合預期， sin的數學函數實現分兩個部分，把角度映射到[0，2Pi]，將耗費大量指令，然後使用泰勒級數展開，同時Mi25的FMA64只有1/16的速度， 雙精度Sin超過了140条指令。 有興趣嘅可以嘗試單精度sin，cos，log， exp，tan，  arcsin，sqrt， rsqrt等常用超越函數嘅使費。 

基礎嘅數學定理可以大大減少計算開銷，例如exp （ x，y ） * exp （ x，z ）等價於exp （ x，y + z ），if （ sqrt （ a ）< b ）等價於if （ A < b *b ），  if （ arcsin （ a ）<arcsin （ b ））等價於if （ A < b ）。 

2.12 Reduction：歸約
-----------------

Reduction是一個非常常見的操作，例如求一個數組的最大、最小值，或者求和。 常見的GPU實現，第一步將所有數據寫到LDS，第二步有效Threads減半，每個有效線程讀兩個數，求和，然後結果寫回LDS，重複步驟二直到有效線程數為1 。 根據我哋前面嘅測試，LDS讀寫嘅延遲比較大，如果每次對4个數求和，是否可以大大提高讀寫速度？ 
```
__global__ void

test_kernel(hipLaunchParm lp,

        int* __restrict__ buf, int reduce_number_once)

{

        int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

        __shared__ int ldsData[256]; 

        ldsData[hipThreadIdx_x] = buf[x]; 

        __syncthreads(); 

        int sum =0; 

        if (reduce_number_once == 2)

        {

                for (int s = 256 >> 1; s > 0; s = s >> 1)

                {

                        if (s > hipThreadIdx_x) {

                                ldsData[hipThreadIdx_x] = ldsData[hipThreadIdx_x] +

                                                           ldsData[hipThreadIdx_x + s]; 

                        }

                        __syncthreads(); 

                }

                if (hipThreadIdx_x == 0)

                {

                        sum += ldsData[0]; 

                }

        }

        if (reduce_number_once == 4)

        {

            for (int s = 256 >> 2; s > 0; s = s >> 2)

            {

               if (s > hipThreadIdx_x) {

                    ldsData[hipThreadIdx_x] =  ldsData[hipThreadIdx_x] +

                                               ldsData[hipThreadIdx_x + s] +

                                              ldsData[hipThreadIdx_x + 2 * s] +

                                               ldsData[hipThreadIdx_x + 3 * s]; 

                }

            }

            if (hipThreadIdx_x == 0)

            {

               sum += ldsData[0]; 

            }

        }

        if （ （hipThreadIdx_x == 0 ） && sum >9999 ）

        {

                buf[hipBlockIdx_x] = sum; 

        }

}
```
編譯執行example-12a.cpp得到如下結果：
<table><tr><td bgcolor=“#707070”>
Reduce 2 once:  elapsed time:4.80159

Reduce 4 once:  elapsed time:2.817486
</td></tr></table>
每次讀4个LDS數據比每次讀2两个數據性能提高了70%。 Reduction可以看作是LDS讀寫延遲為主要性能瓶頸， 減少程序需要等待的LDS讀寫延遲將大大提高程序性能。  如果每次讀8个LDS數據，並對八個數據求和，咁需要8*8*8=512个元素。 讀者可以自己嘗試是否可以進一步提高性能。 

2.13  Padding Before Convolution
--------------------------------

### 2.13.1 1st Padding Kernel

在CNN的Convolution，如果Filter Size大於1x1，那麼Padding（填充）是一個非常重要的函數。 假設BatchSize=1024，Channels=1024，Height=Width=7，Padding=3X3，咁Padding之後嘅Height= Width=13x13，13x13=169 遠遠小於256，因此我們需要每個Threads讀寫超過一個Channel的數據。 下面嘅代碼BlockDim=Dim3 （ 256，1，1 ），GridDim= （【13 * 13/256】，  Channeles=1024, BatchSize=1024)。 代碼先計算輸入原始輸入數據的地址，如果在【7，7】的範圍內，那麼需要讀取顯存數據，否則設置為Padding Value== 0.
```
__global__ void

test_kernel(hipLaunchParm lp,

       float* __restrict__ bufA, float* __restrict__ bufB, int channels_once, int c, int h, int w,  int padding )

{

       int hw =  hipThreadIdx_x; 

       int cc = channels_once * hipBlockIdx_y; 

       int n = hipBlockIdx_z; 

       float org_data[16]; 

       if （hw <（ h * w ））

       {

              int hh = hw / w - padding; 

              int ww = hw % w - padding ; 

              for (int i = 0; i < 16; i++)

              {

org_data[i] = 0. 0f; 

              }

              int in_w = w - 2 * padding; 

              int in_h = h - 2 * padding; 

              bool needFetching = （ww >=0） &&ww < （in_w）） &&hh > = 0 ） && 

                                 （hh <（in_h））; 

              if (needFetching == true) {

                      int base = n * c * in_h * in_w + cc * in_h * in_w +

                            hh * in_w + ww; 

                      for (int i = 0; I < channels_once; i++)

                      {

                             org_data[i] = bufA[base + i * in_h * in_w]; 

                      }

              }

              int base = n * c * h * w + cc * h * w + hw; 

              for (int i = 0; I < channels_once; i++)

              {

                      bufB[base + i * h * w] = org_data[i]; 

              }

       }

}     
```
編譯並執行example-13a.cpp。 得到如下輸出結果：
<table><tr><td bgcolor=“#707070”>
Read/Write [1] Channels per thread:  elapsed time:29.635487

Read/Write [1] Channels per thread:  ==> Estimated Bandwidth 44  GB/s

Read/Write [2] Channels per thread:  elapsed time:21.011665

Read/Write [2] Channels per thread:  ==> Estimated Bandwidth 62  GB/s

Read/Write [4] Channels per thread:  elapsed time:14.498355

Read/Write [4] Channels per thread:  ==> Estimated Bandwidth 91  GB/s

Read/Write [8] Channels per thread:  elapsed time:11.157874

Read/Write [8] Channels per thread:  ==> Estimated Bandwidth 118  GB/s

Read/Write [16] Channels per thread:  elapsed time:9.165571

Read/Write [16] Channels per thread:  ==> Estimated Bandwidth 144  GB/s
</td></tr></table>
獲得嘅性能非常低，遠遠低於480 GB/s嘅理論極限。  使用“extractkernels example-13.exe”獲得編譯之後嘅GCN匯編程序， 發現以下奇怪代碼，總共包含16条buffer_ store_dword，同一條buffer_load_dword值令。 
<table><tr><td bgcolor=“#707070”>
               v_mov_b32_e32 v4, 0

              buffer_store_dword v4, off, s[0:3], s11 offset:64

              buffer_store_dword v4, off, s[0:3], s11 offset:56

              buffer_store_dword v4, off, s[0:3], s11 offset:48

              buffer_store_dword v4, off, s[0:3], s11 offset:44

              buffer_store_dword v4, off, s[0:3], s11 offset:36

              buffer_store_dword v4, off, s[0:3], s11 offset:32

              buffer_store_dword v4, off, s[0:3], s11 offset:20

              buffer_store_dword v4, off, s[0:3], s11 offset:16

              buffer_store_dword v4, off, s[0:3], s11 offset:8 

              buffer_store_dword v4, off, s[0:3], s11 offset:60

              buffer_store_dword v4, off, s[0:3], s11 offset:52

              buffer_store_dword v4, off, s[0:3], s11 offset:40

              buffer_store_dword v4, off, s[0:3], s11 offset:28

              buffer_store_dword v4, off, s[0:3], s11 offset:24

              buffer_store_dword v4, off, s[0:3], s11 offset:12

              buffer_store_dword v4, off, s[0:3], s11 offset:4 

              ...

              buffer_store_dword v4, v2, s[0:3], s11 offen

              buffer_load_dword v6, v2, s[0:3], s11 offen
</td></tr></table>
而同時我哋由以前嘅經驗獲知，HIPCC編譯器通常使用global_load_dword和global_store_dwor指令讀寫顯存數據。 16條寫顯存指令和程序中初始化“org_data[i] =0.0f” 最接近，為證實這個猜測修改為“org_data[i ] =0.1111f“，”v_mov_b32_e32 v4，0“變左”v_mov_b32_e32 v4，0x3de38da4“。 編譯器喺16个org_data嘅初始化為0之後，然後將org_data緩存到顯存，然後使用時再由顯存讀出，咁程序嘅效率大大降低。  通常只有喺寄存器超過256时，編譯器才需要使用顯存補充缺失嘅存儲器。 這個簡單程序顯然唔需要咁多寄存器。 HIPCC編譯器將嚿顯存稱為scratch （參考產生嘅GCN匯編程序中嘅scratch_hi同scratch_lo ）。 

         一個可能嘅猜測係循環變量channles_once作為輸入參數出現，而編譯器無法判別總嘅循環次數，唔可以判別需要org_data嘅實際大小，而將導致org_data被分配到scratch memory。 

### 2.13.2 Optimize Kernel to Remove Scratch Memory

Example-13b.cpp將所有整數參數轉為了常量，已嘗試是否會消除scratch memory。 

編譯並測試example-13b.cpp得到如下結果：
<table><tr><td bgcolor=“#707070”>
Read/Write [16] Channels per thread:  elapsed time:2.929695

Read/Write [16] Channels per thread:  ==> Estimated Bandwidth 450  GB/s
</td></tr></table>
本實例的每個線程讀寫16個Channels，完全有可能減低到4個Channels都能獲得非常接近的性能。 讀者可以試一試。 另外，讀者也可以嘗試讀取1，2，8個Channels的不同性能。 

2.14 BatchNorm
--------------

BatchNorm的基本原理參考： [https://blog.csdn.net/hjimce/article/details/50866313]（https://blog.csdn.net/hjimce/article/details/50866313）

根據基本原理，最簡單的實現需要讀取每個元素三次，第一次是計算平均值，第二次是計算平均方差，第三次是計算BN值，每次存儲讀取失敗需要重新向L2請求數據，這樣無法獲得最佳性能。 GCN架構嘅L1 Cache總共有256 Cachelines （ 16 KB /64 Bytes per CacheLine ），如果有 256个像素，BatchSize大於16，咁需要讀取嘅Cacheline將超過256。 平均方差和平均值可以用同一個循環完成，這樣可以減少一次L1 Cache嘅數據讀取。 再進一步，如果讀取嘅數據能夠保存喺VGPR中，咁淨係讀取一次L1 Cache即可。 總共設計了四個測試：

* Example-14a .cpp：使用了三次L1 Cache讀寫的方式，性能為22G Pixels/s。 
* Example-14b.cpp：使用了一次L1 Cache讀寫，把128个Batch嘅數據保存喺2个Threads中，性能為 15 G Pixels/s。 
* Example-14c .cpp：使用了一次L1 Cache讀寫，將128個Batch的數據保存在4個Threads中，性能為 32G Pixels/s。 
* Example-14d.cpp：使用了兩次L1 Cache 讀寫，第一次讀L1 Cache計算平均方差和平均值，第二次讀L1 Cache做（L1/L2可能是命中失敗） ，性能為30G Pixels/s。 

理論上方法14b同14c應該取得一樣嘅性能，因為兩個方法僅僅讀取一次L1 Cache，而且需要嘅VPGR數都係小於80。 而實際測試的結果完全不符合預期，方法14b和14c應該遠遠高於方法14d。 基本嘅猜測係HIPCC編譯器有不為人知嘅特性。 使用extractkernels工具產生 GCN assembly代碼，並進行分析：

* Example-14a.cpp：產生的代碼極為簡單，使用的VGPR數量低於16個;
* Example-14b .cpp：產生的代碼非常複雜，VGPR達到了最大值255，而且使用scratch memory來替代不足的VPGRs;
* Example-14c .cpp：代碼比較複雜， 使用超過105个VGPR，低於128个VGPR，冇使用scratch  memory；
* Example-14d.cpp：產生的代碼極為簡單，使用的VGPR數量低於16個;
*   所有四個實例中計算顯存地址部分冇任何優化，浪費咗大量計算指令;

HIPCC嘅寄存器分配同顯存地址計算嘅性能較差，在本例中無法獲得最佳性能，如需要獲得最佳性能，需要用匯編代碼進行優化。 

3     其他
========

Miopen提供了大量實例使用匯編指令提高性能，可以作為參考。 [https://github.com/adityaatluri/gemm-vega64]（https://github.com/adityaatluri/gemm-vega64）提供了inline assembly的方式簡化 GCN架構的HIP/OpenCL匯編程序，可以作為極好的參考。 

4 Convert Word to MarkDown
==========================

WORD to MD file

將WORD文件內容放入下面網站， 轉換為HTML

[https://wordhtml.com/](https://wordhtml.com/)

然後把HTML內容透過另外一個網站轉換為MarkDown

[https://tool.lu/markdown/](https://tool.lu/markdown/)Contents
