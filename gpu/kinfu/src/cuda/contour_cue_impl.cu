//@author zhangxaochen

// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <pcl/pcl_macros.h>
// 
// #include "internal.h"
 #include "device.hpp"
// 
// using namespace std;
// using namespace cv;

#include <contour_cue_impl.h>


namespace zc{
    using namespace pcl::device;

//@brief given *inOut* as input, inpaint the holes of *inOut* on GPU
//@param depthInOut, both input & output storage
template<typename T>
__global__ void 
inpaintGpuKernel(PtrStepSz<T> depthInOut){
    //int bid = blockIdx.x + blockIdx.y * gridDim.x;
    //int tid = bid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; //tid 计算对不对呢？√... 繁琐。弃用。参考 mergePointNormal
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //一个 tid 对应*一行*像素：
    if(tid > depthInOut.rows)
        return;
    
    T *row = depthInOut.ptr(tid);
    //左右有效-无效值边界值：
    T lValidBorderVal = 0,
        rValidBorderVal = 0;
    //对应下标：
    int lbIdx = -1,
        rbIdx = -1;

    for(size_t j = 0; j < depthInOut.cols - 1; j++){
        if(row[j] != 0 && row[j+1] == 0){ //j有效，j+1无效
            lbIdx = j;
            lValidBorderVal = row[j];

            //重置 rValidBorderVal： (其实未用)
            rValidBorderVal = 0;
        }
        else if(row[j] == 0 && row[j+1] != 0){ //j无效，j+1有效
            rbIdx = j+1;
            rValidBorderVal = row[j+1];

            //右边界触发修补, 向左回溯 (idx=0 极左位置无效，也能正确处理)
            //INP_V2 逻辑已改： 2015-9-30 11:28:50
#if INP_V2  //v2, 单边无效时*不*修补
            if (lValidBorderVal != 0){
#endif  //INP_V2
                T inpaintVal = max(int(lValidBorderVal), rValidBorderVal);
                for(int k = j; k > lbIdx; k--){
                    row[k] = inpaintVal;
                }
#if INP_V2
            }
#endif  //INP_V2

            //重置 lValidBorderVal: (有用，下面判定)
            lValidBorderVal = 0;
        }

#if INP_V1  //v1, 左/右单边无效也修补
        if(j+1 == depthInOut.cols - 1 && row[j+1] == 0 //dst.cols-1 极右位置，若无效，特别处理
            && lValidBorderVal != 0) //若 ==0，且极右无效，说明整行都为零，不处理
        {
            //此时必存在已更新但未使用的 lValidBorderVal：
            T inpaintVal = lValidBorderVal; //== max(lValidBorderVal, rValidBorderVal); since rValidBorderVal==0
            for(int k = j+1; k > lbIdx; k--){
                row[k] = inpaintVal;
            }
        }
#endif  //v1, 左/右单边无效也修补
    }//for-j
}//inpaintGpuKernel

void inpaintGpu(const DepthMap& src, DepthMap& dst){
    dst.create(src.rows(), src.cols());
    src.copyTo(dst);

    //dim3 block(480, 1);
    //dim3 grid = divUp(src.rows(), block.x);
    int block = 256;
    int grid = divUp(src.rows(), block);

    inpaintGpuKernel<ushort><<<grid, block>>>(dst);

    cudaSafeCall(cudaGetLastError());
}//inpaintGpu

#define CONT_V1 0
#if !CONT_V1
#define CONT_V2 1
#endif  //!CONT_V1

__global__ void
computeContoursKernel(const PtrStepSz<ushort> src, PtrStepSz<uchar> dst, int thresh){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= src.cols || y >= src.rows)
        return;

    dst.ptr(y)[x] = 0;
    ushort depVal = src.ptr(y)[x];
    int xleft = max(x-1, 0), xright = min(x+1, src.cols-1),
        ytop = max(y-1, 0), ydown = min(y+1, src.rows-1);

    bool isContour = false;
    for(int ix = xleft; ix <= xright; ix++){
        for(int iy = ytop; iy <= ydown; iy++){
            ushort neighbor = src.ptr(iy)[ix];
#if CONT_V1
            if(neighbor != 0                        //无效邻域不判定
                && neighbor - depVal > thresh)    // nbr - self, 表示物体轮廓应该更浅
            {
                dst.ptr(y)[x] = UCHAR_MAX;
                return;
            }
#elif CONT_V2
            if(neighbor == 0) //若邻域有无效值，则self不算做轮廓
                return;
            else if(neighbor - depVal > thresh){  // nbr - self, 表示物体轮廓应该更浅
                isContour = true;
            }
#endif
        }//for-iy
    }//for-ix

#if CONT_V2
    if(isContour)
        dst.ptr(y)[x] = UCHAR_MAX;
#endif  //CONT_V2

}//computeContoursKernel

void computeContours( const DepthMap& src, MaskMap& dst, int thresh /*= 50*/ ){
    dst.create(src.rows(), src.cols());

    dim3 block(32, 8);
    dim3 grid(divUp(src.cols(), block.x), divUp(src.rows(), block.y));

    computeContoursKernel<<<grid, block>>>(src, dst, thresh);

    cudaSafeCall(cudaGetLastError());
}//computeContours

//@brief gpu kernel function to generate the Contour-Correspondence-Candidates
//@param[in] angleThreshCos, MAX cosine of the angle threshold
//@注意 kernel 函数参数必须为 GPU 内存指针或对象拷贝，e.g., 必须为 float3 而非 float3&
__global__ void 
cccKernel(const float3 camPos, const PtrStep<float> vmap, const PtrStep<float> nmap, float angleThreshCos, PtrStepSz<uchar> outMask){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;
    //printf("### %d, %d\n", x, y);

    int cols = outMask.cols,
        rows = outMask.rows;

    if(!(x < cols && y < rows))
        return;

    outMask.ptr(y)[x] = 0;

    if(isnan(nmap.ptr(y)[x]) || isnan(vmap.ptr(y)[x])){
        //printf("\tisnan: %d, %d\n", x, y);
        return;
    }

    float3 n, vRay;
    n.x = nmap.ptr(y)[x];
    n.y = nmap.ptr(y + rows)[x];
    n.z = nmap.ptr(y + 2 * rows)[x];

    vRay.x = camPos.x - vmap.ptr(y)[x];
    vRay.y = camPos.y - vmap.ptr(y + rows)[x];
    vRay.z = camPos.z - vmap.ptr(y + 2 * rows)[x];

    double nMod = norm(n); //理论上恒等于1？
    double vRayMod = norm(vRay);
    //printf("@@@ %f, %f\n", nMod, vRayMod);

    double cosine = dot(n, vRay) / (vRayMod * nMod);
    if(abs(cosine) < angleThreshCos)
        outMask.ptr(y)[x] = UCHAR_MAX;
}//cccKernel

void contourCorrespCandidate(const float3 &camPos, const MapArr &vmap, const MapArr &nmap, int angleThresh, MaskMap &outMask ){
    int cols = vmap.cols();
    int rows = vmap.rows() / 3;
    
    outMask.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    const float angleThreshCos = cos(angleThresh * 3.14159265359f / 180.f);
    //printf("vmap, nmap shape: [%d, %d], [%d, %d]\n", vmap.rows(), vmap.cols(), nmap.rows(), nmap.cols()); //test OK
    cccKernel<<<grid, block>>>(camPos, vmap, nmap, angleThreshCos, outMask);

    sync();
    //cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
}//contourCorrespCandidate

__global__ void
testPclCudaKernel(PtrStepSz<ushort> o1, PtrStep<float> o2){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x<3 && y<2)
        printf("--testPclCudaKernel\n");
}//testPclCudaKernel

void testPclCuda(DepthMap &o1, MapArr &o2){
    int rows = 480,
        cols = 640;

    o1.create(rows, cols);
    o2.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    testPclCudaKernel<<<grid, block>>>(o1, o2);

    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
}//testPclCuda

__global__ void
transformVmapKernel(int rows, int cols, const PtrStep<float> vmap_src, const Mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst){
    int x = threadIdx.x + blockIdx.x * blockDim.x,
        y = threadIdx.y + blockIdx.y * blockDim.y;

    const float qnan = pcl::device::numeric_limits<float>::quiet_NaN();
    if(!(x < cols && y < rows))
        return;

    float3 vsrc, vdst = make_float3(qnan, qnan, qnan);
    vsrc.x = vmap_src.ptr(y)[x];

    if(!isnan(vsrc.x)){
        vsrc.y = vmap_src.ptr(y + rows)[x];
        vsrc.z = vmap_src.ptr(y + 2 * rows)[x];

        vdst = Rmat * vsrc + tvec;

        vmap_dst.ptr (y + rows)[x] = vdst.y;
        vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
    }

    //确实应放在这里！无论是否 isnan(vdst.x)
    vmap_dst.ptr(y)[x] = vdst.x;
}//transformVmapKernel

void transformVmap( const MapArr &vmap_src, const Mat33 &Rmat, const float3 &tvec, MapArr &vmap_dst ){
    int cols = vmap_src.cols(),
        rows = vmap_src.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    
    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

    transformVmapKernel<<<grid, block>>>(rows, cols, vmap_src, Rmat, tvec, vmap_dst);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}//transformVmap

}//namespace zc


// void zc::foo(){}
