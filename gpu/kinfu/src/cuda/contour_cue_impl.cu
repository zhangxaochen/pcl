//@author zhangxaochen

// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <pcl/pcl_macros.h>
// 
// #include "internal.h"
// #include "device.hpp"
// 
// using namespace std;
// using namespace cv;

#include <contour_cue_impl.h>

namespace zc{

//@brief given *inOut* as input, inpaint the holes of *inOut* on GPU
//@param inOut, both input & output storage
template<typename T>
__global__ void 
inpaintGpuKernel(PtrStepSz<T> inOut){
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = bid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; //tid 计算对不对呢？未测试
    //一个 tid 对应*一行*像素：
    if(tid > inOut.rows)
        return;
    
    T *row = inOut.ptr(tid);
    //左右有效-无效值边界值：
    T lValidBorderVal = 0,
        rValidBorderVal = 0;
    //对应下标：
    int lbIdx = -1,
        rbIdx = -1;

    for(size_t j = 0; j < inOut.cols - 1; j++){
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
        if(j+1 == inOut.cols - 1 && row[j+1] == 0 //dst.cols-1 极右位置，若无效，特别处理
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

    dim3 block(480, 1);
    dim3 grid = divUp(src.rows(), block.x);

    inpaintGpuKernel<ushort><<<grid, block>>>(dst);

    cudaSafeCall(cudaGetLastError());
}//inpaintGpu

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
    for(int ix = xleft; ix <= xright; ix++){
        for(int iy = ytop; iy <= ydown; iy++){
            ushort neighbor = src.ptr(iy)[ix];
            if(neighbor != 0                        //无效邻域不判定
                && neighbor - depVal > thresh)    // nbr - self, 表示物体轮廓应该更浅
            {
                dst.ptr(y)[x] = UCHAR_MAX;
                return;
            }
        }
    }
}//computeContoursKernel

void computeContours( const DepthMap& src, MaskMap& dst, int thresh /*= 50*/ ){
    dst.create(src.rows(), src.cols());

    dim3 block(32, 8);
    dim3 grid(divUp(src.cols(), block.x), divUp(src.rows(), block.y));

    computeContoursKernel<<<grid, block>>>(src, dst, thresh);

    cudaSafeCall(cudaGetLastError());
}//computeContours

}//namespace zc


// void zc::foo(){}
