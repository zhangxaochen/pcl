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
    int tid = bid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; //tid ����Բ����أ�δ����
    //һ�� tid ��Ӧ*һ��*���أ�
    if(tid > inOut.rows)
        return;
    
    T *row = inOut.ptr(tid);
    //������Ч-��Чֵ�߽�ֵ��
    T lValidBorderVal = 0,
        rValidBorderVal = 0;
    //��Ӧ�±꣺
    int lbIdx = -1,
        rbIdx = -1;

    for(size_t j = 0; j < inOut.cols - 1; j++){
        if(row[j] != 0 && row[j+1] == 0){ //j��Ч��j+1��Ч
            lbIdx = j;
            lValidBorderVal = row[j];

            //���� rValidBorderVal�� (��ʵδ��)
            rValidBorderVal = 0;
        }
        else if(row[j] == 0 && row[j+1] != 0){ //j��Ч��j+1��Ч
            rbIdx = j+1;
            rValidBorderVal = row[j+1];

            //�ұ߽紥���޲�, ������� (idx=0 ����λ����Ч��Ҳ����ȷ����)
            //INP_V2 �߼��Ѹģ� 2015-9-30 11:28:50
#if INP_V2  //v2, ������Чʱ*��*�޲�
            if (lValidBorderVal != 0){
#endif  //INP_V2
                T inpaintVal = max(int(lValidBorderVal), rValidBorderVal);
                for(int k = j; k > lbIdx; k--){
                    row[k] = inpaintVal;
                }
#if INP_V2
            }
#endif  //INP_V2

            //���� lValidBorderVal: (���ã������ж�)
            lValidBorderVal = 0;
        }

#if INP_V1  //v1, ��/�ҵ�����ЧҲ�޲�
        if(j+1 == inOut.cols - 1 && row[j+1] == 0 //dst.cols-1 ����λ�ã�����Ч���ر���
            && lValidBorderVal != 0) //�� ==0���Ҽ�����Ч��˵�����ж�Ϊ�㣬������
        {
            //��ʱ�ش����Ѹ��µ�δʹ�õ� lValidBorderVal��
            T inpaintVal = lValidBorderVal; //== max(lValidBorderVal, rValidBorderVal); since rValidBorderVal==0
            for(int k = j+1; k > lbIdx; k--){
                row[k] = inpaintVal;
            }
        }
#endif  //v1, ��/�ҵ�����ЧҲ�޲�
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
            if(neighbor != 0                        //��Ч�����ж�
                && neighbor - depVal > thresh)    // nbr - self, ��ʾ��������Ӧ�ø�ǳ
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
