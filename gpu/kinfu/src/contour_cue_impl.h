#ifndef _CONTOUR_CUE_IMPLEMENTATION_
#define _CONTOUR_CUE_IMPLEMENTATION_

#include <iostream>
#include <opencv2/opencv.hpp>
//#include <pcl/pcl_macros.h> //与 .cu 冲突 "error C2375: 'log2f' : redefinition; different linkage"
#include <pcl/pcl_exports.h>

#include "internal.h"
// #include <pcl/gpu/containers/device_array.h>

using namespace std;

namespace zc{
    using namespace cv;
    using namespace pcl;
    using namespace pcl::device;

#define INP_V1 0
#if !INP_V1
#define INP_V2 1
#endif  //!INP_V1

//@author zhangxaochen
//@brief implementing the INPAINTING algorithm in paper *cvpr2015-contour-cue*
//@param[in] src, input src depth map(opencv-Mat)
//@return dst, inpainted depth map
template<typename T> 
PCL_EXPORTS Mat inpaintCpu(const Mat& src){
    //CV_Assert(src.depth() == CV_16UC1); //不要。因为 8u / 16u 都可能
    Mat dst = src.clone(); //copy data true
    for(size_t i = 0; i < dst.rows; i++){
        //逐行扫描：
        T *row = dst.ptr<T>(i);
        //左右有效-无效值边界值：
        T lValidBorderVal = 0,
            rValidBorderVal = 0;
        //对应下标：
        int lbIdx = -1,
            rbIdx = -1;
        for(size_t j = 0; j < dst.cols - 1; j++){
            if(row[j] != 0 && row[j+1] == 0){ //j有效，j+1无效
                lbIdx = j;
                lValidBorderVal = row[j];

                //重置 rValidBorderVal： (其实未用)
                rValidBorderVal = 0;
            }
            else if(row[j] == 0 && row[j+1] != 0){ //j无效，j+1有效
                rbIdx = j+1;
                rValidBorderVal = row[j+1];

                //右边界触发修补, 向左回溯 (idx=0 极左位置无效，也能正确处理); 
                //INP_V2 逻辑已改： 2015-9-30 11:28:50
#if INP_V2  //v2, 单边无效时*不*修补
                if (lValidBorderVal != 0){
#endif  //INP_V2
                    T inpaintVal = max(lValidBorderVal, rValidBorderVal);
                    for(int k = j; k > lbIdx; k--){ //注意是 int k, 否则若 size_t k, (k>-1) 无效出错！
                        row[k] = inpaintVal;
                    }
#if INP_V2
                }
#endif  //INP_V2

                //重置 lValidBorderVal: (有用，下面判定)
                lValidBorderVal = 0;
            }

#if INP_V1  //v1, 左/右单边无效也修补
            if(j+1 == dst.cols - 1 && row[j+1] == 0 //dst.cols-1 极右位置，若无效，特别处理
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
    }//for-i
    return dst;
}//inpaintCpu


using pcl::device::DepthMap;
using pcl::device::MapArr;
using pcl::gpu::DeviceArray2D;
using pcl::gpu::divUp;        
using pcl::gpu::PtrSz;
using pcl::gpu::PtrStep;
using pcl::gpu::PtrStepSz;

typedef unsigned short ushort;
typedef unsigned char uchar;
typedef DeviceArray2D<uchar> MaskMap;

//@author zhangxaochen
//@brief implementing the CONTOUR-DETECTION algorithm in paper *cvpr2015-contour-cue*
//@param[in] src, depth map on gpu, type==ushort
//@param[out] dst, mask map on gpu, type==uchar
//@param[in] thresh, discontinunity threshold δ in paper, in (mm)
void computeContours(const DepthMap& src, MaskMap& dst, int thresh = 50);

// template<typename T>
// void inpaintGpu(const DeviceArray2D<T>& src, DeviceArray2D<T>& dst);
// 
// //specialization for *ushort*
// template<>
// inpaintGpu<ushort>(const DeviceArray2D<ushort>& src, DeviceArray2D<ushort>& dst)

//@brief similar to *inpaintCpu*, but not a func. template
void inpaintGpu(const DepthMap& src, DepthMap& dst);

//@brief generate the contour correspondence candidates (a MaskMap repr.) using the "tangency property": "for all points along the contour generator, the normal is orthogonal to the view ray."
//@param[in] camPos, the camera coords in global reference frame
//@param[in] vmap, the vertex map in *GLOBAL FRAME*, to compute "view ray" with *camPos*
//@param[in] nmap, the normal map in *GLOBAL FRAME*
//@param[in] angleThresh, angle threshold of the "tangency property" in *degree*.
//@param[out] outMask, contour mask
void contourCorrespCandidate(const float3 &camPos, const MapArr &vmap, const MapArr &nmap, int angleThresh, MaskMap &outMask);

//@brief test basic operations with pcl-cuda, since error occured when impl. *contourCorrespCandidate*; see: http://www.evernote.com/l/AY82PvIZaq9MAYJsLwSy_b43fCV0DwdcDI0/
void testPclCuda(DepthMap &o1, MapArr &o2);

//@brief Perform affine transform of vmap (part of @tranformMaps)
//@param[in] vmap_src source vertex map
//@param[in] Rmat rotation mat
//@param[in] tvec translation
//@param[out] vmap_dst destination vertex map
void transformVmap(const MapArr &vmap_src, const Mat33 &Rmat, const float3 &tvec, MapArr &vmap_dst);

}//namespace zc

#endif //_CONTOUR_CUE_IMPLEMENTATION_
