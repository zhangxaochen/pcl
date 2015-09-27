#ifndef _CONTOUR_CUE_IMPLEMENTATION_
#define _CONTOUR_CUE_IMPLEMENTATION_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/pcl_macros.h>

using namespace std;
using namespace cv;

namespace zc{
//@author zhangxaochen
//@brief implementing the inpainting algorithm in paper *cvpr2015-contour-cue*
//@param src, input src depth map(opencv-Mat)
//@return dst, inpainted depth map
template<typename T> 
PCL_EXPORTS Mat inpaint(const Mat& src){
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

                //重置 rValidBorderVal：
                rValidBorderVal = 0;
            }
            else if(row[j] == 0 && row[j+1] != 0){ //j无效，j+1有效
                rbIdx = j+1;
                rValidBorderVal = row[j+1];

                //右边界触发修补, 向左回溯 (idx=0 极左位置无效，也能正确处理)
                T inpaintVal = max(lValidBorderVal, rValidBorderVal);
                for(int k = j; k > lbIdx; k--){
                    row[k] = inpaintVal;
                }

                //重置 lValidBorderVal:
                lValidBorderVal = 0;
            }

            if(j+1 == dst.cols - 1 && row[j+1] == 0 //dst.cols-1 极右位置，若无效，特别处理
                && lValidBorderVal != 0) //若 ==0，且极右无效，说明整行都为零，不处理
            {
                //此时必存在已更新但未使用的 lValidBorderVal：
                T inpaintVal = lValidBorderVal; //== max(lValidBorderVal, rValidBorderVal); since rValidBorderVal==0
                for(size_t k = j+1; k > lbIdx; k--){
                    row[k] = inpaintVal;
                }
            }
        }//for-j
    }//for-i
    return dst;
}//inpaint

}//namespace zc

#endif //_CONTOUR_CUE_IMPLEMENTATION_
