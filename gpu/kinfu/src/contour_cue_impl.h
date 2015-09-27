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
    //CV_Assert(src.depth() == CV_16UC1); //��Ҫ����Ϊ 8u / 16u ������
    Mat dst = src.clone(); //copy data true
    for(size_t i = 0; i < dst.rows; i++){
        //����ɨ�裺
        T *row = dst.ptr<T>(i);
        //������Ч-��Чֵ�߽�ֵ��
        T lValidBorderVal = 0,
            rValidBorderVal = 0;
        //��Ӧ�±꣺
        int lbIdx = -1,
            rbIdx = -1;
        for(size_t j = 0; j < dst.cols - 1; j++){
            if(row[j] != 0 && row[j+1] == 0){ //j��Ч��j+1��Ч
                lbIdx = j;
                lValidBorderVal = row[j];

                //���� rValidBorderVal��
                rValidBorderVal = 0;
            }
            else if(row[j] == 0 && row[j+1] != 0){ //j��Ч��j+1��Ч
                rbIdx = j+1;
                rValidBorderVal = row[j+1];

                //�ұ߽紥���޲�, ������� (idx=0 ����λ����Ч��Ҳ����ȷ����)
                T inpaintVal = max(lValidBorderVal, rValidBorderVal);
                for(int k = j; k > lbIdx; k--){
                    row[k] = inpaintVal;
                }

                //���� lValidBorderVal:
                lValidBorderVal = 0;
            }

            if(j+1 == dst.cols - 1 && row[j+1] == 0 //dst.cols-1 ����λ�ã�����Ч���ر���
                && lValidBorderVal != 0) //�� ==0���Ҽ�����Ч��˵�����ж�Ϊ�㣬������
            {
                //��ʱ�ش����Ѹ��µ�δʹ�õ� lValidBorderVal��
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
