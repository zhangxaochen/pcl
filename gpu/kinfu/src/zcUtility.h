#ifndef _ZC_UTILITY_H_
#define _ZC_UTILITY_H_
#pragma once

#include <pcl/common/time.h>
#include <pcl/common/transforms.h> //transformPointCloud, etc.
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/gpu/kinfu/tsdf_volume.h> //ֻҪ cu �а����ͱ���, ���� cu ���ܰ�����ͷ�ļ�

#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#include "internal.h"

namespace zc{
using namespace cv;
using namespace Eigen;
using namespace pcl;
using namespace pcl::gpu;

using pcl::device::DepthMap;
using pcl::device::MapArr;
using pcl::gpu::DeviceArray2D;
using pcl::gpu::divUp;        
using pcl::gpu::PtrSz;
using pcl::gpu::PtrStep;
using pcl::gpu::PtrStepSz;
using pcl::gpu::DeviceArray;

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;

namespace test{
    //@brief ���� inpaint �㷨ʵ����ȷ�ԣ�1. CPU & GPU ���һ��; 2. GPU ����Ч
    void testInpaintImplCpuAndGpu(const DepthMap &src, bool debugDraw = false);

    //@brief ���� vmap �Ƿ���ȷ, ͨ�� nmap opencv ���ƹ۲�. ��ֹĳЩ vmap ����ת�� etc. ֮�����.
    //@param[in] vmap, vertex map
    //@param[in] winName, opencv imshow window name
    void testVmap(const MapArr &vmap, const char *winName);
}//namespace test

class PCL_EXPORTS ScopeTimeMicroSec : public StopWatch{
    std::string title_;
public:
    ScopeTimeMicroSec(const char *title=""):
      title_(title){}

      ~ScopeTimeMicroSec(){
          double val = this->getTimeMicros();
          std::cerr << title_ << " took " << val << " micro seconds.\n";
      }

      //@brief return execution time in micro-seconds
      //@issue&fix move function body to .cpp file to avoid http://stackoverflow.com/questions/11540962/tell-nvcc-to-not-preprocess-host-code-to-avoid-boost-compiler-redefinition
      //traceback: @sgf's-PC: http://codepad.org/3M0tgmrb
      double getTimeMicros();

};//class ScopeTimeMillis

//@brief normal-map(SOA�ڴ�ṹ)ת��Ϊopencv-Mat, ���ڵ�����ʾ. ���ԸĽ����̼�: http://www.evernote.com/l/AY8AtmAVfmZBvKb96aRiC4VW_wQp59sAUzE/
//@param nmap, ����ͼ, ����Ϊ SOA �ڴ�ṹ(��Ϊ�õ�device::convert)
//@param debugDraw, �Ƿ� imshow
//@return cv::Mat of type CV_32FC3
PCL_EXPORTS Mat nmap2rgb(const MapArr &nmap, bool debugDraw = false);

//@brief alias of *nmap2rgb*
PCL_EXPORTS inline Mat renderNmap(const MapArr &nmap, bool debugDraw = false){
    return nmap2rgb(nmap, debugDraw);
}

//+++++++++++++++�� syn_model_impl.h Ǩ�ƹ���, һЩ�ϲ�ṹ(Eigen) ���ܱ������� .cu �ļ���  //2016-4-7 15:59:49
//@brief wrapper of device version of *cloud2tsdf*, HOST version for convinience
//@param[in] cloud, CPU�ϵĵ���
//@param[in] weight, ����Ԥ�õ� TSDF ʱ��Ȩ��
//@param[out] tsdf_volume, CPU�ϵ�TSDFģ��
PCL_EXPORTS void cloud2tsdf(const PointCloud<PointXYZ> &cloud, int weight, TsdfVolume &tsdf_volume);

//@brief ����һ���ӵ� & ����, ������ӵ���͸��������(����Ͷ��)�ĵ���(organized)
//@param[in] cloud, ����, һ�� un-organized
//@param[in] pose, ������ {R, t}, ����������ת���������ϵ
//@param[in] intr, ����ڲ� {fx, fy, cx, cy}, �� �������ϵ�� ת�� �������ͼ
//@param[in] (imWidth, imHeight), ��ת��ͼ��Ŀ��, e.g. 640x480
//@param[out] depOut, ע��: 1. depOut ����ʱ��û�����ڴ�, ���Բ���Ҫ(imWidth, imHeight);
//                          2. ������, ��ΪҪ�޸� Mat-metaͷ���ɵ��� + �ӵ����ɵ����ͼ;
PCL_EXPORTS void cloud2depth(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, Mat &depOut);
//@brief similar to it's overload, except the [out] param
//@param[out] depOutDevice, depth map on GPU
PCL_EXPORTS void cloud2depth(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, DeviceArray2D<ushort> &depOutDevice);

//@brief ������ cloud2depthKernel �������, GPU���� bug, ���� CPU �ϲ���:
//@param[in] cloud ��(m)�߶�
//@param[out] depOut ����(mm)�߶�
PCL_EXPORTS void cloud2depthCPU(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, Mat &depOut);

PCL_EXPORTS void raycastCloudSubset(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, const Mat &dmat, PointCloud<PointXYZ> &outCloud);

//@brief ��׼�������ͼ, �����б��� (R,t), ���� srcDmat ���������ϵΪ��������ϵ
void align2dmapsGPU(const DepthMap &srcDmat, const Intr &intr, const DepthMap &dstDmat, const MapArr &dstGradu, const MapArr &dstGradv);

void align2dmapsCPU(const Mat &srcDmat, const Intr &intr, const Mat &dstDmat, const Mat &dstGradu, const Mat &dstGradv);

//@brief mean(sum(|m1-m2|^2)) �������(������), ��Ӧ���ض���Ч�ż���, CPU �汾
float twoDmatsMSE(Mat m1, Mat m2);
//@brief GPU �汾
float twoDmatsMSE(const DepthMap &m1_device, const DepthMap &m2_device);

}//namespace zc

#endif //_ZC_UTILITY_H_
