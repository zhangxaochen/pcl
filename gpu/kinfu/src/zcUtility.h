#ifndef _ZC_UTILITY_H_
#define _ZC_UTILITY_H_
#pragma once

#include <pcl/common/time.h>
#include <pcl/common/transforms.h> //transformPointCloud, etc.
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/gpu/kinfu/tsdf_volume.h> //只要 cu 中包含就报错, 所以 cu 不能包含此头文件

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
    //@brief 测试 inpaint 算法实现正确性：1. CPU & GPU 结果一致; 2. GPU 更高效
    void testInpaintImplCpuAndGpu(const DepthMap &src, bool debugDraw = false);

    //@brief 测试 vmap 是否正确, 通过 nmap opencv 绘制观察. 防止某些 vmap 坐标转换 etc. 之后出错.
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

//@brief normal-map(SOA内存结构)转换为opencv-Mat, 用于调试显示. 策略改进过程见: http://www.evernote.com/l/AY8AtmAVfmZBvKb96aRiC4VW_wQp59sAUzE/
//@param nmap, 法向图, 必须为 SOA 内存结构(因为用到device::convert)
//@param debugDraw, 是否 imshow
//@return cv::Mat of type CV_32FC3
PCL_EXPORTS Mat nmap2rgb(const MapArr &nmap, bool debugDraw = false);

//@brief alias of *nmap2rgb*
PCL_EXPORTS inline Mat renderNmap(const MapArr &nmap, bool debugDraw = false){
    return nmap2rgb(nmap, debugDraw);
}

//+++++++++++++++从 syn_model_impl.h 迁移过来, 一些上层结构(Eigen) 不能被包含到 .cu 文件中  //2016-4-7 15:59:49
//@brief wrapper of device version of *cloud2tsdf*, HOST version for convinience
//@param[in] cloud, CPU上的点云
//@param[in] weight, 点云预置到 TSDF 时的权重
//@param[out] tsdf_volume, CPU上的TSDF模型
PCL_EXPORTS void cloud2tsdf(const PointCloud<PointXYZ> &cloud, int weight, TsdfVolume &tsdf_volume);

//@brief 给定一个视点 & 点云, 输出此视点下透视消隐后(光线投射)的点云(organized)
//@param[in] cloud, 点云, 一般 un-organized
//@param[in] pose, 相机外参 {R, t}, 将点云坐标转到相机坐标系
//@param[in] intr, 相机内参 {fx, fy, cx, cy}, 将 相机坐标系点 转到 像素深度图
//@param[in] (imWidth, imHeight), 所转到图像的宽高, e.g. 640x480
//@param[out] depOut, 注意: 1. depOut 传入时还没分配内存, 所以才需要(imWidth, imHeight);
//                          2. 是引用, 因为要修改 Mat-meta头！由点云 + 视点生成的深度图;
PCL_EXPORTS void cloud2depth(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, Mat &depOut);
//@brief similar to it's overload, except the [out] param
//@param[out] depOutDevice, depth map on GPU
PCL_EXPORTS void cloud2depth(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, DeviceArray2D<ushort> &depOutDevice);

//@brief 基本从 cloud2depthKernel 搬过来的, GPU上有 bug, 先在 CPU 上测试:
//@param[in] cloud 米(m)尺度
//@param[out] depOut 毫米(mm)尺度
PCL_EXPORTS void cloud2depthCPU(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, Mat &depOut);

PCL_EXPORTS void raycastCloudSubset(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, const Mat &dmat, PointCloud<PointXYZ> &outCloud);

//@brief 配准两幅深度图, 参数列表不带 (R,t), 假设 srcDmat 的相机坐标系为世界坐标系
void align2dmapsGPU(const DepthMap &srcDmat, const Intr &intr, const DepthMap &dstDmat, const MapArr &dstGradu, const MapArr &dstGradv);

void align2dmapsCPU(const Mat &srcDmat, const Intr &intr, const Mat &dstDmat, const Mat &dstGradu, const Mat &dstGradv);

//@brief mean(sum(|m1-m2|^2)) 均方误差(不开方), 对应像素都有效才计入, CPU 版本
float twoDmatsMSE(Mat m1, Mat m2);
//@brief GPU 版本
float twoDmatsMSE(const DepthMap &m1_device, const DepthMap &m2_device);

}//namespace zc

#endif //_ZC_UTILITY_H_
