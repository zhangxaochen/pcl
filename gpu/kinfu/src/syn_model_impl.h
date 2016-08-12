//@author zhangxaochen  //2016-4-5 04:17:49

#ifndef _SYN_MODEL_IMPLEMENTATION_
#define _SYN_MODEL_IMPLEMENTATION_

//#include <pcl/gpu/kinfu/tsdf_volume.h> //只要 cu 中包含就报错, 所以 cu 不能包含此头文件
//  见: https://www.evernote.com/shard/s399/nl/67976577/8e16d9c1-d24b-45ca-95c6-8f1855ff64b6
#include <pcl/pcl_exports.h>

//#include <Eigen/src/Geometry/Transform.h> //Affine3f, etc.
//#include <Eigen/Geometry>

//#include "cuda/device.hpp"
#include "internal.h"

namespace zc{
using namespace pcl;
using namespace pcl::gpu;
using namespace pcl::device;
//using namespace cv;

using pcl::device::DepthMap;
using pcl::device::MapArr;
using pcl::gpu::DeviceArray2D;
using pcl::gpu::divUp;        
using pcl::gpu::PtrSz;
using pcl::gpu::PtrStep;
using pcl::gpu::PtrStepSz;
using pcl::gpu::DeviceArray;

typedef unsigned short ushort;
typedef unsigned char uchar;
typedef DeviceArray2D<uchar> MaskMap;


//@brief see *integrateTsdfVolume*, DEVICE version.
//@param[in] cloud_device, cloud in device memory
//@param[in] pcen, centroid point of *cloud_device*
//@param[in] cell_sz, size of a voxel cell
//@param[in] truncDist, truncation distance of TSDF volume
//@param[in] weight, preset weight of the TSDF of the single *cloud_device*, make it "confident"
//@param[out] tsdf_volume, the TSDF to be updated
void cloud2tsdf(const DeviceArray<float4> &cloud_device, const float3 &pcen, const float3 &cell_sz, float truncDist, int weight, PtrStep<short2> tsdf_volume);

//@brief DEVICE version, 参考 *integrateTsdfVolume*
//@param[in] rot_inv, R^-1, 相机姿态的逆矩阵, world->camera, g->c
//@param[in] trans, 相机在世界的位置, 是 c->g
//@param[out] dmat, 输出深度图, 此处还是GPU内存, 此时已有宽高信息, 所以接口没有 (w, h) 参数
PCL_EXPORTS void cloud2depth(DeviceArray<float4> cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, DepthMap &dmat);

//@brief 参数列表带 (w,h) 版本, 以防直接调用之前版本时 dmat 未分配内存
PCL_EXPORTS void cloud2depth(DeviceArray<float4> cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, int imWidth, int imHeight, DepthMap &dmat);

//@brief 参数列表不含外参(R,t), 意味着点云vmap是相机坐标系下的
//@param[in] vmap, 相机坐标系点云, 且是有序点云
PCL_EXPORTS void cloud2depth(DeviceArray2D<float> vmap, const Intr &intr, DepthMap &dmat);

//@brief 给定一个点云A, 以及一个可见深度图B, 生成这个A在深度图B上的可见子集C. 参数接口与 cloud2depth 类似, 但是必须新开一个函数, 因为此处 A.size 与 cloud2depth中的点集大小未必相同
//@param[in] cloud_device, GPU上, unorganized, *A*.
//@param[in] dmat, GPU内存上的深度图, *B*. 含窗口宽高信息.
//@param[out] outCloud, GPU上, *C*. bool indices 标记对应位置的取舍
PCL_EXPORTS void raycastCloudSubset(const DeviceArray<float4> &cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, const DepthMap &dmat, DeviceArray<int> &outIdx);

//@brief 一次迭代, 注意参数中带着(R,t)
//@param[in] (Rcurr, tcurr) 是 c->g, 因为 srcDmat 是深度图, 而非世界坐标系点云
//@param[out] poseBuf, 6DOF, 为什么不返回矩阵? 因为: cuda 函数, Mat33 仍不如 eigen-Matrix3frm 有用
PCL_EXPORTS void align2dmapsOnce(const DepthMap &srcDmat, const Intr &intr, const Mat33 &Rcurr, const float3 &tcurr, const DepthMap &dstDmat, const MapArr &dstGradu, const MapArr &dstGradv, float *poseBuf);

//+++++++++++++++下面都是一些GPU内存读写测试
PCL_EXPORTS void foo_in_syn_cu();
PCL_EXPORTS void foo_in_syn_cpp();

PCL_EXPORTS void test_write_gpu_mem_in_syn_cu();
}//namespace zc
#endif //_SYN_MODEL_IMPLEMENTATION_
