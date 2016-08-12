//@author zhangxaochen  //2016-4-5 04:17:49

#ifndef _SYN_MODEL_IMPLEMENTATION_
#define _SYN_MODEL_IMPLEMENTATION_

//#include <pcl/gpu/kinfu/tsdf_volume.h> //ֻҪ cu �а����ͱ���, ���� cu ���ܰ�����ͷ�ļ�
//  ��: https://www.evernote.com/shard/s399/nl/67976577/8e16d9c1-d24b-45ca-95c6-8f1855ff64b6
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

//@brief DEVICE version, �ο� *integrateTsdfVolume*
//@param[in] rot_inv, R^-1, �����̬�������, world->camera, g->c
//@param[in] trans, ����������λ��, �� c->g
//@param[out] dmat, ������ͼ, �˴�����GPU�ڴ�, ��ʱ���п����Ϣ, ���Խӿ�û�� (w, h) ����
PCL_EXPORTS void cloud2depth(DeviceArray<float4> cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, DepthMap &dmat);

//@brief �����б�� (w,h) �汾, �Է�ֱ�ӵ���֮ǰ�汾ʱ dmat δ�����ڴ�
PCL_EXPORTS void cloud2depth(DeviceArray<float4> cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, int imWidth, int imHeight, DepthMap &dmat);

//@brief �����б������(R,t), ��ζ�ŵ���vmap���������ϵ�µ�
//@param[in] vmap, �������ϵ����, �����������
PCL_EXPORTS void cloud2depth(DeviceArray2D<float> vmap, const Intr &intr, DepthMap &dmat);

//@brief ����һ������A, �Լ�һ���ɼ����ͼB, �������A�����ͼB�ϵĿɼ��Ӽ�C. �����ӿ��� cloud2depth ����, ���Ǳ����¿�һ������, ��Ϊ�˴� A.size �� cloud2depth�еĵ㼯��Сδ����ͬ
//@param[in] cloud_device, GPU��, unorganized, *A*.
//@param[in] dmat, GPU�ڴ��ϵ����ͼ, *B*. �����ڿ����Ϣ.
//@param[out] outCloud, GPU��, *C*. bool indices ��Ƕ�Ӧλ�õ�ȡ��
PCL_EXPORTS void raycastCloudSubset(const DeviceArray<float4> &cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, const DepthMap &dmat, DeviceArray<int> &outIdx);

//@brief һ�ε���, ע������д���(R,t)
//@param[in] (Rcurr, tcurr) �� c->g, ��Ϊ srcDmat �����ͼ, ������������ϵ����
//@param[out] poseBuf, 6DOF, Ϊʲô�����ؾ���? ��Ϊ: cuda ����, Mat33 �Բ��� eigen-Matrix3frm ����
PCL_EXPORTS void align2dmapsOnce(const DepthMap &srcDmat, const Intr &intr, const Mat33 &Rcurr, const float3 &tcurr, const DepthMap &dstDmat, const MapArr &dstGradu, const MapArr &dstGradv, float *poseBuf);

//+++++++++++++++���涼��һЩGPU�ڴ��д����
PCL_EXPORTS void foo_in_syn_cu();
PCL_EXPORTS void foo_in_syn_cpp();

PCL_EXPORTS void test_write_gpu_mem_in_syn_cu();
}//namespace zc
#endif //_SYN_MODEL_IMPLEMENTATION_
