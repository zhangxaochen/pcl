//@author zhangxaochen  //2016-4-5 04:17:49


#include <iostream>
//#include <math.h>
#include <stdio.h>
#include <float.h>

//#include <Eigen/Core>

// #include <pcl/common/centroid.h>
// #include <pcl/point_cloud.h>
//#include <pcl/point_types.h> //��eigen, ��Ҫ���� cu
//#include <pcl/pcl_macros.h>

#include "internal.h"

#include "syn_model_impl.h" //�� eigen
#include "device.hpp"

#define M_PI       3.14159265358979323846

namespace zc{
using namespace std;

using namespace pcl;
using namespace pcl::gpu;
using namespace pcl::device;

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

//@ref: ������ http://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda //2016-6-30 15:13:24
__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(val));
    }
    return __int_as_float(old);
}

//@brief �� atomicMaxf, �� '>' Ϊ '<'
__device__ float atomicMinf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(val));
    }
    return __int_as_float(old);
}

#pragma region //utils copy from @ray_caster.cu
__device__ __forceinline__ bool
checkInds (const int3& g)
{
    return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < VOLUME_X && g.y < VOLUME_Y && g.z < VOLUME_Z);
}//checkInds

__device__ __forceinline__ float
readTsdf (PtrStep<short2> &volume, int x, int y, int z)
{
    return unpack_tsdf (volume.ptr (VOLUME_Y * z + y)[x]);
}//readTsdf

__device__ __forceinline__ int3
getVoxel (float3 point, float3 cell_size)
{
    int vx = __float2int_rd (point.x / cell_size.x);        // round to negative infinity
    int vy = __float2int_rd (point.y / cell_size.y);
    int vz = __float2int_rd (point.z / cell_size.z);

    return make_int3 (vx, vy, vz);
}//getVoxel
#pragma endregion

__device__ int tsdfKrnlCnt = 0;

//@brief ���޽ṹ����, ��GPU�ϼ��� tsdf. ������ initializeVolume / tsdf23@tsdf_volume.cu
//@param[in] cloud_device, GPU�ϵĵ���
//@param[in] pcen, centroid point of *cloud_device*, (meter)
//@param[in] cell_size, ������������ߴ�(meter)
//@param[in] tuncDist, tsdf truncation distance in meters
//@param[in] weight, ����תΪtsdfʱ������Ҫ���ߵ�Ȩ�� 
//@param[out] tsdf_volume, GPU�ϵ�TSDFģ��
__global__ void
cloud2tsdfKernel(const PtrSz<float4> cloud_device, float3 pcen, float3 cell_size, float tuncDist, int weight, PtrStep<short2> tsdf_volume){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; //�޽ṹ, ��һά
    //y = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx >= cloud_device.size){ //cloud û��ô���
        return;
    }

    float4 pt4 = cloud_device.data[idx]; //pt(i)
    float3 pi;
    pi.x = pt4.x; 
    pi.y = pt4.y;
    pi.z = pt4.z;

    int3 g = getVoxel(pi, cell_size); //a voxel grid (x,y,z) index
    //���� @interpolateTrilineary ���߽��ж�:
    if(g.x <= 0 || g.x >= VOLUME_X - 1 ||
        g.y <= 0 || g.y >= VOLUME_Y - 1 ||
        g.z <= 0 || g.z >= VOLUME_Z - 1)
        return;

    //�������ĵ�����, ����ߴ�
    float3 vi;
    vi.x = (g.x + 0.5f) * cell_size.x;
    vi.y = (g.y + 0.5f) * cell_size.y;
    vi.z = (g.z + 0.5f) * cell_size.z;

    //@ray_caster.cu ��������(����һ��)��Ŀ�ĺ��ڣ� ԭ�򲻶�, �ݲ���
    //g.x = (point.x < vx) ? (g.x - 1) : g.x;
    //g.y = (point.y < vy) ? (g.y - 1) : g.y;
    //g.z = (point.z < vz) ? (g.z - 1) : g.z;

    float3 pipc = pi - pcen; //pt(i)-pcen

//     pipc.x = pi.x - pcen.x;
//     pipc.y = pi.y - pcen.y;
//     pipc.z = pi.z - pcen.z;

    //rsqrtf: �� 1/norm(ptic)
    //����http://datamining.xmu.edu.cn/documentation/cuda4.1/group__CUDA__MATH__SINGLE_g5a9bc318028131cfd13d10abfae1ae13.html
    float pipc_norm_inv = rsqrtf(dot(pipc, pipc));

    //+++++++++++++++1. ��Vi��
    //������ (vi-pi) �� (pi-pcen) �ϵ�ͶӰ������Ϊ sdf
    //float sdf = ((vx - pt.x) * ptic.x + (vy - pt.y) * ptic.y + (yz - pt.z) * ptic.z) * ptic_norm_inv;
    float sdf = device::dot(vi - pi, pipc) * pipc_norm_inv;
    float tsdfVal = fmax(-1.f, fmin(1.f, sdf / tuncDist)); //�����Բ��ԣ�

#if 01 //��λV1
    int elem_step = VOLUME_Y * tsdf_volume.step / sizeof(short2); //һ��xy��ռ�õ�Ԫ����, ��ʵ�Ͳ�� 512*512 //��ʱ����
    short2 *pos = tsdf_volume.ptr(g.y) + g.x; //��λ�Բ��ԣ�
    pos += elem_step * g.z;

    //pos = tsdf_volume.ptr(); //���Խ�д��tsdf [0]λ��
    //pos->x = 1;
#elif 1 //���� readTSDF ��λ��ʽV2�� //������Ӧ����V1��ͬ, ����: XYz+Xy+x
    short2 *pos = tsdf_volume.ptr (VOLUME_Y * g.z + g.y) + g.x;
#endif //���ֶ�λ��ʽ
    //��ʵ��������� if��
    //if(g.x > 0 && g.x < VOLUME_X - 1 &&
    //    g.y > 0 && g.y < VOLUME_Y - 1 &&
    //    g.z > 0 && g.z < VOLUME_Z - 1)
    //{
    //atomicAdd(&tsdfKrnlCnt, 1); //����ȷʵ==cloud_device.size()
    //printf("tsdfKrnlCnt: %d\n", tsdfKrnlCnt); //��Ҫkernel��printȫ��device����
    float F_old = 9527;
    int W_old = 521;

    unpack_tsdf(*pos, F_old, W_old);
    if(W_old == 0){
        //printf�����, �����:
        //printf("0-grid: %d, %d, %d; F: %f, W: %d\n", g.x, g.y, g.z, F, W);
        atomicAdd(&tsdfKrnlCnt, 1);
        //��־: �����治 pack_tsdf, �˴���ȻtsdfKrnlCnt==cloud_device.size(), ��: 52666
        //������� pack_tsdf, tsdfKrnlCnt ����, ��: 10240
    }
    //printf("grid: %d, %d, %d; F: %f, W: %d\n", g.x, g.y, g.z, tsdfVal, weight);
    //printf("grid: %d, %d, %d\n", g.x, g.y, g.z);
    //printf("---------------\n");
    //���� W_old�Ƿ�Ϊ��, ֱ�Ӽ�Ȩ: //weight ��Ȩ�ᵼ�³����˹������趨, �Ȳ���
    tsdfVal = (tsdfVal * weight + F_old * W_old) / (weight + W_old);
    weight = (weight + W_old) % 128; //�ο� MAX_WEIGHT=128 @tsdf_volume.cu
    pack_tsdf(tsdfVal, weight, *pos);
    //}

    //+++++++++++++++2. ��26����Vk��
    for(int ix = g.x - 1; ix <= g.x + 1; ix++){
        for(int iy = g.y - 1; iy <= g.y + 1; iy++){
            for(int iz = g.z - 1; iz <= g.z + 1; iz++){
                //pos = tsdf_volume.ptr(iy)[ix] + elem_step * iz; //��, '+'����
                pos = tsdf_volume.ptr (VOLUME_Y * iz + iy) + ix;
                float tsdf_k_old;
                int wt_k_old;
                unpack_tsdf(*pos, tsdf_k_old, wt_k_old);
                //if(0 != wt_k) //˵���ѳ�ʼ����, ���� //��! �ѳ�ʼ��Ҳ��Ҫ����
                //    continue;

                float3 vk;
                vk.x = (ix + 0.5f) * cell_size.x;
                vk.y = (iy + 0.5f) * cell_size.y;
                vk.z = (iz + 0.5f) * cell_size.z;

                //������ (Vk-Pi), (Pi-pcen)�ļн�, <45���>135����, Ҳ����tsdf
                float3 vkpi = vk - pi;
                float vkpi_norm_inv = rsqrtf(dot(vkpi, vkpi));

                //float vkpi_dot_pipc = dot(vkpi, pipc);
                float vkpi_proj_pipc = dot(vkpi, pipc) * pipc_norm_inv ;
                float angle = acosf(vkpi_proj_pipc * vkpi_norm_inv); //���Ҷ���, radians
                float angDeg = angle * 180 / M_PI;
                if(angDeg < 45 || angDeg > 135){
                    float sdf = vkpi_proj_pipc; //�����������и�
                    float tsdfVal = fmax(-1.f, fmin(1.f, sdf / tuncDist));
                    //��Ȩ��
                    tsdfVal = (tsdfVal * weight + tsdf_k_old * wt_k_old) / (weight+wt_k_old);
                    weight = (weight + wt_k_old)%256; //����255
                    pack_tsdf(tsdfVal, weight, *pos);
                }
            }//for-iz
        }//for-iy
    }//for-ix
}//cloud2tsdfKernel

void cloud2tsdf(const DeviceArray<float4> &cloud_device, const float3 &pcen, const float3 &cell_sz, float truncDist, int weight, PtrStep<short2> tsdf_volume){
//void cloud2tsdf(DeviceArray<PointXYZ> cloud_device, PtrStep<short2> tsdf_volume){
    //grid & block shape ������, �� 512^2=262144, ���ڵ��Ƶ�������
    //dim3 block(256, 1);
    //dim3 grid(divUp(512, block.x), divUp(512, block.y));
    //dim3 grid(divUp(cloud_device.size(), block.x), 1);
    int block = 256,
        //grid = 512;
        grid = divUp(cloud_device.size(), block);
    cloud2tsdfKernel<<<grid, block>>>(cloud_device, pcen, cell_sz, truncDist, weight, tsdf_volume);

    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());

    //tsdfKrnlCnt
    int tsdfKrnlCntHost;
//     cudaMemcpy(&tsdfKrnlCntHost, &tsdfKrnlCnt, sizeof(int), cudaMemcpyDeviceToHost); //��
//     cout<<"cudaMemcpy, tsdfKrnlCntHost: "<<tsdfKrnlCntHost<<endl;

    cudaMemcpyFromSymbol(&tsdfKrnlCntHost, tsdfKrnlCnt, sizeof(int)); //��
    cout<<"cudaMemcpyFromSymbol, tsdfKrnlCntHost: "<<tsdfKrnlCntHost<<endl;

}//cloud2tsdf

//@brief GPU impl. kernel
//@param[in] ptSz, ���С����ʵ��ָ�����С��e.g., ptSz=2������2x2����, ptSz=3������3x3���������ƽ�ϡ�裬������ptSz�������Զ�㴩͸�Ͻ�����
//@param[out] dmat, ����Ͷ��(����)��õ������ͼ
__global__ void
//cloud2depthKernel(const PtrSz<float4> &cloud_device, const Mat33 rot_inv, const float3 trans, const Intr intr, PtrStepSz<ushort> dmat){ //�������ԣ��� ���ܴ�����, ���봫ֵ
cloud2depthKernel(const PtrSz<float4> cloud_device, const Mat33 rot_inv, const float3 trans, const Intr intr, int ptSz, PtrStepSz<ushort> dmat){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; //�޽ṹ, ��һά
    //printf("threadIdx.x + blockIdx.x * blockDim.x; idx, cloud_device.size: %d, %d, %d, %d, %u\n", threadIdx.x, blockIdx.x, blockDim.x, idx, cloud_device.size); //�����, ��ֻ�� 4096��, Ϊʲô?
    //printf("threadIdx.x + blockIdx.x * blockDim.x; idx: %d, %d, %d, %d, %f\n", threadIdx.x, blockIdx.x, blockDim.x, idx, cloud_device.data[idx].x); //��, cloud_device.data ���Ի�ȡ��, ����ֻ�� 4096��, Ϊʲô?
    if(idx >= cloud_device.size){ //cloud û��ô���
        //printf("idx >= cloud_device.size\n");
        return;
    }

    float4 pt4 = cloud_device.data[idx]; //pt(i)
    float3 pi;
    pi.x = pt4.x; 
    pi.y = pt4.y;
    pi.z = pt4.z;

    float3 pi2cam = pi - trans; //pi �� camera ԭ������
    float3 vcam = rot_inv * pi2cam; //pi world->camera

    int u = int((vcam.x * intr.fx) / vcam.z + intr.cx); //����ͼ������ (u,v), Ӧ���� m->pixel?
    int v = int((vcam.y * intr.fy) / vcam.z + intr.cy);

    //������ͼ��Χ�ڣ�
    if(u < 0 || u > dmat.cols-1 ||
        v < 0 || v > dmat.rows-1)
        return;

    const int MAX_DEPTH = 10000;
    float z = vcam.z * 1000;
#if 01 //V1�汾, ֻ���ǵ�ǰ����
    ushort *px = dmat.ptr(v) + u;
    //ushort *px = dmat.ptr(10) + 10; //���ǲ���д, ��, ֮ǰʵ�����Ƕ� cloud_device �Ĵ�, ��������
    if(0 < z && z < MAX_DEPTH && //��z���ֵ����
        (0 == *px || *px > z))//�Ҵ�����δ����, ��z��������ֵС
        *px = (ushort)z; //�˴����ص��� float->ushort �ض�
#elif 0 //V2���� ptSz ����, ������������
//     if(0 < z && z < MAX_DEPTH){
//         //��ǰ���أ� 2016-6-21 14:28:32
//         ushort *px = dmat.ptr(v) + u;
//         if(0 == *px || *px > z) //��������δ����, ��z��������ֵС
//             *px = (ushort)z;
//         //ptSz����(Ҳ������ǰ����)
//         for(int iu = max(0, u - (ptSz-1)/2); iu < min(dmat.cols, u + 1 + ptSz/2); iu++){
//             for(int iv = max(0, v - (ptSz-1)/2); iv < min(dmat.rows, v + 1 + ptSz/2); iv++){
//                 ushort *pxNbr = dmat.ptr(iv) + iu;
//                 //if(0 == *px || *px > z)
//                 if(*pxNbr > z + 50){ //��⵽ͻ�䣨��ֵ50mm��, ��ǿ�и�ֵ; ֮ǰ��ֻҪ z ��С, һ�ɸ�ֵ)   //��ȥ�� if 0==*px
//                                     //��ʵ�߼�����, ֻ�����ڹ�����ƽ������, ֻ��ϣ������������ʱ��ģ����ȷ������, ��ֹԶ���㴩͸����������ʾ
//                     printf("pxNbr, z:=%d, %d\n", *pxNbr, z); //Ŀǰû������if��
// 
//                     *pxNbr = (ushort)z;
//                 }
//             }
//         }
//     }
#endif
}//cloud2depthKernel


void cloud2depth(DeviceArray<float4> cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, DepthMap &dmat){
    int block = 256,
        //grid = 512;
        grid = divUp(cloud_device.size(), block);
    //printf("cloud2depth, (grid,block): (%d, %d), cloud_device.size: %d, dmat.step: %d, .colsBytes: %d\n", grid, block, cloud_device.size(), dmat.step(), dmat.colsBytes());
    int ptSz = 5; //����ʱ�����������С. �����ǣ����ڵ���3mm�ĵ���, =1ʱ, �ȶ�3ms; =3,4ʱ������΢覴ã�0~4ms; =5ʱ, ��ȫ��覴�, 0~4ms
    cloud2depthKernel<<<grid, block>>>(cloud_device, rot_inv, trans, intr, ptSz, dmat);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
}//cloud2depth

void cloud2depth(DeviceArray<float4> cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, int imWidth, int imHeight, DepthMap &dmat){
    //�����б�� (w,h) �汾, �Է�ֱ�ӵ���֮ǰ�汾ʱ dmat δ�����ڴ�
    dmat.create(imHeight, imWidth);

    cloud2depth(cloud_device, rot_inv, trans, intr, dmat);
}//cloud2depth

void cloud2depth(DeviceArray2D<float> vmap, const Intr &intr, DepthMap &dmat){
    //TODO: vmap Ϊ 2D arr, �������ϵ, ���Բ����б���(R,t)
}//cloud2depth

__global__
void raycastCloudSubsetKernel(const PtrSz<float4> cloud_device, const Mat33 rot_inv, const float3 trans, const Intr intr, const PtrStepSz<ushort> dmat, PtrSz<int> outIdx){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; //�޽ṹ, ��һά
    if(idx >= cloud_device.size) //cloud û��ô���
        return;

    float4 pt4 = cloud_device.data[idx]; //pt(i)
    float3 pi;
    pi.x = pt4.x; 
    pi.y = pt4.y;
    pi.z = pt4.z;

    float3 pi2cam = pi - trans; //pi �� camera ԭ������
    float3 vcam = rot_inv * pi2cam; //pi world->camera

    int u = int((vcam.x * intr.fx) / vcam.z + intr.cx); //����ͼ������ (u,v), Ӧ���� m->pixel?
    int v = int((vcam.y * intr.fy) / vcam.z + intr.cy);

    //������ͼ��Χ�ڣ�
    if(u < 0 || u > dmat.cols-1 ||
        v < 0 || v > dmat.rows-1)
        return;

    const int MAX_DEPTH = 10000;
    float z = vcam.z * 1000; //ת��Ϊmm

    ushort depth = dmat(v, u);
    if(z -depth < 50) //������Ӧ�� z>=depth, ͬʱ̫��(>10mm)�����
        //����: <10 ̫С, Զ�˱�Ե������; <30 ��һЩ, ��������ʱԶ���Բ���;
        outIdx.data[idx] = 1;
}//raycastCloudSubsetKernel

void raycastCloudSubset(const DeviceArray<float4> &cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, const DepthMap &dmat, DeviceArray<int> &outIdx){
    int block = 256,
        grid = divUp(cloud_device.size(), block);

    raycastCloudSubsetKernel<<<grid, block>>>(cloud_device, rot_inv, trans, intr, dmat, outIdx);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
}//raycastCloudSubset

#define SZ_6DOF 6
__device__ int gGdCnt = 0;
__device__ float gRtArr[SZ_6DOF]; //��ʵ�Ǵ洢 sum(delta 6dof)/gGdCnt
__device__ float gRtArrMax[SZ_6DOF]; //�洢�ҵ������/Сֵ, ���ڹ�һ��normalize
__device__ float gRtArrMin[SZ_6DOF];

//@brief һ���ݶ��½�����, ��������6DOFƫ��, �ۼӵ�ȫ�ֱ��� gRtArr, ������ *1/gGdCnt
//@param[in] Rcurr, c->g, ��Ϊ srcDmat �����ͼ, ������������ϵ����
__global__
void align2dmapsKernel(const PtrStepSz<ushort> srcDmat, const Intr intr, const Mat33 Rcurr, const float3 tcurr, const PtrStepSz<ushort> dstDmat, const PtrStepSz<float> dstGradu, const PtrStepSz<float> dstGradv){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if(u >= srcDmat.cols || v >= srcDmat.rows)
        return;

    int z_c = srcDmat.ptr(v)[u];
    int huv = dstDmat.ptr(v)[u];
    if(z_c == 0 || huv == 0) //��������src��dst ��һ��0��������
        return;

    int depthDiff = z_c - huv; //��Ӧ������Ȳ�, ��������
    if(abs(depthDiff) > 100){ //��ֵ��⣺��������Ȳ����ĳ��ֵ, ������. �ݶ�10cm
        //printf("abs(depthDiff) > 100, == %d; (u,v)=(%d, %d)\n", depthDiff, u, v);
        return;
    }

    //������Ч��Ե���, ��Ч��Ч֮��, sobel �޴�, ϣ���ų��˸���, ����ȥ����Щ���� //�Ժ�Ҳ���Ը��á���Ч���ء�����������, �˴��ݲ�
    //��������7����
    int ksz = 7;
    int halfKsz = ksz / 2;
    //�ο� zc.cloud2depthKernel:
    //         for(int iu = max(0, u - (ptSz-1)/2); iu < min(dmat.cols, u + 1 + ptSz/2); iu++){
    //             for(int iv = max(0, v - (ptSz-1)/2); iv < min(dmat.rows, v + 1 + ptSz/2); iv++){
    //                 ushort *pxNbr = dmat.ptr(iv) + iu;
    for(int iu = max(0, u - halfKsz); iu <= min(dstDmat.cols, u + halfKsz); iu++){
        for(int iv = max(0, v - halfKsz); iv <= min(dstDmat.rows, v + halfKsz); iv++){
            ushort pxNbr = dstDmat.ptr(iv)[iu];
//             if(u==360 && v==280)
//                 printf("pxNbr == %d, (iu, iv)=(%d, %d)\n", pxNbr, iu, iv);

            if(pxNbr == 0){
                //printf("pxNbr == 0\n");
                return;
            }
        }
    }

    float su = dstGradu.ptr(v)[u],
          sv = dstGradv.ptr(v)[u]; //sobel (u,v), or alike

    float fu = intr.fx,
          fv = intr.fy,
          cu = intr.cx,
          cv = intr.cy;
    float A = su * fu / z_c,
          B = sv * fv / z_c,
          K = (u - cu) / fu,
          L = (v - cv) / fv,
          M = 1 + A * K + B * L;
    float3 vcurr; //src (u,v) ��Ӧ�������
    vcurr.x = K * z_c;
    vcurr.y = L * z_c;
    vcurr.z = z_c;
    float3 vcurr_g = Rcurr * vcurr + tcurr; //�� xg, yg, zg
    //printf("z_c, huv:= %d, %d\n", z_c, huv); //����printf��ͬ��, ������д
    //printf("z_c, huv:= %d, %d;\tvcurr:=(%f, %f, %f);\tvcurr_g:=(%f, %f, %f)\n", z_c, huv, vcurr.x, vcurr.y, vcurr.z, vcurr_g.x, vcurr_g.y, vcurr_g.z); //����֤���Ǻ��׳߶�, ��ӡֵ���������� //2016-6-26 21:38:05

    //����6�������� delta ��:
    float alpha = M * vcurr_g.y - B * vcurr_g.z,
          beta   = -M * vcurr_g.x - A * vcurr_g.z,
          gamma  = A * vcurr_g.y - B * vcurr_g.x,
          tx = -A,
          ty = -B,
          tz = M;
    //��Ҫ������Ȳ�:
    alpha *= depthDiff;
    beta *= depthDiff;
    gamma *= depthDiff;
    tx *= depthDiff;
    ty *= depthDiff;
    tz *= depthDiff;

    atomicAdd(&gGdCnt, 1);

    atomicAdd(&gRtArr[0], alpha);
    atomicAdd(&gRtArr[1], beta);
    atomicAdd(&gRtArr[2], gamma);
    atomicAdd(&gRtArr[3], tx);
    atomicAdd(&gRtArr[4], ty);
    atomicAdd(&gRtArr[5], tz);

    //��������һ��Ԥ���� //2016-6-30 14:57:54
    atomicMaxf(&gRtArrMax[0], alpha);
    atomicMaxf(&gRtArrMax[1], beta);
    atomicMaxf(&gRtArrMax[2], gamma);
    atomicMaxf(&gRtArrMax[3], tx);
    atomicMaxf(&gRtArrMax[4], ty);
    atomicMaxf(&gRtArrMax[5], tz);

    atomicMinf(&gRtArrMin[0], alpha);
    atomicMinf(&gRtArrMin[1], beta);
    atomicMinf(&gRtArrMin[2], gamma);
    atomicMinf(&gRtArrMin[3], tx);
    atomicMinf(&gRtArrMin[4], ty);
    atomicMinf(&gRtArrMin[5], tz);
}//align2dmapsKernel

void align2dmapsOnce(const DepthMap &srcDmat, const Intr &intr, const Mat33 &Rcurr, const float3 &tcurr, const DepthMap &dstDmat, const MapArr &dstGradu, const MapArr &dstGradv, float *poseBuf){
    dim3 block(32, 8);
    dim3 grid(divUp(srcDmat.cols(), block.x), divUp(srcDmat.rows(), block.y));

    float rtArrHost[SZ_6DOF] = {0};
//     for(size_t i=0; i<SZ_6DOF; i++)
//         rtArrHost[i] = 0;
    cudaMemcpyToSymbol(gRtArr, &rtArrHost, sizeof(float)*SZ_6DOF); //�洢���� 6DOF (dR,dt) ������

    int gdCnt = 0;
    cudaMemcpyToSymbol(gGdCnt, &gdCnt, sizeof(int)); //��Ч���ؼ�����

    //��������һ��Ԥ���� //2016-6-30 14:57:54
    float rtArrMaxHost[SZ_6DOF] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    cudaMemcpyToSymbol(gRtArrMax, &rtArrMaxHost, sizeof(float)*SZ_6DOF);
    float rtArrMinHost[SZ_6DOF] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    cudaMemcpyToSymbol(gRtArrMin, &rtArrMinHost, sizeof(float)*SZ_6DOF);

    //+++++++++++++++���ļ��㺯��
    align2dmapsKernel<<<grid, block>>>(srcDmat, intr, Rcurr, tcurr, dstDmat, dstGradu, dstGradv);

    cudaMemcpyFromSymbol(&gdCnt, gGdCnt, sizeof(int));
    cudaMemcpyFromSymbol(&rtArrHost, gRtArr, sizeof(float)*SZ_6DOF);

    cudaMemcpyFromSymbol(&rtArrMaxHost, gRtArrMax, sizeof(float)*SZ_6DOF);
    cudaMemcpyFromSymbol(&rtArrMinHost, gRtArrMin, sizeof(float)*SZ_6DOF);

    cout<<"gdCnt: "<<gdCnt<<endl;
    float gdStepSz = 1e-6; //��������: learn-rate, ����, ѧϰ��
//     float gdStepSz[SZ_6DOF] = //����pwp3d, �ĳɸ�������ͬ����, RСt��
//         //{1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4};
//     {1e-9, 1e-9, 1e-5, 1e-4, 1e-4, 1e-4};
    //{1e-5, 1e-5, 1e-1, 1, 1, 1};

            //static int callCnt = 0; //������ĳ��ѭ��֮��, ����alpha����, ������ǲ� //2016-6-29 23:23:05
            //callCnt++;
            //if(callCnt>10)
            //    gdStepSz[0]=1e-6;
    //float gdStepSz[2] =
    //{1e-7, 1e-4}; //t:1e-3����ҡ��, 1e-4 ǡ��; 1e-5Ҳ��Ʈ, Ϊʲô?  //R:1e-7������ err=2.5; 1e-8::err=1.01; 1e-9::err=1.006; 1e-10:err=1.002
    //{1e-5, 1e-3};
    for(size_t i=0; i<SZ_6DOF; i++){
        cout<<"rtArrHost["<<i<<"]:"<<rtArrHost[i];//<<endl;
        rtArrHost[i] /= gdCnt; //(��x)/n
        //��һ��: ��kernel֮��, �����Ǻ����һ��, ���Ǽ�ʱ�� //2016-6-30 15:36:36
        //rtArrHost[i] = (rtArrHost[i] - (rtArrMaxHost[i] + rtArrMinHost[i])/2) / (rtArrMaxHost[i] - rtArrMinHost[i]); //����ܲ�, ��ʱ���� 2016-6-30 15:53:22
        rtArrHost[i] *= gdStepSz;
        //rtArrHost[i] *= gdStepSz[i];
        //rtArrHost[i] *= gdStepSz[i / 3]; //����z��ת����, xyzͬ�����Ľ������, Ӧ������rz����
        poseBuf[i] = rtArrHost[i]; //+++++++++++++++��ֵ��������
        cout<<",\t"<<rtArrHost[i]<<"; (max, min)= "<<rtArrMaxHost[i]<<", "<<rtArrMinHost[i]<<endl;
    }
    for(size_t i=0; i<SZ_6DOF; i++)
        cout<<rtArrHost[i]<<", ";
    cout<<endl;

}//align2dmapsOnce


//+++++++++++++++���涼��һЩGPU�ڴ��д����
void foo_in_syn_cu(){}

//����ȫ��dev����
__device__ int gVarCntDev;

__global__
//void test_write_gpu_mem_in_syn_cu_kernel(PtrStep<ushort> mem_device){
void fooKernel(PtrStep<ushort> mem_device, int *cntDev){

    int idx = threadIdx.x + blockIdx.x * blockDim.x; //�޽ṹ, ��һά
    //if(idx > mem_device.step){
    if(idx > 654){
        //printf("idx, step: %d, %u\n", idx, mem_device.step);
        return;
    }
    atomicAdd(&gVarCntDev, 1);
    //atomicAdd(cntDev, 1); //��
    //printf("cntDev: %d\n", cntDev); //��, �������ĩ����ֵ, ����Ϊ������?

    //mem_device.ptr(10)[10]=321;
    //mem_device(0, idx) = idx;
    //mem_device(0, idx) = (ushort)idx; //һ��
    //if(40 <= idx && idx < mem_device.step) //40��ʼ 256����, Ϊʲô?
    //if(40 <= idx && idx < 44) //ȫ0
    //if(40 <= idx && idx < 84) //ȫ0, ����� mem_device.step �ù��찡
//     if(0 <= idx && idx < 84) //0~39��, ���滹��0, �ѵ� idx �����ܵ� 80 ��?
//         mem_device.ptr()[idx] = idx;
    //mem_device(2, 10) = 123;//[90]=123��
    printf("fooKernel: %d, %d, %d, %d, %d\n", idx, threadIdx.x, blockIdx.x, blockDim.x, gVarCntDev);
}//test_write_gpu_mem_in_syn_cu_kernel

void test_write_gpu_mem_in_syn_cu(){
    int block = 512,
        //grid = 512;
        grid = divUp(4321, block);

    DeviceArray2D<ushort> depOut_device;
    int imHeight=30, imWidth=40;
    depOut_device.create(imHeight, imWidth);

    int varCntHost = -1;
    int *pVarCntDev;
    cudaMalloc(&pVarCntDev, sizeof(int));
    cudaMemcpy(pVarCntDev, &varCntHost, sizeof(int), cudaMemcpyHostToDevice);

    fooKernel<<<grid, block>>>(depOut_device, pVarCntDev);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    //int varCntHost;
    //gVarCntDev
    cudaError rc = cudaMemcpyFromSymbol(&varCntHost, gVarCntDev, sizeof(int));
    printf("cudaMemcpyFromSymbol, varCntHost: %d, rc= %d\n", varCntHost, rc);

    cudaMemcpy(&varCntHost, pVarCntDev, sizeof(int), cudaMemcpyDeviceToHost);
    cout<<"varCntHost: "<<varCntHost<<endl;

    vector<ushort> cpu_mem(imHeight*imWidth);
    depOut_device.download(cpu_mem, imWidth); //��
    //depOut_device.download(cpu_mem.data(), imWidth * sizeof(ushort)); //��, һ��

//     for(int i=0; i<100; i++){
//         cout<<i<<", "<<cpu_mem[i]<<endl;
//     }
}//test_write_gpu_mem_in_syn_cu

}//namespace zc
