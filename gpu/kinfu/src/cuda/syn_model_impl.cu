//@author zhangxaochen  //2016-4-5 04:17:49


#include <iostream>
//#include <math.h>
#include <stdio.h>
#include <float.h>

//#include <Eigen/Core>

// #include <pcl/common/centroid.h>
// #include <pcl/point_cloud.h>
//#include <pcl/point_types.h> //有eigen, 不要放在 cu
//#include <pcl/pcl_macros.h>

#include "internal.h"

#include "syn_model_impl.h" //有 eigen
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

//@ref: 拷贝自 http://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda //2016-6-30 15:13:24
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

//@brief 按 atomicMaxf, 改 '>' 为 '<'
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

//@brief 用无结构点云, 在GPU上计算 tsdf. 参照了 initializeVolume / tsdf23@tsdf_volume.cu
//@param[in] cloud_device, GPU上的点云
//@param[in] pcen, centroid point of *cloud_device*, (meter)
//@param[in] cell_size, 单个体素物理尺寸(meter)
//@param[in] tuncDist, tsdf truncation distance in meters
//@param[in] weight, 点云转为tsdf时可能需要更高的权重 
//@param[out] tsdf_volume, GPU上的TSDF模型
__global__ void
cloud2tsdfKernel(const PtrSz<float4> cloud_device, float3 pcen, float3 cell_size, float tuncDist, int weight, PtrStep<short2> tsdf_volume){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; //无结构, 仅一维
    //y = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx >= cloud_device.size){ //cloud 没那么多点
        return;
    }

    float4 pt4 = cloud_device.data[idx]; //pt(i)
    float3 pi;
    pi.x = pt4.x; 
    pi.y = pt4.y;
    pi.z = pt4.z;

    int3 g = getVoxel(pi, cell_size); //a voxel grid (x,y,z) index
    //仿照 @interpolateTrilineary 做边界判定:
    if(g.x <= 0 || g.x >= VOLUME_X - 1 ||
        g.y <= 0 || g.y >= VOLUME_Y - 1 ||
        g.z <= 0 || g.z >= VOLUME_Z - 1)
        return;

    //体素中心的坐标, 物理尺寸
    float3 vi;
    vi.x = (g.x + 0.5f) * cell_size.x;
    vi.y = (g.y + 0.5f) * cell_size.y;
    vi.z = (g.z + 0.5f) * cell_size.z;

    //@ray_caster.cu 里这样做(后退一格)的目的何在？ 原因不懂, 暂不用
    //g.x = (point.x < vx) ? (g.x - 1) : g.x;
    //g.y = (point.y < vy) ? (g.y - 1) : g.y;
    //g.z = (point.z < vz) ? (g.z - 1) : g.z;

    float3 pipc = pi - pcen; //pt(i)-pcen

//     pipc.x = pi.x - pcen.x;
//     pipc.y = pi.y - pcen.y;
//     pipc.z = pi.z - pcen.z;

    //rsqrtf: 求 1/norm(ptic)
    //见：http://datamining.xmu.edu.cn/documentation/cuda4.1/group__CUDA__MATH__SINGLE_g5a9bc318028131cfd13d10abfae1ae13.html
    float pipc_norm_inv = rsqrtf(dot(pipc, pipc));

    //+++++++++++++++1. 求Vi处
    //用向量 (vi-pi) 在 (pi-pcen) 上的投影长度作为 sdf
    //float sdf = ((vx - pt.x) * ptic.x + (vy - pt.y) * ptic.y + (yz - pt.z) * ptic.z) * ptic_norm_inv;
    float sdf = device::dot(vi - pi, pipc) * pipc_norm_inv;
    float tsdfVal = fmax(-1.f, fmin(1.f, sdf / tuncDist)); //这样对不对？

#if 01 //定位V1
    int elem_step = VOLUME_Y * tsdf_volume.step / sizeof(short2); //一面xy面占用的元素数, 其实就差不多 512*512 //暂时弃用
    short2 *pos = tsdf_volume.ptr(g.y) + g.x; //定位对不对？
    pos += elem_step * g.z;

    //pos = tsdf_volume.ptr(); //尝试仅写入tsdf [0]位置
    //pos->x = 1;
#elif 1 //改用 readTSDF 定位方式V2： //理论上应该与V1相同, 都是: XYz+Xy+x
    short2 *pos = tsdf_volume.ptr (VOLUME_Y * g.z + g.y) + g.x;
#endif //两种定位方式
    //其实不必用这个 if：
    //if(g.x > 0 && g.x < VOLUME_X - 1 &&
    //    g.y > 0 && g.y < VOLUME_Y - 1 &&
    //    g.z > 0 && g.z < VOLUME_Z - 1)
    //{
    //atomicAdd(&tsdfKrnlCnt, 1); //最终确实==cloud_device.size()
    //printf("tsdfKrnlCnt: %d\n", tsdfKrnlCnt); //不要kernel内print全局device变量
    float F_old = 9527;
    int W_old = 521;

    unpack_tsdf(*pos, F_old, W_old);
    if(W_old == 0){
        //printf无输出, 很奇怪:
        //printf("0-grid: %d, %d, %d; F: %f, W: %d\n", g.x, g.y, g.z, F, W);
        atomicAdd(&tsdfKrnlCnt, 1);
        //日志: 若下面不 pack_tsdf, 此处仍然tsdfKrnlCnt==cloud_device.size(), 如: 52666
        //若后面加 pack_tsdf, tsdfKrnlCnt 减少, 如: 10240
    }
    //printf("grid: %d, %d, %d; F: %f, W: %d\n", g.x, g.y, g.z, tsdfVal, weight);
    //printf("grid: %d, %d, %d\n", g.x, g.y, g.z);
    //printf("---------------\n");
    //不管 W_old是否为零, 直接加权: //weight 加权会导致超出人工传参设定, 先不管
    tsdfVal = (tsdfVal * weight + F_old * W_old) / (weight + W_old);
    weight = (weight + W_old) % 128; //参考 MAX_WEIGHT=128 @tsdf_volume.cu
    pack_tsdf(tsdfVal, weight, *pos);
    //}

    //+++++++++++++++2. 求26邻域Vk处
    for(int ix = g.x - 1; ix <= g.x + 1; ix++){
        for(int iy = g.y - 1; iy <= g.y + 1; iy++){
            for(int iz = g.z - 1; iz <= g.z + 1; iz++){
                //pos = tsdf_volume.ptr(iy)[ix] + elem_step * iz; //×, '+'不对
                pos = tsdf_volume.ptr (VOLUME_Y * iz + iy) + ix;
                float tsdf_k_old;
                int wt_k_old;
                unpack_tsdf(*pos, tsdf_k_old, wt_k_old);
                //if(0 != wt_k) //说明已初始化过, 跳过 //不! 已初始化也不要跳过
                //    continue;

                float3 vk;
                vk.x = (ix + 0.5f) * cell_size.x;
                vk.y = (iy + 0.5f) * cell_size.y;
                vk.z = (iz + 0.5f) * cell_size.z;

                //求向量 (Vk-Pi), (Pi-pcen)的夹角, <45°或>135°者, 也求其tsdf
                float3 vkpi = vk - pi;
                float vkpi_norm_inv = rsqrtf(dot(vkpi, vkpi));

                //float vkpi_dot_pipc = dot(vkpi, pipc);
                float vkpi_proj_pipc = dot(vkpi, pipc) * pipc_norm_inv ;
                float angle = acosf(vkpi_proj_pipc * vkpi_norm_inv); //余弦定理, radians
                float angDeg = angle * 180 / M_PI;
                if(angDeg < 45 || angDeg > 135){
                    float sdf = vkpi_proj_pipc; //理论上有正有负
                    float tsdfVal = fmax(-1.f, fmin(1.f, sdf / tuncDist));
                    //加权：
                    tsdfVal = (tsdfVal * weight + tsdf_k_old * wt_k_old) / (weight+wt_k_old);
                    weight = (weight + wt_k_old)%256; //上限255
                    pack_tsdf(tsdfVal, weight, *pos);
                }
            }//for-iz
        }//for-iy
    }//for-ix
}//cloud2tsdfKernel

void cloud2tsdf(const DeviceArray<float4> &cloud_device, const float3 &pcen, const float3 &cell_sz, float truncDist, int weight, PtrStep<short2> tsdf_volume){
//void cloud2tsdf(DeviceArray<PointXYZ> cloud_device, PtrStep<short2> tsdf_volume){
    //grid & block shape 随便设的, 总 512^2=262144, 大于点云点数就行
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
//     cudaMemcpy(&tsdfKrnlCntHost, &tsdfKrnlCnt, sizeof(int), cudaMemcpyDeviceToHost); //×
//     cout<<"cudaMemcpy, tsdfKrnlCntHost: "<<tsdfKrnlCntHost<<endl;

    cudaMemcpyFromSymbol(&tsdfKrnlCntHost, tsdfKrnlCnt, sizeof(int)); //√
    cout<<"cudaMemcpyFromSymbol, tsdfKrnlCntHost: "<<tsdfKrnlCntHost<<endl;

}//cloud2tsdf

//@brief GPU impl. kernel
//@param[in] ptSz, 点大小，其实是指邻域大小。e.g., ptSz=2即右下2x2邻域, ptSz=3即四周3x3邻域。若点云较稀疏，则增大ptSz，以免较远点穿透较近点云
//@param[out] dmat, 光线投射(消隐)后得到的深度图
__global__ void
//cloud2depthKernel(const PtrSz<float4> &cloud_device, const Mat33 rot_inv, const float3 trans, const Intr intr, PtrStepSz<ushort> dmat){ //不长记性！！ 不能传引用, 必须传值
cloud2depthKernel(const PtrSz<float4> cloud_device, const Mat33 rot_inv, const float3 trans, const Intr intr, int ptSz, PtrStepSz<ushort> dmat){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; //无结构, 仅一维
    //printf("threadIdx.x + blockIdx.x * blockDim.x; idx, cloud_device.size: %d, %d, %d, %d, %u\n", threadIdx.x, blockIdx.x, blockDim.x, idx, cloud_device.size); //可输出, 但只有 4096次, 为什么?
    //printf("threadIdx.x + blockIdx.x * blockDim.x; idx: %d, %d, %d, %d, %f\n", threadIdx.x, blockIdx.x, blockDim.x, idx, cloud_device.data[idx].x); //√, cloud_device.data 可以获取了, 但仍只有 4096次, 为什么?
    if(idx >= cloud_device.size){ //cloud 没那么多点
        //printf("idx >= cloud_device.size\n");
        return;
    }

    float4 pt4 = cloud_device.data[idx]; //pt(i)
    float3 pi;
    pi.x = pt4.x; 
    pi.y = pt4.y;
    pi.z = pt4.z;

    float3 pi2cam = pi - trans; //pi 到 camera 原点向量
    float3 vcam = rot_inv * pi2cam; //pi world->camera

    int u = int((vcam.x * intr.fx) / vcam.z + intr.cx); //计算图像坐标 (u,v), 应该是 m->pixel?
    int v = int((vcam.y * intr.fy) / vcam.z + intr.cy);

    //若不在图像范围内：
    if(u < 0 || u > dmat.cols-1 ||
        v < 0 || v > dmat.rows-1)
        return;

    const int MAX_DEPTH = 10000;
    float z = vcam.z * 1000;
#if 01 //V1版本, 只考虑当前像素
    ushort *px = dmat.ptr(v) + u;
    //ushort *px = dmat.ptr(10) + 10; //还是不可写, ×, 之前实际上是读 cloud_device 的错, 传了引用
    if(0 < z && z < MAX_DEPTH && //若z深度值合理
        (0 == *px || *px > z))//且此像素未填充过, 或z比已填充的值小
        *px = (ushort)z; //此处不必担心 float->ushort 截断
#elif 0 //V2增加 ptSz 参数, 考虑邻域像素
//     if(0 < z && z < MAX_DEPTH){
//         //当前像素： 2016-6-21 14:28:32
//         ushort *px = dmat.ptr(v) + u;
//         if(0 == *px || *px > z) //若此像素未填充过, 或z比已填充的值小
//             *px = (ushort)z;
//         //ptSz邻域：(也包括当前像素)
//         for(int iu = max(0, u - (ptSz-1)/2); iu < min(dmat.cols, u + 1 + ptSz/2); iu++){
//             for(int iv = max(0, v - (ptSz-1)/2); iv < min(dmat.rows, v + 1 + ptSz/2); iv++){
//                 ushort *pxNbr = dmat.ptr(iv) + iu;
//                 //if(0 == *px || *px > z)
//                 if(*pxNbr > z + 50){ //检测到突变（阈值50mm）, 才强行赋值; 之前是只要 z 更小, 一律赋值)   //且去掉 if 0==*px
//                                     //其实逻辑不对, 只适用于孤立、平滑表面, 只是希望点云做输入时能模拟正确的消隐, 阻止远处点穿透近处点面显示
//                     printf("pxNbr, z:=%d, %d\n", *pxNbr, z); //目前没有至此if过
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
    int ptSz = 5; //消隐时，考虑邻域大小. 经验是：对于点间距3mm的点云, =1时, 稳定3ms; =3,4时，有轻微瑕疵，0~4ms; =5时, 完全无瑕疵, 0~4ms
    cloud2depthKernel<<<grid, block>>>(cloud_device, rot_inv, trans, intr, ptSz, dmat);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
}//cloud2depth

void cloud2depth(DeviceArray<float4> cloud_device, const Mat33 &rot_inv, const float3 &trans, const Intr &intr, int imWidth, int imHeight, DepthMap &dmat){
    //参数列表带 (w,h) 版本, 以防直接调用之前版本时 dmat 未分配内存
    dmat.create(imHeight, imWidth);

    cloud2depth(cloud_device, rot_inv, trans, intr, dmat);
}//cloud2depth

void cloud2depth(DeviceArray2D<float> vmap, const Intr &intr, DepthMap &dmat){
    //TODO: vmap 为 2D arr, 相机坐标系, 所以参数列表不带(R,t)
}//cloud2depth

__global__
void raycastCloudSubsetKernel(const PtrSz<float4> cloud_device, const Mat33 rot_inv, const float3 trans, const Intr intr, const PtrStepSz<ushort> dmat, PtrSz<int> outIdx){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; //无结构, 仅一维
    if(idx >= cloud_device.size) //cloud 没那么多点
        return;

    float4 pt4 = cloud_device.data[idx]; //pt(i)
    float3 pi;
    pi.x = pt4.x; 
    pi.y = pt4.y;
    pi.z = pt4.z;

    float3 pi2cam = pi - trans; //pi 到 camera 原点向量
    float3 vcam = rot_inv * pi2cam; //pi world->camera

    int u = int((vcam.x * intr.fx) / vcam.z + intr.cx); //计算图像坐标 (u,v), 应该是 m->pixel?
    int v = int((vcam.y * intr.fy) / vcam.z + intr.cy);

    //若不在图像范围内：
    if(u < 0 || u > dmat.cols-1 ||
        v < 0 || v > dmat.rows-1)
        return;

    const int MAX_DEPTH = 10000;
    float z = vcam.z * 1000; //转换为mm

    ushort depth = dmat(v, u);
    if(z -depth < 50) //理论上应该 z>=depth, 同时太大(>10mm)的舍掉
        //经验: <10 太小, 远端边缘误舍弃; <30 好一些, 近似正对时远端仍不好;
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
__device__ float gRtArr[SZ_6DOF]; //其实是存储 sum(delta 6dof)/gGdCnt
__device__ float gRtArrMax[SZ_6DOF]; //存储找到的最大/小值, 用于归一化normalize
__device__ float gRtArrMin[SZ_6DOF];

//@brief 一次梯度下降迭代, 逐像素求6DOF偏导, 累加到全局变量 gRtArr, 不负责 *1/gGdCnt
//@param[in] Rcurr, c->g, 因为 srcDmat 是深度图, 而非世界坐标系点云
__global__
void align2dmapsKernel(const PtrStepSz<ushort> srcDmat, const Intr intr, const Mat33 Rcurr, const float3 tcurr, const PtrStepSz<ushort> dstDmat, const PtrStepSz<float> dstGradu, const PtrStepSz<float> dstGradv){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if(u >= srcDmat.cols || v >= srcDmat.rows)
        return;

    int z_c = srcDmat.ptr(v)[u];
    int huv = dstDmat.ptr(v)[u];
    if(z_c == 0 || huv == 0) //若此像素src、dst 有一个0，则跳过
        return;

    int depthDiff = z_c - huv; //对应像素深度差, 带正负号
    if(abs(depthDiff) > 100){ //阈值检测：两像素深度差大于某阈值, 则跳过. 暂定10cm
        //printf("abs(depthDiff) > 100, == %d; (u,v)=(%d, %d)\n", depthDiff, u, v);
        return;
    }

    //增加无效边缘检测, 无效有效之间, sobel 巨大, 希望排除此干扰, 于是去掉这些像素 //以后也可以改用“有效像素”作函数参数, 此处暂不
    //暂用最大的7邻域
    int ksz = 7;
    int halfKsz = ksz / 2;
    //参考 zc.cloud2depthKernel:
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
    float3 vcurr; //src (u,v) 对应相机坐标
    vcurr.x = K * z_c;
    vcurr.y = L * z_c;
    vcurr.z = z_c;
    float3 vcurr_g = Rcurr * vcurr + tcurr; //求 xg, yg, zg
    //printf("z_c, huv:= %d, %d\n", z_c, huv); //两个printf不同步, 别这样写
    //printf("z_c, huv:= %d, %d;\tvcurr:=(%f, %f, %f);\tvcurr_g:=(%f, %f, %f)\n", z_c, huv, vcurr.x, vcurr.y, vcurr.z, vcurr_g.x, vcurr_g.y, vcurr_g.z); //已验证：是毫米尺度, 打印值看起来正常 //2016-6-26 21:38:05

    //下面6个量都是 delta 量:
    float alpha = M * vcurr_g.y - B * vcurr_g.z,
          beta   = -M * vcurr_g.x - A * vcurr_g.z,
          gamma  = A * vcurr_g.y - B * vcurr_g.x,
          tx = -A,
          ty = -B,
          tz = M;
    //还要乘以深度差:
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

    //尝试做归一化预处理 //2016-6-30 14:57:54
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
    cudaMemcpyToSymbol(gRtArr, &rtArrHost, sizeof(float)*SZ_6DOF); //存储最终 6DOF (dR,dt) 的数组

    int gdCnt = 0;
    cudaMemcpyToSymbol(gGdCnt, &gdCnt, sizeof(int)); //有效像素计数器

    //尝试做归一化预处理 //2016-6-30 14:57:54
    float rtArrMaxHost[SZ_6DOF] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    cudaMemcpyToSymbol(gRtArrMax, &rtArrMaxHost, sizeof(float)*SZ_6DOF);
    float rtArrMinHost[SZ_6DOF] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    cudaMemcpyToSymbol(gRtArrMin, &rtArrMinHost, sizeof(float)*SZ_6DOF);

    //+++++++++++++++核心计算函数
    align2dmapsKernel<<<grid, block>>>(srcDmat, intr, Rcurr, tcurr, dstDmat, dstGradu, dstGradv);

    cudaMemcpyFromSymbol(&gdCnt, gGdCnt, sizeof(int));
    cudaMemcpyFromSymbol(&rtArrHost, gRtArr, sizeof(float)*SZ_6DOF);

    cudaMemcpyFromSymbol(&rtArrMaxHost, gRtArrMax, sizeof(float)*SZ_6DOF);
    cudaMemcpyFromSymbol(&rtArrMinHost, gRtArrMin, sizeof(float)*SZ_6DOF);

    cout<<"gdCnt: "<<gdCnt<<endl;
    float gdStepSz = 1e-6; //物理意义: learn-rate, 步长, 学习率
//     float gdStepSz[SZ_6DOF] = //仿照pwp3d, 改成各参数不同步长, R小t大
//         //{1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4};
//     {1e-9, 1e-9, 1e-5, 1e-4, 1e-4, 1e-4};
    //{1e-5, 1e-5, 1e-1, 1, 1, 1};

            //static int callCnt = 0; //测试在某次循环之后, 增加alpha步长, 结果还是差 //2016-6-29 23:23:05
            //callCnt++;
            //if(callCnt>10)
            //    gdStepSz[0]=1e-6;
    //float gdStepSz[2] =
    //{1e-7, 1e-4}; //t:1e-3导致摇摆, 1e-4 恰好; 1e-5也会飘, 为什么?  //R:1e-7收敛到 err=2.5; 1e-8::err=1.01; 1e-9::err=1.006; 1e-10:err=1.002
    //{1e-5, 1e-3};
    for(size_t i=0; i<SZ_6DOF; i++){
        cout<<"rtArrHost["<<i<<"]:"<<rtArrHost[i];//<<endl;
        rtArrHost[i] /= gdCnt; //(Σx)/n
        //归一化: 在kernel之外, 所以是后处理归一化, 不是即时的 //2016-6-30 15:36:36
        //rtArrHost[i] = (rtArrHost[i] - (rtArrMaxHost[i] + rtArrMinHost[i])/2) / (rtArrMaxHost[i] - rtArrMinHost[i]); //结果很差, 暂时放弃 2016-6-30 15:53:22
        rtArrHost[i] *= gdStepSz;
        //rtArrHost[i] *= gdStepSz[i];
        //rtArrHost[i] *= gdStepSz[i / 3]; //对绕z旋转情形, xyz同步长的结果不好, 应该增大rz步长
        poseBuf[i] = rtArrHost[i]; //+++++++++++++++赋值传出参数
        cout<<",\t"<<rtArrHost[i]<<"; (max, min)= "<<rtArrMaxHost[i]<<", "<<rtArrMinHost[i]<<endl;
    }
    for(size_t i=0; i<SZ_6DOF; i++)
        cout<<rtArrHost[i]<<", ";
    cout<<endl;

}//align2dmapsOnce


//+++++++++++++++下面都是一些GPU内存读写测试
void foo_in_syn_cu(){}

//测试全局dev变量
__device__ int gVarCntDev;

__global__
//void test_write_gpu_mem_in_syn_cu_kernel(PtrStep<ushort> mem_device){
void fooKernel(PtrStep<ushort> mem_device, int *cntDev){

    int idx = threadIdx.x + blockIdx.x * blockDim.x; //无结构, 仅一维
    //if(idx > mem_device.step){
    if(idx > 654){
        //printf("idx, step: %d, %u\n", idx, mem_device.step);
        return;
    }
    atomicAdd(&gVarCntDev, 1);
    //atomicAdd(cntDev, 1); //√
    //printf("cntDev: %d\n", cntDev); //√, 都输出的末了数值, 是因为串行吗?

    //mem_device.ptr(10)[10]=321;
    //mem_device(0, idx) = idx;
    //mem_device(0, idx) = (ushort)idx; //一样
    //if(40 <= idx && idx < mem_device.step) //40开始 256递增, 为什么?
    //if(40 <= idx && idx < 44) //全0
    //if(40 <= idx && idx < 84) //全0, 那这个 mem_device.step 好诡异啊
//     if(0 <= idx && idx < 84) //0~39√, 后面还是0, 难道 idx 不是能到 80 吗?
//         mem_device.ptr()[idx] = idx;
    //mem_device(2, 10) = 123;//[90]=123√
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
    depOut_device.download(cpu_mem, imWidth); //√
    //depOut_device.download(cpu_mem.data(), imWidth * sizeof(ushort)); //√, 一样

//     for(int i=0; i<100; i++){
//         cout<<i<<", "<<cpu_mem[i]<<endl;
//     }
}//test_write_gpu_mem_in_syn_cu

}//namespace zc
