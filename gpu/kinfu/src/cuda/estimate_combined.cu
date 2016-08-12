/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 * 
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

//#include <pcl/gpu/utils/device/block.hpp>
//#include <pcl/gpu/utils/device/funcattrib.hpp>
#include "device.hpp"

//zhangxaochen:
#include <assert.h>

namespace pcl
{
  namespace device
  {
    typedef double float_type;

    template<int CTA_SIZE_, typename T>
    static __device__ __forceinline__ void reduce(volatile T* buffer)
    {
      int tid = Block::flattenedThreadId();
      T val =  buffer[tid];

      if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; __syncthreads(); }
      if (CTA_SIZE_ >=  512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; __syncthreads(); }
      if (CTA_SIZE_ >=  256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; __syncthreads(); }
      if (CTA_SIZE_ >=  128) { if (tid <  64) buffer[tid] = val = val + buffer[tid +  64]; __syncthreads(); }

      if (tid < 32)
      {
        if (CTA_SIZE_ >=   64) { buffer[tid] = val = val + buffer[tid +  32]; }
        if (CTA_SIZE_ >=   32) { buffer[tid] = val = val + buffer[tid +  16]; }
        if (CTA_SIZE_ >=   16) { buffer[tid] = val = val + buffer[tid +   8]; }
        if (CTA_SIZE_ >=    8) { buffer[tid] = val = val + buffer[tid +   4]; }
        if (CTA_SIZE_ >=    4) { buffer[tid] = val = val + buffer[tid +   2]; }
        if (CTA_SIZE_ >=    2) { buffer[tid] = val = val + buffer[tid +   1]; }
      }
    }

    struct Combined
    {
      enum
      {
        CTA_SIZE_X = 32,
        CTA_SIZE_Y = 8,
        CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
      };


      Mat33 Rcurr;
      float3 tcurr;

      PtrStep<float> vmap_curr;
      PtrStep<float> nmap_curr;

      //sunguofei
      PtrStep<float> vmap_contour;
      PtrStep<float> vmap_candidate;
      PtrStep<float> nmap_candidate;

      Mat33 Rprev_inv;
      float3 tprev;

      Intr intr;

      PtrStep<float> vmap_g_prev;
      PtrStep<float> nmap_g_prev;

      float distThres;
      float angleThres;

      int cols;
      int rows;

      //sunguofei
      int rows_contour;
      double weight;

      mutable PtrStep<float_type> gbuf;

      __device__ __forceinline__ bool
      search (int x, int y, float3& n, float3& d, float3& s) const
      {
//#ifdef GCOO_2_CAMCOO //g2c
#if 0
          //若 GCOO_2_CAMCOO, 外面 v/nmap_g_prev 是在当前视点下获得的, 所以此处不必如 kinfu.orig 那样做 cam_curr->world->cam_prev 两次转换
          float3 ncurr;
          ncurr.x = nmap_curr.ptr(y)[x];
          if(isnan(ncurr.x))
              return false;

          ncurr.y = nmap_curr.ptr(y+rows)[x];
          ncurr.z = nmap_curr.ptr(y+2*rows)[x];

          float3 vcurr;
          vcurr.x = vmap_curr.ptr(y)[x];
          vcurr.y = vmap_curr.ptr(y+rows)[x];
          vcurr.z = vmap_curr.ptr(y+2*rows)[x];

          //这里不需要像 kinfu.orig 那样求解 ukr 像素坐标, 直接用 [x,y]
          float3 nprev_g;
          nprev_g.x = nmap_g_prev.ptr(y)[x];
          if(isnan(nprev_g.x))
              return false;

          nprev_g.y = nmap_g_prev.ptr(y+rows)[x];
          nprev_g.z = nmap_g_prev.ptr(y+2*rows)[x];

          float3 vprev_g;
          vprev_g.x = vmap_g_prev.ptr(y)[x];
          vprev_g.y = vmap_g_prev.ptr(y+rows)[x];
          vprev_g.z = vmap_g_prev.ptr(y+2*rows)[x];

          //这里求 Rcurr.inv, 放在 kernel 里会低效, 但是暂时不管
          Mat33 Rcurr_inv = Rcurr; //以下做转置, 求逆矩阵：
          Rcurr_inv.data[0].y = Rcurr.data[1].x; //21->12
          Rcurr_inv.data[0].z = Rcurr.data[2].x; //31->13
          Rcurr_inv.data[1].z = Rcurr.data[2].y; //32->23

          Rcurr_inv.data[1].x = Rcurr.data[0].y; //12->21
          Rcurr_inv.data[2].x = Rcurr.data[0].z; //13->31
          Rcurr_inv.data[2].y = Rcurr.data[1].z; //23->32

          float3 vprev_cc = Rcurr_inv * (vprev_g - tcurr); //cc: curr-cam_coo
          float3 nprev_cc = Rcurr_inv * nprev_g;

          float dist = norm (vcurr - vprev_cc);
          if (dist > distThres)
              return false;

          float sine = norm (cross (ncurr, nprev_cc));
          if (sine >= angleThres)
              return false;

          n = nprev_cc;
          d = vprev_cc;
          s = vcurr; //这三个赋值会不会导致错误? 暂时认为不会 【未测试】
          return true;

#else //kinfu.orig
        float3 ncurr;
        ncurr.x = nmap_curr.ptr (y)[x];

        if (isnan (ncurr.x))
          return (false);

        float3 vcurr;
        vcurr.x = vmap_curr.ptr (y       )[x];
        vcurr.y = vmap_curr.ptr (y + rows)[x];
        vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

        float3 vcurr_g = Rcurr * vcurr + tcurr;

        float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);         // prev camera coo space

        int2 ukr;         //projection
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);      //4
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);                      //4

        //测试: 外部传参 Rprev_inv == Rcurr_inv 之后, 是否这里不必重新计算?   //2016-5-29 15:19:37
        //答: 正确！
        //ukr.x = x;
        //ukr.y = y;

        if (ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
          return (false);

        float3 nprev_g;
        nprev_g.x = nmap_g_prev.ptr (ukr.y)[ukr.x];

        if (isnan (nprev_g.x))
          return (false);

        float3 vprev_g;
        vprev_g.x = vmap_g_prev.ptr (ukr.y       )[ukr.x];
        vprev_g.y = vmap_g_prev.ptr (ukr.y + rows)[ukr.x];
        vprev_g.z = vmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x];

        float dist = norm (vprev_g - vcurr_g);
        if (dist > distThres)
          return (false);

        ncurr.y = nmap_curr.ptr (y + rows)[x];
        ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

        float3 ncurr_g = Rcurr * ncurr;

        nprev_g.y = nmap_g_prev.ptr (ukr.y + rows)[ukr.x];
        nprev_g.z = nmap_g_prev.ptr (ukr.y + 2 * rows)[ukr.x];

        float sine = norm (cross (ncurr_g, nprev_g));

        if (sine >= angleThres)
          return (false);
        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;
        return (true);
#endif //GCOO_2_CAMCOO
      }

      //sunguofei---contour cue //zhangxaochen: 1. 原来只是填充, 匹配对应点是在 cu 之外做的; 2. 稍改为: 增加 !qnan 判定 //2016-3-27 17:05:43
      __device__ __forceinline__ bool
      search_contourCue (int x, int y, float3& n, float3& d, float3& s) const
      {
        float3 vcurr;
        vcurr.x = vmap_contour.ptr (y       )[x];
        if(isnan(vcurr.x)) //cores_v_curr_new @kinfu.cpp 做了qnan填充, 因此此处要过滤
            return (false);
            ;
        vcurr.y = vmap_contour.ptr (y + rows_contour)[x];
        vcurr.z = vmap_contour.ptr (y + 2 * rows_contour)[x];

        float3 vcurr_g = Rcurr * vcurr + tcurr;

        float3 nprev_g;
        nprev_g.x = nmap_candidate.ptr (y)[x];

        if (isnan (nprev_g.x))
          return (false);

        float3 vprev_g;
        vprev_g.x = vmap_candidate.ptr (y       )[x];
        vprev_g.y = vmap_candidate.ptr (y + rows_contour)[x];
        vprev_g.z = vmap_candidate.ptr (y + 2 * rows_contour)[x];

        nprev_g.y = nmap_candidate.ptr (y + rows_contour)[x];
        nprev_g.z = nmap_candidate.ptr (y + 2 * rows_contour)[x];

        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;
        return (true);
      }

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        float3 n, d, s;
        bool found_coresp = false;

        if (x < cols && y < rows+rows_contour)
        {
            if (y<rows)
                found_coresp = search (x, y, n, d, s);
            else
            {
                found_coresp = search_contourCue (x, y-rows, n, d, s);
                n=n*weight;
            }
        }

        float row[7];

        if (found_coresp)
        {
          *(float3*)&row[0] = cross (s, n);
          *(float3*)&row[3] = n;
          row[6] = dot (n, d - s);
        }
        else
          row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

        __shared__ float_type smem[CTA_SIZE];
        int tid = Block::flattenedThreadId ();

        int shift = 0;
        for (int i = 0; i < 6; ++i)        //rows
        {
          #pragma unroll
          for (int j = i; j < 7; ++j)          // cols + b
          {
            __syncthreads ();
            smem[tid] = row[i] * row[j];
            __syncthreads ();

            reduce<CTA_SIZE>(smem);

            if (tid == 0)
              gbuf.ptr (shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
          }
        }
      }
    };

    __global__ void
    combinedKernel (const Combined cs) 
    {
      cs ();
    }

    //@brief 为了验证 combinedKernel 传参是否导致每个 kernel thread 构造一个 CombinedCCW 实例，又不想手写 CombinedCCW-copy-ctor
    struct CombinedCCWLog{
        CombinedCCWLog(){
            printf("CombinedCCWLog-ctor\n");
        }
    };

    __device__ int g_contCorrespCounter = 0;

    //@author zhangxaochen
    //@brief CCW, i.e., Contour-Cue increased Weight
    struct CombinedCCW : public Combined{
        PtrStepSz<unsigned char> contourMsk;
        float contWeight;
        int contCorrespCounter;
        //CombinedCCWLog tmpLog; //结果： default-ctor 前输出了 log； copy-ctor 没有

        CombinedCCW() : contWeight(-1), contCorrespCounter(0)
        {
            //printf("CombinedCCW-default-ctor~\n"); //默认构造, estimateCombined(host)中调用，非每个线程构造一次
                                                   //def-ctor, copy-ctor 中都故意不初始化 CombinedCCWLog， 但是其构造函数居然都调用了！
                                                   //copy-ctor 亦非每个线程构造一次。见：http://www.evernote.com/l/AY_3ITYZDqhPuZ4RuNiQzt6SjXKq9pcdvqw/
        }
        //CombinedCCW(const CombinedCCW &other)
        //{
        //    Rcurr = other.Rcurr;
        //    tcurr = other.tcurr;
        //    vmap_curr = other.vmap_curr;
        //    nmap_curr = other.nmap_curr;
        //    vmap_contour = other.vmap_contour;
        //    vmap_candidate = other.vmap_candidate;
        //    nmap_candidate = other.nmap_candidate;
        //    Rprev_inv = other.Rprev_inv;
        //    tprev = other.tprev;
        //    intr = other.intr;
        //    vmap_g_prev = other.vmap_g_prev;
        //    nmap_g_prev = other.nmap_g_prev;
        //    distThres = other.distThres;
        //    angleThres = other.angleThres;
        //    cols = other.cols;
        //    rows = other.rows;
        //    rows_contour = other.rows_contour;
        //    weight = other.weight;
        //    gbuf = other.gbuf;

        //    printf("CombinedCCW-copy-ctor~\n"); //
        //}

        __device__ __forceinline__ void
        operator () () const
        {
            int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
            int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

            float3 n, d, s;
            bool found_coresp = false;

            if (x < cols && y < rows)
                found_coresp = search (x, y, n, d, s);

            float row[7];

            if (found_coresp)
            {
                if(contWeight > 0){
                    unsigned char mskVal = contourMsk.ptr(y)[x];
                    if(mskVal != 0){
                        //atomicAdd(&contCorrespCounter, 1); //Error: unsupported operation, 可能 __device__ 原因？ 不懂，未解决！
                        atomicAdd(&g_contCorrespCounter, 1);
                        n = n * contWeight;
                        //printf("mskVal: %d, [%f, %f, %f]\n", mskVal, n.x, n.y, n.z);
                    }
                }

                *(float3*)&row[0] = cross (s, n);
                *(float3*)&row[3] = n;
                row[6] = dot (n, d - s);
            }
            else
                row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

            __shared__ float_type smem[CTA_SIZE];
            int tid = Block::flattenedThreadId ();

            int shift = 0;
            for (int i = 0; i < 6; ++i)        //rows
            {
#pragma unroll
                for (int j = i; j < 7; ++j)          // cols + b
                {
                    __syncthreads ();
                    smem[tid] = row[i] * row[j];
                    __syncthreads ();

                    reduce<CTA_SIZE>(smem);

                    if (tid == 0)
                        gbuf.ptr (shift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
                }
            }
        }//operator () () const
    };//struct CombinedCCW

    //@author zhangxaochen
    __global__ void
    combinedKernelCCW (const CombinedCCW cs) 
    {
        //g_contCorrespCounter = 0; //不行。永远是0
        cs ();
    }

    struct TranformReduction
    {
      enum
      {
        CTA_SIZE = 512,
        STRIDE = CTA_SIZE,

        B = 6, COLS = 6, ROWS = 6, DIAG = 6,
        UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG,
        TOTAL = UPPER_DIAG_MAT + B,

        GRID_X = TOTAL
      };

      PtrStep<float_type> gbuf;
      int length;
      mutable float_type* output;

      __device__ __forceinline__ void
      operator () () const
      {
        const float_type *beg = gbuf.ptr (blockIdx.x);
        const float_type *end = beg + length;

        int tid = threadIdx.x;

        float_type sum = 0.f;
        for (const float_type *t = beg + tid; t < end; t += STRIDE)
          sum += *t;

        __shared__ float_type smem[CTA_SIZE];

        smem[tid] = sum;
        __syncthreads ();

		reduce<CTA_SIZE>(smem);

        if (tid == 0)
          output[blockIdx.x] = smem[0];
      }
    };

    __global__ void
    TransformEstimatorKernel2 (const TranformReduction tr) 
    {
      tr ();
    }
  }//namespace device
}//namespace pcl


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::estimateCombined (const Mat33& Rcurr, const float3& tcurr, 
                               const MapArr& vmap_curr, const MapArr& nmap_curr, 
                               const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
                               const MapArr& vmap_g_prev, const MapArr& nmap_g_prev, 
                               float distThres, float angleThres,
                               DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf,
                               float_type* matrixA_host, float_type* vectorB_host)
{
  int cols = vmap_curr.cols ();
  int rows = vmap_curr.rows () / 3;

  Combined cs;

  cs.Rcurr = Rcurr;
  cs.tcurr = tcurr;

  cs.vmap_curr = vmap_curr;
  cs.nmap_curr = nmap_curr;

  cs.Rprev_inv = Rprev_inv;
  cs.tprev = tprev;

  cs.intr = intr;

  cs.vmap_g_prev = vmap_g_prev;
  cs.nmap_g_prev = nmap_g_prev;

  cs.distThres = distThres;
  cs.angleThres = angleThres;

  cs.cols = cols;
  cs.rows = rows;
  //sunguofei
  cs.rows_contour = 0;

//////////////////////////////

  dim3 block (Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  mbuf.create (TranformReduction::TOTAL);
  if (gbuf.rows () != TranformReduction::TOTAL || gbuf.cols () < (int)(grid.x * grid.y))
    gbuf.create (TranformReduction::TOTAL, grid.x * grid.y);

  cs.gbuf = gbuf;

  combinedKernel<<<grid, block>>>(cs);
  cudaSafeCall ( cudaGetLastError () );
  //cudaSafeCall(cudaDeviceSynchronize());

  //printFuncAttrib(combinedKernel);

  TranformReduction tr;
  tr.gbuf = gbuf;
  tr.length = grid.x * grid.y;
  tr.output = mbuf;

  TransformEstimatorKernel2<<<TranformReduction::TOTAL, TranformReduction::CTA_SIZE>>>(tr);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());

  float_type host_data[TranformReduction::TOTAL];
  mbuf.download (host_data);

  int shift = 0;
  for (int i = 0; i < 6; ++i)  //rows
    for (int j = i; j < 7; ++j)    // cols + b
    {
      float_type value = host_data[shift++];
      if (j == 6)       // vector b
        vectorB_host[i] = value;
      else
        matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
    }
}



//@author zhangxaochen
//@brief 比原 estimateCombined 函数增加 ①当前帧contourMsk；②contour权重参数contWeight(>1)
void
pcl::device::estimateCombined (const Mat33 &Rcurr, const float3 &tcurr, 
                               const MapArr &vmap_curr, const MapArr &nmap_curr, 
                               const Mat33 &Rprev_inv, const float3 &tprev, const Intr &intr,
                               const MapArr &vmap_g_prev, const MapArr &nmap_g_prev, 
                               float distThres, float angleThres,
                               DeviceArray2D<float_type> &gbuf, DeviceArray<float_type> &mbuf,
                               float_type *matrixA_host, float_type *vectorB_host,
                               DeviceArray2D<unsigned char> &contourMsk, float contWeight)
{
  int cols = vmap_curr.cols ();
  int rows = vmap_curr.rows () / 3;

  //zhangxaochen:
  assert(contWeight > 0);
  pcl::device::CombinedCCW cs; 
  cs.contWeight = contWeight;
  cs.contourMsk = contourMsk;

  cs.Rcurr = Rcurr;
  cs.tcurr = tcurr;

  cs.vmap_curr = vmap_curr;
  cs.nmap_curr = nmap_curr;

  cs.Rprev_inv = Rprev_inv;
  cs.tprev = tprev;

  cs.intr = intr;

  cs.vmap_g_prev = vmap_g_prev;
  cs.nmap_g_prev = nmap_g_prev;

  cs.distThres = distThres;
  cs.angleThres = angleThres;

  cs.cols = cols;
  cs.rows = rows;
  //sunguofei
  cs.rows_contour = 0;

//////////////////////////////

  dim3 block (Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  mbuf.create (TranformReduction::TOTAL);
  if (gbuf.rows () != TranformReduction::TOTAL || gbuf.cols () < (int)(grid.x * grid.y))
    gbuf.create (TranformReduction::TOTAL, grid.x * grid.y);

  cs.gbuf = gbuf;

  //g_contCorrespCounter = 0; //半错, 无法重新初始化归零，持续递增
                              //不应在 host code 操作。见我的问答：http://stackoverflow.com/questions/19185484/passing-value-from-device-memory-as-kernel-parameter-in-cuda
                            //首次迭代值 299, 可认作总量？
  int contCorrespCounter = 0;

  cudaMemcpyToSymbol(g_contCorrespCounter, &contCorrespCounter, sizeof(int)); //挂了, invalid device symbol
  //cudaMemset(&g_contCorrespCounter, 0, sizeof(int)); //挂了, invalid argument

  cudaMemcpyFromSymbol(&contCorrespCounter, g_contCorrespCounter, sizeof(int));
  printf("g_contCorrespCounter~: %d\n", contCorrespCounter);

  combinedKernelCCW<<<grid, block>>>(cs);

  //sync(); //不必
  //cudaMemcpy(&contCorrespCounter, &g_contCorrespCounter, sizeof(int), cudaMemcpyDeviceToHost); //错。必须用 cudaMemcpyFromSymbol
  cudaMemcpyFromSymbol(&contCorrespCounter, g_contCorrespCounter, sizeof(int));
  printf("g_contCorrespCounter: %d\n", contCorrespCounter);
  cudaSafeCall ( cudaGetLastError () );
  //cudaSafeCall(cudaDeviceSynchronize());

  //printFuncAttrib(combinedKernel);

  TranformReduction tr;
  tr.gbuf = gbuf;
  tr.length = grid.x * grid.y;
  tr.output = mbuf;

  TransformEstimatorKernel2<<<TranformReduction::TOTAL, TranformReduction::CTA_SIZE>>>(tr);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());

  float_type host_data[TranformReduction::TOTAL];
  mbuf.download (host_data);

  int shift = 0;
  for (int i = 0; i < 6; ++i)  //rows
    for (int j = i; j < 7; ++j)    // cols + b
    {
      float_type value = host_data[shift++];
      if (j == 6)       // vector b
        vectorB_host[i] = value;
      else
        matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
    }
}//estimateCombin (CCW)


//sunguofei
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::estimateCombined (const Mat33& Rcurr, const float3& tcurr, 
                               const MapArr& vmap_curr, const MapArr& nmap_curr, 
                               const MapArr& vmap_contour, const MapArr& vmap_candidate, const MapArr& nmap_candidate,
                               const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
                               const MapArr& vmap_g_prev, const MapArr& nmap_g_prev, 
                               float distThres, float angleThres,
                               DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf,
                               float_type* matrixA_host, float_type* vectorB_host,double weight)
{
  int cols = vmap_curr.cols ();
  int rows = vmap_curr.rows () / 3;
  int rows_contour = vmap_contour.rows () / 3;

  Combined cs;

  cs.Rcurr = Rcurr;
  cs.tcurr = tcurr;

  cs.vmap_curr = vmap_curr;
  cs.nmap_curr = nmap_curr;

  cs.vmap_contour = vmap_contour;
  cs.vmap_candidate = vmap_candidate;
  cs.nmap_candidate = nmap_candidate;

  cs.Rprev_inv = Rprev_inv;
  cs.tprev = tprev;

  cs.intr = intr;

  cs.vmap_g_prev = vmap_g_prev;
  cs.nmap_g_prev = nmap_g_prev;

  cs.distThres = distThres;
  cs.angleThres = angleThres;

  cs.cols = cols;
  cs.rows = rows;
  cs.rows_contour = rows_contour;
  cs.weight=weight;

//////////////////////////////

  dim3 block (Combined::CTA_SIZE_X, Combined::CTA_SIZE_Y);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows+rows_contour, block.y);

  mbuf.create (TranformReduction::TOTAL);
  if (gbuf.rows () != TranformReduction::TOTAL || gbuf.cols () < (int)(grid.x * grid.y))
    gbuf.create (TranformReduction::TOTAL, grid.x * grid.y);

  cs.gbuf = gbuf;

  combinedKernel<<<grid, block>>>(cs);
  cudaSafeCall ( cudaGetLastError () );
  //cudaSafeCall(cudaDeviceSynchronize());

  //printFuncAttrib(combinedKernel);

  TranformReduction tr;
  tr.gbuf = gbuf;
  tr.length = grid.x * grid.y;
  tr.output = mbuf;

  TransformEstimatorKernel2<<<TranformReduction::TOTAL, TranformReduction::CTA_SIZE>>>(tr);
  cudaSafeCall (cudaGetLastError ());
  cudaSafeCall (cudaDeviceSynchronize ());

  float_type host_data[TranformReduction::TOTAL];
  mbuf.download (host_data);

  int shift = 0;
  for (int i = 0; i < 6; ++i)  //rows
    for (int j = i; j < 7; ++j)    // cols + b
    {
      float_type value = host_data[shift++];
      if (j == 6)       // vector b
        vectorB_host[i] = value;
      else
        matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
    }
}
