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

#include "device.hpp"

namespace pcl
{
  namespace device
  {
    const float sigma_color = 30;     //in mm
    const float sigma_space = 4.5;     // in pixels

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    bilateralKernel (const PtrStepSz<ushort> src, 
                     PtrStep<ushort> dst, 
                     float sigma_space2_inv_half, float sigma_color2_inv_half)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= src.cols || y >= src.rows)
        return;

      const int R = 6;       //static_cast<int>(sigma_space * 1.5);
      const int D = R * 2 + 1;

      int value = src.ptr (y)[x];

      int tx = min (x - D / 2 + D, src.cols - 1);
      int ty = min (y - D / 2 + D, src.rows - 1);

      float sum1 = 0;
      float sum2 = 0;

      for (int cy = max (y - D / 2, 0); cy < ty; ++cy)
      {
        for (int cx = max (x - D / 2, 0); cx < tx; ++cx)
        {
          int tmp = src.ptr (cy)[cx];

          float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          float color2 = (value - tmp) * (value - tmp);

          float weight = __expf (-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

          sum1 += tmp * weight;
          sum2 += weight;
        }
      }

      int res = __float2int_rn (sum1 / sum2);
      dst.ptr (y)[x] = max (0, min (res, numeric_limits<short>::max ()));
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    pyrDownGaussKernel (const PtrStepSz<ushort> src, PtrStepSz<ushort> dst, float sigma_color)
    {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x >= dst.cols || y >= dst.rows)
        return;

      const int D = 5;

      int center = src.ptr (2 * y)[2 * x];

      int x_mi = max(0, 2*x - D/2) - 2*x;
      int y_mi = max(0, 2*y - D/2) - 2*y;

      int x_ma = min(src.cols, 2*x -D/2+D) - 2*x;
      int y_ma = min(src.rows, 2*y -D/2+D) - 2*y;
            
      float sum = 0;
      float wall = 0;
      
      float weights[] = {0.375f, 0.25f, 0.0625f} ;

      for(int yi = y_mi; yi < y_ma; ++yi)
          for(int xi = x_mi; xi < x_ma; ++xi)
          {
              int val = src.ptr (2*y + yi)[2*x + xi];

              if (abs (val - center) < 3 * sigma_color)
              {                                 
                sum += val * weights[abs(xi)] * weights[abs(yi)];
                wall += weights[abs(xi)] * weights[abs(yi)];
              }
          }


      dst.ptr (y)[x] = static_cast<int>(sum /wall);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    pyrDownKernel (const PtrStepSz<ushort> src, PtrStepSz<ushort> dst, float sigma_color)
    {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x >= dst.cols || y >= dst.rows)
        return;

      const int D = 5;

      int center = src.ptr (2 * y)[2 * x];

      int tx = min (2 * x - D / 2 + D, src.cols - 1);
      int ty = min (2 * y - D / 2 + D, src.rows - 1);
      int cy = max (0, 2 * y - D / 2);

      int sum = 0;
      int count = 0;

      for (; cy < ty; ++cy)
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
          int val = src.ptr (cy)[cx];
          if (abs (val - center) < 3 * sigma_color)
          {
            sum += val;
            ++count;
          }
        }
      dst.ptr (y)[x] = sum / count;
    }

	__global__ void
    truncateDepthKernel(PtrStepSz<ushort> depth, ushort max_distance_mm)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < depth.cols && y < depth.rows)		
			if(depth.ptr(y)[x] > max_distance_mm)
				depth.ptr(y)[x] = 0;
	}

    //sunguofei---contour cue

    __global__ void
    computeContoursKernel(const PtrStepSz<ushort>& src, PtrStepSz<_uchar> dst, int thresh)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < src.cols && y < src.rows)
        {
            dst.ptr(y)[x]=0;
            int x_left=max(0,x-1),x_right=min(x+1,src.cols-1),y_top=max(0,y-1),y_down=min(y+1,src.cols-1);
            int val=src.ptr(y)[x];
            int d[8];
            d[0]=src.ptr(y)[x_left];d[1]=src.ptr(y_top)[x_left];d[2]=src.ptr(y_top)[x];d[3]=src.ptr(y_top)[x_right];
            d[4]=src.ptr(y)[x_right];d[5]=src.ptr(y_down)[x_right];d[6]=src.ptr(y_down)[x];d[7]=src.ptr(y_down)[x_left];
            for(int i=0;i<8;++i)
            {
                if (d[i]-val>thresh && val!=0)
                {
                    dst.ptr(y)[x]=255;//如果满足轮廓要求，则mask值为255，否则为0
                    break;
                }
            }
        }
	}

    __global__ void
    computeCandidateKernel(const PtrStepSz<float>& nmap,const PtrStepSz<float>& vmap,float t_x,float t_y,float t_z,PtrStepSz<_uchar> dst,double thresh)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x<dst.cols && y<dst.rows)
        {
            dst.ptr(y)[x]=0;
            double x1=nmap.ptr(y)[x],y1=nmap.ptr(y+dst.rows)[x],z1=nmap.ptr(y+2*dst.rows)[x];
            double x2=t_x-vmap.ptr(y)[x],y2=t_y-vmap.ptr(y+dst.rows)[x],z2=t_z-vmap.ptr(y+2*dst.rows)[x];
            double mod1=sqrt(x1*x1+y1*y1+z1*z1),mod2=sqrt(x2*x2+y2*y2+z2*z2);
            if (mod1>1e-3&&mod2>1e-3)
            {
                double res=(x1*x2+y1*y2+z1*z2)/(mod1*mod2);
                if (abs(res)<thresh)
                    dst.ptr(y)[x]=255;//如果满足轮廓要求，则mask值为255，否则为0
            }
        }
    }

    __global__ void
    inpaintKernel(const PtrStepSz<ushort>& src,PtrStepSz<ushort> dst)
    {
        int x=blockIdx.x*blockDim.x+threadIdx.x;
        if(x<dst.rows)
        {
            int depth_paint=0;
            bool flag=true;
            for(int y=0;y<dst.cols;++y)
            {
                int depth=src.ptr(x)[y];
                if(depth!=0)
                {
                    dst.ptr(x)[y]=depth;
                    if(flag)
                        depth_paint=depth;
                    else
                    {
                        depth_paint=max(depth,depth_paint);
                        for(int i=y-1;y>=0;--y)
                        {
                            if (dst.ptr(x)[i]!=0)
                                dst.ptr(x)[i]=depth_paint;
                            else
                                break;
                        }
                        flag=true;
                    }
                }
                else
                {
                    dst.ptr(x)[y]=0;
                    flag=false;
                }
            }
        }
    }

    __global__ void
    computeNormalsContourcueKernal(const Intr& intr, const PtrStepSz<ushort>& src,const PtrStepSz<float>& grandient_x,const PtrStepSz<float>& grandient_y,PtrStepSz<ushort> dst)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x<src.cols && y<src.rows)
        {
            float dx,dy,dz;
            dz = -1;
            //先计算逆矩阵
            float fx = intr.fx,fy = intr.fy,cx = intr.cx,cy = intr.cy;
            int depth = src.ptr(y)[x];
            float du = grandient_x.ptr(y)[x],dv=grandient_y.ptr(y)[x];
            float m00 = 1.0/fx*(depth+(x-cx)*du),
                m01 = 1.0/fx*(x-cx)*dv,
                m10 = 1.0/fy*(y-cy)*du,
                m11 = 1.0/fy*(depth+(y-cy)*dv);
            float det = m00*m11-m01*m10;
            if(abs(det) < 1e-2)
            {
                dx = 0;dy = 0;
            }
            else
            {
                float m00_inv = m11/det,
                    m01_inv = -m01/det,
                    m10_inv = -m10/det,
                    m11_inv = m00/det;
                dx = du*m00_inv+dv*m10_inv;
                dy = du*m01_inv+dv*m11_inv;
                //normalize
                float norm=sqrt(dx*dx+dy*dy+dz*dz);
                dx = dx/norm;
                dy = dy/norm;
                dz = dz/norm;
            }
        }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::bilateralFilter (const DepthMap& src, DepthMap& dst)
{
  dim3 block (32, 8);
  dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

  cudaFuncSetCacheConfig (bilateralKernel, cudaFuncCachePreferL1);
  bilateralKernel<<<grid, block>>>(src, dst, 0.5f / (sigma_space * sigma_space), 0.5f / (sigma_color * sigma_color));

  cudaSafeCall ( cudaGetLastError () );
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::pyrDown (const DepthMap& src, DepthMap& dst)
{
  dst.create (src.rows () / 2, src.cols () / 2);

  dim3 block (32, 8);
  dim3 grid (divUp (dst.cols (), block.x), divUp (dst.rows (), block.y));

  //pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
  pyrDownKernel<<<grid, block>>>(src, dst, sigma_color);
  cudaSafeCall ( cudaGetLastError () );
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::device::truncateDepth(DepthMap& depth, float max_distance)
{
  dim3 block (32, 8);
  dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

  truncateDepthKernel<<<grid, block>>>(depth, static_cast<ushort>(max_distance * 1000.f));

  cudaSafeCall ( cudaGetLastError () );
}

//sunguofei---contour cue

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::device::computeContours(const DepthMap& src,ContourMask& dst)
{
  dim3 block (32, 8);
  dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));
  dst.create(src.rows(),src.cols());
  computeContoursKernel<<<grid, block>>>(src, dst, 50);

  cudaSafeCall ( cudaGetLastError () );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::device::computeCandidate(const MapArr& nmap, const MapArr& vmap, float t_x,float t_y,float t_z,ContourMask& dst,double thresh)
{
  dst.create(nmap.rows ()/3, nmap.cols ());
  dim3 block (32, 8);
  dim3 grid (divUp (nmap.cols (), block.x), divUp (nmap.rows (), block.y));

  computeCandidateKernel<<<grid, block>>>(nmap,vmap, t_x,t_y,t_z, dst, thresh);

  cudaSafeCall ( cudaGetLastError () );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::device::inpaint(const DepthMap& src,DepthMap& dst)
{
  dst.create(src.rows(), src.cols());
  dim3 block (32, 8);
  dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

  inpaintKernel<<<grid, block>>>(src, dst);

  cudaSafeCall ( cudaGetLastError () );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void pcl::device::computeNormalsContourcue(const Intr& intr, const DepthMap& depth,const MapArr& grandient_x, const MapArr& grandient_y, MapArr& nmap)
{
    nmap.create(depth.rows()*3,depth.cols());
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

    computeNormalsContourcueKernal<<<grid,block>>>(intr,depth,grandient_x,grandient_y,nmap);

    cudaSafeCall ( cudaGetLastError () );
}