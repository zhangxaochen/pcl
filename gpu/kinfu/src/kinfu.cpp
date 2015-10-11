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

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>

#ifdef HAVE_OPENCV
  #include <opencv2/opencv.hpp>
  #include <opencv2/gpu/gpu.hpp>
using namespace cv;
#endif

//zhangxaochen:
#include "zcUtility.h"

using namespace std;
using namespace pcl::device;
using namespace pcl::gpu;

using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;

namespace pcl
{
  namespace gpu
  {
    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::KinfuTracker::KinfuTracker (int rows, int cols) : rows_(rows), cols_(cols), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), disable_icp_(false)
{
  const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  const Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);
   
  tsdf_volume_ = TsdfVolume::Ptr( new TsdfVolume(volume_resolution) );
  tsdf_volume_->setSize(volume_size);
  
  setDepthIntrinsics (KINFU_DEFAULT_DEPTH_FOCAL_X, KINFU_DEFAULT_DEPTH_FOCAL_Y); // default values, can be overwritten
  
  init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {10, 5, 4};
  std::copy (iters, iters + LEVELS, icp_iterations_);

  const float default_distThres = 0.10f; //meters
  const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
  const float default_tranc_dist = 0.03f; //meters

  setIcpCorespFilteringParams (default_distThres, default_angleThres);
  tsdf_volume_->setTsdfTruncDist (default_tranc_dist);

  allocateBufffers (rows, cols);

  rmats_.reserve (30000);
  tvecs_.reserve (30000);

  reset ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthIntrinsics (float fx, float fy, float cx, float cy)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2-0.5f : cx;
  cy_ = (cy == -1) ? rows_/2-0.5f : cy;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getDepthIntrinsics (float& fx, float& fy, float& cx, float& cy)
{
  fx = fx_;
  fy = fy_;
  cx = cx_;
  cy = cy_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setInitalCameraPose (const Eigen::Affine3f& pose)
{
  init_Rcam_ = pose.rotation ();
  init_tcam_ = pose.translation ();
  reset ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthTruncationForICP (float max_icp_distance)
{
  max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setCameraMovementThreshold(float threshold)
{
  integration_metric_threshold_ = threshold;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setIcpCorespFilteringParams (float distThreshold, float sineOfAngle)
{
  distThres_  = distThreshold; //mm
  angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::rows ()
{
  return (rows_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::reset()
{
  if (global_time_){
    cout << "zhangxaochen: Reset" << endl;
  }

  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);

  tsdf_volume_->reset();
    
  if (color_volume_) // color integration mode is enabled
    color_volume_->reset();    
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::allocateBufffers (int rows, int cols)
{    
  depths_curr_.resize (LEVELS);
  vmaps_g_curr_.resize (LEVELS);
  nmaps_g_curr_.resize (LEVELS);

  vmaps_g_prev_.resize (LEVELS);
  nmaps_g_prev_.resize (LEVELS);

  vmaps_curr_.resize (LEVELS);
  nmaps_curr_.resize (LEVELS);

  coresps_.resize (LEVELS);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    depths_curr_[i].create (pyr_rows, pyr_cols);

    vmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);

    vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

    vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_curr_[i].create (pyr_rows*3, pyr_cols);

    coresps_[i].create (pyr_rows, pyr_cols);
  }  
  depthRawScaled_.create (rows, cols);
  // see estimate tranform for the magic numbers
  gbuf_.create (27, 20*60);
  sumbuf_.create (27);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth_raw, 
    Eigen::Affine3f *hint)
{  
  device::Intr intr (fx_, fy_, cx_, cy_);

  if (!disable_icp_)
  {
      ContourMask normal_mask;
      ContourMask mask;
      uchar* contour=(uchar*)malloc(640*480*sizeof(uchar));
      uchar* candidate=(uchar*)malloc(640*480*sizeof(uchar));
      vector<float> normals(640*480*3);
      vector<float> vertexes(640*480*3);
      vector<float> vertexes_curr(640*480*3);
      float position_camera_x,position_camera_y,position_camera_z;
      if (global_time_!=0)
      {
          position_camera_x=tvecs_[global_time_-1][0];
          position_camera_y=tvecs_[global_time_-1][1];
          position_camera_z=tvecs_[global_time_-1][2];
      }
      //std::vector<int> a(640*480);
      Mat contMskHost;
      {
        //ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
        //depth_raw.copyTo(depths_curr[0]);
        device::bilateralFilter (depth_raw, depths_curr_[0]);

        if (max_icp_distance_ > 0)
          device::truncateDepth(depths_curr_[0], max_icp_distance_);

        //zhangxaochen: test inpaint impl. both CPU & GPU version
        bool debugDraw = true;
        //zc::test::testInpaintImplCpuAndGpu(depths_curr_[0], debugDraw); //test OK.
        zc::inpaintGpu(depths_curr_[0], depths_curr_[0]);

        zc::computeContours(depths_curr_[0], contMsk_);
        contMskHost = Mat(contMsk_.rows(), contMsk_.cols(), CV_8UC1);
        contMsk_.download(contMskHost.data, contMskHost.cols * contMskHost.elemSize());
        
        Vector3f &tprev = tvecs_[global_time_ > 0 ? global_time_ - 1 : 0]; //  tranfrom from camera to global coo space for (i-1)th camera pose
        float3& device_tprev     = device_cast<float3> (tprev);

        DepthMap testDmap;
        MapArr testMarr;
        //zc::testPclCuda(testDmap, testMarr); //test OK
        if (global_time_ != 0)
            zc::contourCorrespCandidate(device_tprev, vmaps_g_prev_[0], nmaps_g_prev_[0], 75, contCorrespMsk_);

        //sunguofei---contour cue
        if (global_time_!=0)
        {
            device::computeContours(depths_curr_[0],mask);
        }

        for (int i = 1; i < LEVELS; ++i)
          device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

        for (int i = 0; i < LEVELS; ++i)
        {
          device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
          //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
          computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
        }
        if (global_time_!=0)
        {
            device::computeCandidate(nmaps_g_prev_[0],vmaps_g_prev_[0],position_camera_x,position_camera_y,position_camera_z,normal_mask,0.26);
        }
        pcl::device::sync ();
      }

      //build kd tree for vertex on the normal mask
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      vector<float> contour_candidate;
      vector<float> contour_candidate_normal;
      vector<float> contour_curr;
      if (global_time_!=0)
      {
          int c=640;
          mask.download(contour,c);
          normal_mask.download(candidate,c);
          //nmaps_g_prev_[0].download(normals,c);
          nmaps_g_prev_[0].download(normals,c);
          vmaps_g_prev_[0].download(vertexes,c);
          vmaps_curr_[0].download(vertexes_curr,c);

          //根据得到的数据，将contour candidate取出来，用来构建kd tree
          for (int i=0;i<vertexes.size()/3;++i)
          {
              if (candidate[i]==255)
              {
                  contour_candidate.push_back(vertexes[i]);
                  contour_candidate.push_back(vertexes[i+vertexes.size()/3]);
                  contour_candidate.push_back(vertexes[i+2*vertexes.size()/3]);


                  contour_candidate_normal.push_back(normals[i]);
                  contour_candidate_normal.push_back(normals[i+vertexes.size()/3]);
                  contour_candidate_normal.push_back(normals[i+2*vertexes.size()/3]);
              }
          }
          for (int i=0;i<vertexes_curr.size()/3;++i)
          {
              if(contour[i]==255)
              {
                  contour_curr.push_back(vertexes_curr[i]);
                  contour_curr.push_back(vertexes_curr[i+vertexes_curr.size()/3]);
                  contour_curr.push_back(vertexes_curr[i+2*vertexes_curr.size()/3]);
              }
          }

          Mat Contour_map=Mat::zeros(480,640,CV_8U);
          Mat N_map=Mat::zeros(480,640,CV_8U);
          Mat normal_map=Mat::zeros(480,640,CV_32FC3);
          for (int i=0;i<480;++i)
          {
              for (int j=0;j<640;++j)
              {
                  Contour_map.at<uchar>(i,j)=contour[i*640+j];
                  N_map.at<uchar>(i,j)=candidate[i*640+j];
                  normal_map.at<Vector3f>(i,j)[0]=normals[i*640+j];
                  normal_map.at<Vector3f>(i,j)[1]=normals[(i+480)*640+j];
                  normal_map.at<Vector3f>(i,j)[2]=normals[(i+480*2)*640+j];
                  //cout<<a[i*640+j]<<" ";
              }
              //cout<<endl;
          }
          Contour_map.setTo(128, contMskHost != 0);
          imshow("contours",Contour_map);
          imshow("candidates",N_map);
          imshow("normals",normal_map);
          //waitKey(1);
          free(contour);
          free(candidate);



          // Generate pointcloud data
          cloud->width = contour_candidate.size()/3;
          cloud->height = 1;
          cloud->points.resize (cloud->width * cloud->height);

          for (size_t i = 0; i < cloud->points.size (); ++i)
          {
              cloud->points[i].x = contour_candidate[i*3];
              cloud->points[i].y = contour_candidate[i*3+1];
              cloud->points[i].z = contour_candidate[i*3+2];
          }


          kdtree.setInputCloud (cloud);

      }


      //can't perform more on first frame
      if (global_time_ == 0)
      {
        Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

        //sunguofei
        if (hint)
        {
            Matrix3frm init_Rcam_1 = hint->rotation().matrix();
            Vector3f init_tcam_1 = hint->translation().matrix();
            //根据第一帧时两个不同的初始R和t计算两个坐标系之间变换的dR和dt
            d_R = init_Rcam*init_Rcam_1.inverse();
            d_t = -d_R*init_tcam_1+init_tcam;
        }


        Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
        float3& device_tcam = device_cast<float3>(init_tcam);

        Matrix3frm init_Rcam_inv = init_Rcam.inverse ();
        Mat33&   device_Rcam_inv = device_cast<Mat33> (init_Rcam_inv);
        float3 device_volume_size = device_cast<const float3>(tsdf_volume_->getSize());

        //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tranc_dist, volume_);    
        device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_);

        for (int i = 0; i < LEVELS; ++i)
          device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);

        ++global_time_;
        return (false);
      }

      ///////////////////////////////////////////////////////////////////////////////////////////
      // Iterative Closest Point
      Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
      Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
      Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();

      //Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
      Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
      float3& device_tprev     = device_cast<float3> (tprev);
      Matrix3frm Rcurr,Rcurr_back;
      Vector3f tcurr,tcurr_back;
      if(hint)
      {
//         Rcurr = hint->rotation().matrix();
//         tcurr = hint->translation().matrix()+d_t;
//         Rcurr_back = hint->rotation().matrix();
//         tcurr_back = hint->translation().matrix()+d_t;
        
        //sunguofei
        //利用外部设备得到的ΔR和Δt，计算这一帧的初值
        Matrix3frm dr=hint->rotation().matrix();
        Vector3f dt=hint->translation().matrix();
        Rcurr=d_R*dr*Rprev;
        tcurr=d_R*dt+tprev;
        Rcurr_back=Rcurr;
        tcurr_back=tcurr;
      }
      else
      {
        Rcurr = Rprev; // tranform to global coo for ith camera pose
        tcurr = tprev;
        //Rcurr_back = init_Rcam_;
        //tcurr_back = init_tcam_;
      }
      const double 
          condThresh = 1e2,
          INVALID_COND = 1e8;
      double cond = INVALID_COND;

      bool illCondMat = false;
      {
        //ScopeTime time("icp-all");
        for (int level_index = LEVELS-1; level_index>=0; --level_index)
        {
          if(hint && illCondMat)
              break;
          int iter_num = icp_iterations_[level_index];

          MapArr& vmap_curr = vmaps_curr_[level_index];
          MapArr& nmap_curr = nmaps_curr_[level_index];

          //MapArr& vmap_g_curr = vmaps_g_curr_[level_index];
          //MapArr& nmap_g_curr = nmaps_g_curr_[level_index];

          MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
          MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

          //CorespMap& coresp = coresps_[level_index];

          for (int iter = 0; iter < iter_num; ++iter)
          {
            Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3& device_tcurr = device_cast<float3>(tcurr);

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
            Eigen::Matrix<double, 6, 1> b;
    #if 0
            device::tranformMaps(vmap_curr, nmap_curr, device_Rcurr, device_tcurr, vmap_g_curr, nmap_g_curr);
            findCoresp(vmap_g_curr, nmap_g_curr, device_Rprev_inv, device_tprev, intr(level_index), vmap_g_prev, nmap_g_prev, distThres_, angleThres_, coresp);
            device::estimateTransform(vmap_g_prev, nmap_g_prev, vmap_g_curr, coresp, gbuf_, sumbuf_, A.data(), b.data());

            //cv::gpu::GpuMat ma(coresp.rows(), coresp.cols(), CV_32S, coresp.ptr(), coresp.step());
            //cv::Mat cpu;
            //ma.download(cpu);
            //cv::imshow(names[level_index] + string(" --- coresp white == -1"), cpu == -1);
    #else
            //sunguofei---contour cue
            vector<float> cores_v_curr;
            vector<float> cores_v_prev;
            vector<float> cores_n_prev;
            //对当前深度图中的contour上的点寻找对应点，并按顺序存起来
            for (int i=0;i<contour_curr.size()/3;++i)
            {
                pcl::PointXYZ searchpoint;
                //每个点是在当前相机坐标系下的，根据R_curr和t_curr，将他们转换到世界坐标系下（因为上一帧模型得到的点都是在世界坐标系下的）
                Vector3f point;
                point(0,0)=contour_curr[i*3];
                point(1,0)=contour_curr[i*3+1];
                point(2,0)=contour_curr[i*3+2];
                point=Rcurr*point+tcurr;
                searchpoint.x=point(0,0);
                searchpoint.y=point(1,0);
                searchpoint.z=point(2,0);

                vector<int> pointIdxSearch(1);
                vector<float> pointSquaredDistance(1);

                float radius = 0.1;

                if ( kdtree.nearestKSearch (searchpoint, 1, pointIdxSearch, pointSquaredDistance) > 0 )
                {
                    if (pointSquaredDistance[0]<=radius*radius)
                    {
                        //找到满足要求的对应点，将他们存在vector中，下标相同的点是一组对应点
                        cores_v_curr.push_back(contour_curr[i*3]);
                        cores_v_curr.push_back(contour_curr[i*3+1]);
                        cores_v_curr.push_back(contour_curr[i*3+2]);

                        cores_v_prev.push_back(cloud->points[ pointIdxSearch[0] ].x);
                        cores_v_prev.push_back(cloud->points[ pointIdxSearch[0] ].y);
                        cores_v_prev.push_back(cloud->points[ pointIdxSearch[0] ].z);

                        cores_n_prev.push_back(contour_candidate_normal[ pointIdxSearch[0]*3 ]);
                        cores_n_prev.push_back(contour_candidate_normal[ pointIdxSearch[0]*3+1 ]);
                        cores_n_prev.push_back(contour_candidate_normal[ pointIdxSearch[0]*3+2 ]);
                    }
                }
            }
            //找到对应关系之后，将其传入下边函数中，在里边进行计算
            //两种方法：
            //1、每一层金字塔都计算contour点和candidate点
            //2、只在最外层金字塔计算contour点和candidate点，目前的解决思路：把已找到的对应关系存到maparr中，列数和当前vmap nmap相同，直接传入estimateCombined中
            //   需要添加三个新的量，vmap_candidate,nmap_candidate,vmap_contour

            //为了能够把对应点按照要求传进去，做两个事
            //1、重新排序，现在是xyzxyz...，改成xxx..yyy..zzz..
            //2、除以当前map的列数，去掉余出来的部分
            int contour_size=cores_v_curr.size()/3/int(vmap_curr.cols())*int(vmap_curr.cols());
            vector<float> cores_n_prev_new,cores_v_prev_new,cores_v_curr_new;
            for (int k1=0;k1<3;++k1)
            {
                for (int k2=0;k2<contour_size;++k2)
                {
                    cores_v_curr_new.push_back(cores_v_curr[k2*3+k1]);
                    cores_v_prev_new.push_back(cores_v_prev[k2*3+k1]);
                    cores_n_prev_new.push_back(cores_n_prev[k2*3+k1]);
                }
            }
            if (cores_v_curr_new.size()==0)
            //if (1)
            {
                estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
                              vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
            }
            else
            {
                int step=vmap_curr.cols();
                MapArr vmap_candidate,nmap_candidate,vmap_contour;
                vmap_candidate.upload(cores_v_prev_new,step);
                nmap_candidate.upload(cores_n_prev_new,step);
                vmap_contour.upload(cores_v_curr_new,step);
                //test
                vector<float> tmp;
                nmap_candidate.download(tmp,step);

                estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, vmap_contour, vmap_candidate, nmap_candidate, device_Rprev_inv, device_tprev, intr (level_index),
                    vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data (),2.0);
            }
    #endif
            //checking nullspace
            double det = A.determinant ();

            //sunguofei
            Eigen::Matrix<double,6,6,Eigen::RowMajor>::EigenvaluesReturnType eigenvalues = A.eigenvalues();
            cond=eigenvalues(0,0).real()/eigenvalues(5,0).real();
            cout << "eigenvalues: " << eigenvalues << endl;
            cond=sqrt(cond);
            cout<<"condition of A: "<<cond<<endl;
//             if (cond>100)
//             {
//                 Rcurr=Rcurr_back;tcurr=tcurr_back;
//                 break;
//             }

            if(hint && cond > condThresh){
                cond = INVALID_COND;
                illCondMat = true;
                break; //仍然往下走，算一下
            }

            if (fabs (det) < 1e-15 || pcl_isnan (det))
            //if (fabs (det) < 1e-15 || pcl_isnan (det) || (hint && cond>INVALID_COND) )
            {
              if (pcl_isnan (det)) cout << "qnan" << endl;

               reset ();
               return (false);

              //sunguofei
              //不进行重置，而是就用外部设备得到的RT作为输出结果
              //Rcurr=Rcurr_back;tcurr=tcurr_back;
              //cond=INVALID_COND;
              //illCondMat = true;
              //if(hint)
              //    break; //仍然往下走，算一下
              //else{
              //    reset ();
              //    return (false);
              //}
            }
            //float maxc = A.maxCoeff();

            Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
            //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

            float alpha = result (0);
            float beta  = result (1);
            float gamma = result (2);

            Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
            Vector3f tinc = result.tail<3> ();

            //compose
            tcurr = Rinc * tcurr + tinc;
            Rcurr = Rinc * Rcurr;
//             if(illCondMat)
//                 break; //仍然往下走，算一下
          }
        }
      }
      //save tranform
      //if (cond>1e5)
      if(hint && illCondMat)
      {
          rmats_.push_back (Rcurr_back);
          tvecs_.push_back (tcurr_back);
      }
      else
      {
          rmats_.push_back (Rcurr);
          tvecs_.push_back (tcurr);
      }

      //sunguofei
      //rmats_.push_back (Rcurr_back);
      //tvecs_.push_back (tcurr_back);
  } 
  else /* if (disable_icp_) */
  {
      if (global_time_ == 0)
        ++global_time_;

      Matrix3frm Rcurr = rmats_[global_time_ - 1];
      Vector3f   tcurr = tvecs_[global_time_ - 1];

      rmats_.push_back (Rcurr);
      tvecs_.push_back (tcurr);

  }

  Matrix3frm Rprev = rmats_[global_time_ - 1];
  Vector3f   tprev = tvecs_[global_time_ - 1];

  Matrix3frm Rcurr = rmats_.back();
  Vector3f   tcurr = tvecs_.back();

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move.  
  float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
  float tnorm = (tcurr - tprev).norm();  
  const float alpha = 1.f;
  bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;

  if (disable_icp_)
    integrate = true;

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Volume integration
  float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

  Matrix3frm Rcurr_inv = Rcurr.inverse ();
  Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
  float3& device_tcurr = device_cast<float3> (tcurr);
  if (integrate)
  {
    //ScopeTime time("tsdf");
    //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
    integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), depthRawScaled_);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Ray casting
  Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);
  {
    //ScopeTime time("ray-cast-all");
    raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
    for (int i = 1; i < LEVELS; ++i)
    {
      resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
      resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
    }
    pcl::device::sync ();
  }

  ++global_time_;
  return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::KinfuTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time];
  aff.translation () = tvecs_[time];
  return (aff);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t
pcl::gpu::KinfuTracker::getNumberOfPoses () const
{
  return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const TsdfVolume& 
pcl::gpu::KinfuTracker::volume() const 
{ 
  return *tsdf_volume_; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TsdfVolume& 
pcl::gpu::KinfuTracker::volume()
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const ColorVolume& 
pcl::gpu::KinfuTracker::colorVolume() const
{
  return *color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ColorVolume& 
pcl::gpu::KinfuTracker::colorVolume()
{
  return *color_volume_;
}
     
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getImage (View& view) const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

  device::LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  view.create (rows_, cols_);
  generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  device::convert (vmaps_g_prev_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
  normals.create (rows_, cols_);
  DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
  device::convert (nmaps_g_prev_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::gpu::KinfuTracker::disableIcp() { disable_icp_ = true; }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::initColorIntegration(int max_weight)
{     
  color_volume_ = pcl::gpu::ColorVolume::Ptr( new ColorVolume(*tsdf_volume_, max_weight) );  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool 
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth, const View& colors)
{ 
  bool res = (*this)(depth);

  if (res && color_volume_)
  {
    const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
    device::Intr intr(fx_, fy_, cx_, cy_);

    Matrix3frm R_inv = rmats_.back().inverse();
    Vector3f   t     = tvecs_.back();
    
    Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
    float3& device_tcurr = device_cast<float3> (t);
    
    device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0], 
        colors, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
  }

  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace gpu
  {
    PCL_EXPORTS void 
    paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
    {
      device::paint3DView(rgb24, view, colors_weight);
    }

    PCL_EXPORTS void
    mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
    {
      const size_t size = min(cloud.size(), normals.size());
      output.create(size);

      const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
      const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
      const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
      device::mergePointNormal(c, n, o);           
    }

    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
    {
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
      Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

      double rx = R(2, 1) - R(1, 2);
      double ry = R(0, 2) - R(2, 0);
      double rz = R(1, 0) - R(0, 1);

      double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
      double c = (R.trace() - 1) * 0.5;
      c = c > 1. ? 1. : c < -1. ? -1. : c;

      double theta = acos(c);

      if( s < 1e-5 )
      {
        double t;

        if( c > 0 )
          rx = ry = rz = 0;
        else
        {
          t = (R(0, 0) + 1)*0.5;
          rx = sqrt( std::max(t, 0.0) );
          t = (R(1, 1) + 1)*0.5;
          ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
          t = (R(2, 2) + 1)*0.5;
          rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

          if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
            rz = -rz;
          theta /= sqrt(rx*rx + ry*ry + rz*rz);
          rx *= theta;
          ry *= theta;
          rz *= theta;
        }
      }
      else
      {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
      }
      return Eigen::Vector3d(rx, ry, rz).cast<float>();
    }
  }
}
