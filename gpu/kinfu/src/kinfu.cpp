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

#include <pcl/filters/extract_indices.h>
//#include <pcl/>

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
using zc::ScopeTimeMicroSec;
using Eigen::Affine3f;
using Eigen::Matrix3f;

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
    ,synModelCloudPtr_(new PointCloud<PointXYZ>), edgeCloud_(new PointCloud<PointXYZ>), edgeCloudVisible_(new PointCloud<PointXYZ>) //zhangxaochen //2016-4-10 22:03:53
{
  //zhangxaochen:
  icp_orig_ = true;
  icp_sgf_cpu_ = false;
  icp_cc_inc_weight = false;
  contWeight_ = 1;
  cc_norm_prev_way_ = 0;

  pRaycaster_ = RayCaster::Ptr(new RayCaster(rows_, cols_, KINFU_DEFAULT_DEPTH_FOCAL_X, KINFU_DEFAULT_DEPTH_FOCAL_Y));

  regObjId_ = 0; //2016-4-20 21:11:39

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

  //2016-4-20 14:50:40
  //if(regObjId_ == 1){//ʵ����һ��Ӱ�� TSDF ��ģ��, �����cube��TSDF, ���˷�
  //ȥ�� if��������Ϊ�˴� regObjId_ ��δ��ʼ��, ����ֱ��Ԥ����500MB�ڴ�
      tsdf_volume_shadow_ = TsdfVolume::Ptr(new TsdfVolume(volume_resolution));
      tsdf_volume_shadow_->setSize(volume_size);
      tsdf_volume_shadow_->setTsdfTruncDist(default_tranc_dist);
      tsdf_volume_shadow_->reset();
  //}
}//KinfuTracker-ctor

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

  //zhangxaochen: //2016-4-21 16:25:32
  vmap_g_model_.create(rows * 3, cols);
  nmap_g_model_.create(rows * 3, cols);

  //sunguofei
  depths_prev_.resize(LEVELS);
  nmaps_g_prev_contourcue.resize(LEVELS);

  vmaps_curr_.resize (LEVELS);
  nmaps_curr_.resize (LEVELS);

  coresps_.resize (LEVELS);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    depths_curr_[i].create (pyr_rows, pyr_cols);

    //sunguofei
    depths_prev_[i].create(pyr_rows,pyr_cols);
    nmaps_g_prev_contourcue[i].create(pyr_rows*3,pyr_cols);

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

  //zhangxaochen: ���ü����� //2016-7-12 21:31:02
  static int callCnt = 0;
  callCnt++;

  if (!disable_icp_)
  {
      //���ص�����������, ����������ת�� ���ͼ
      if(synModelCloudPtr_->size() > 0){
          Affine3f pose = this->getCameraPose();
          cv::Mat depFromCloud;
          //{
          //zc::ScopeTimeMicroSec time("zc::cloud2depth"); //23ms
          zc::cloud2depth(*synModelCloudPtr_, pose, intr, cols_, rows_, depFromCloud); //��
          //}
          //{
          //zc::ScopeTimeMicroSec time("zc::cloud2depthCPU"); //240ms
          //zc::cloud2depthCPU(*synModelCloudPtr_, pose, intr, 640, 480, depFromCloud);//��
          //}

          if(edgeIdxVec_.size() > 0 && edgeCloud_->size() == 0){ //������ -edgeIdx, ��txt��ȷʵ������, �ҵ�һ�ε��� if
              ExtractIndices<PointXYZ> extractInd;
              extractInd.setInputCloud(synModelCloudPtr_);
              extractInd.setIndices(boost::make_shared<vector<int>>(edgeIdxVec_));
              extractInd.filter(*edgeCloud_);
          }
          edgeCloudVisible_.reset(new PointCloud<PointXYZ>); //������, ��ֹ�ۼ�
          zc::raycastCloudSubset(*edgeCloud_, pose, intr, depFromCloud, *edgeCloudVisible_);

          cv::Mat dfc8u(depFromCloud.rows, depFromCloud.cols, CV_8UC1);
          double dmin, dmax;
          minMaxLoc(depFromCloud, &dmin, &dmax);
          depFromCloud.convertTo(dfc8u, CV_8UC1, 255./(dmax-dmin), -dmin*255./(dmax-dmin));
          cv::imshow("dfc8u", dfc8u);
      }

      ContourMask normal_mask;
      ContourMask mask;
      uchar* contour=(uchar*)malloc(640*480*sizeof(uchar)); //contour of frame i
      uchar* candidate=(uchar*)malloc(640*480*sizeof(uchar));
      vector<float> normals(640*480*3);
      vector<float> vertexes(640*480*3); //model synthesized, i.e., frame i-1
      vector<float> vertexes_curr(640*480*3); //frame i
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
        {
        //ScopeTimeMicroSec time("zc-inpaintGpu+computeContours+contourCorrespCandidate"); //3ms, ���������ĸ� ScopeTimeMicroSec �ᵼ��ʱ������Ϊ 15ms

        bool debugDraw = true;
        //zc::test::testInpaintImplCpuAndGpu(depths_curr_[0], debugDraw); //test OK.

        {
        //ScopeTimeMicroSec time("|-zc-inpaintGpu"); //2ms
        //zc::inpaintGpu(depths_curr_[0], depths_curr_[0]); //��Ҫ inpaint ����Դ, 
        zc::inpaintGpu(depths_curr_[0], dmatInp_);
        }

        {
        //ScopeTimeMicroSec time("|-zc-computeContours"); //0ms
        zc::computeContours(dmatInp_, contMsk_);
        }
        {
        //ScopeTimeMicroSec time("|-zc-contMsk_.download"); //1ms
        contMskHost = Mat(contMsk_.rows(), contMsk_.cols(), CV_8UC1);
        contMsk_.download(contMskHost.data, contMskHost.cols * contMskHost.elemSize());
        }

        Vector3f &tprev = tvecs_[global_time_ > 0 ? global_time_ - 1 : 0]; //  tranfrom from camera to global coo space for (i-1)th camera pose
        float3& device_tprev     = device_cast<float3> (tprev);

        //DepthMap testDmap;
        //MapArr testMarr;
        //zc::testPclCuda(testDmap, testMarr); //test OK
        if (global_time_ != 0){
            {
            //ScopeTimeMicroSec time("|-zc-contourCorrespCandidate"); //1ms
            //zc::contourCorrespCandidate(device_tprev, vmaps_g_prev_[0], nmaps_g_prev_[0], 75, contCorrespMsk_);
            zc::contourCorrespCandidate(device_tprev, vmaps_g_prev_[0], nmap_g_prev_choose_, 75, contCorrespMsk_);
            }
        }
        }//ScopeTimeMicroSec time "zc..."

        //sunguofei---contour cue
        if (global_time_!=0)
        {
            {
            //ScopeTimeMicroSec time("|-sgf-computeContours"); //0ms
            device::computeContours(depths_curr_[0],mask);
            }
        }

        for (int i = 1; i < LEVELS; ++i)
          device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

        for (int i = 0; i < LEVELS; ++i)
        {
          device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
          //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
          computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
        }
        Mat depth_prev_host;
        if (global_time_!=0)
        {
#if 0
            {
            depth_prev_host=Mat::zeros(depths_prev_[0].rows(),depths_prev_[0].cols(),CV_16U);
            depths_prev_[0].download(depth_prev_host.data,depth_prev_host.cols*depth_prev_host.elemSize());
            Mat grandient_x,grandient_y;
            Sobel(depth_prev_host,grandient_x,CV_32F,1,0,7);
            Sobel(depth_prev_host,grandient_y,CV_32F,0,1,7);
            //cout<<grandient_y<<endl;
            MapArr grandient_x_device,grandient_y_device;
            grandient_x_device.upload(grandient_x.data,grandient_x.cols*grandient_x.elemSize(),grandient_x.rows,grandient_x.cols);
            grandient_y_device.upload(grandient_y.data,grandient_y.cols*grandient_y.elemSize(),grandient_y.rows,grandient_y.cols);
            device::computeNormalsContourcue(intr(0),depths_prev_[0],grandient_x_device,grandient_y_device,prev_normals);
            vector<float> normals_curr_coo;
            int cols_normal;
            prev_normals.download(normals_curr_coo,cols_normal);
            //��ǰ�������ϵ�µķ���ת����������ϵ�µķ���
            float3 t_tmp;
            t_tmp.x=t_tmp.y=t_tmp.z=0;
            //const Mat33 &R_prev = device_cast<const Mat33>(rmats_[global_time_-1]);
            Affine3f prevPose = this->getCameraPose();
            Matrix3frm Rrm = prevPose.linear();
            const Mat33 &R_prev = device_cast<const Mat33>(Rrm);
            MapArr prev_normals_word_coo;
            zc::transformVmap(prev_normals,R_prev,t_tmp,prev_normals_word_coo);
            Mat prev_normal_show=zc::nmap2rgb(prev_normals_word_coo);
            imshow("normals prev contour cue",prev_normal_show);

            double _min,_max;
            minMaxLoc(depth_prev_host,&_min,&_max);
            cout<<"min--------  "<<_min<<endl
                <<"max--------  "<<_max<<endl;
            depth_prev_host.convertTo(depth_prev_host,CV_8U,255.0/9000,0);
            imshow("previous depth",depth_prev_host);
            }
#endif
            {
            //ScopeTimeMicroSec time("|-sgf-computeCandidate"); //1ms
            //device::computeCandidate(nmaps_g_prev_[0],vmaps_g_prev_[0],position_camera_x,position_camera_y,position_camera_z,normal_mask,0.26);

            //zhangxaochen: ���� nmap_g_prev_choose_
            device::computeCandidate(nmap_g_prev_choose_,vmaps_g_prev_[0],position_camera_x,position_camera_y,position_camera_z,normal_mask,0.26);
            }
        }
        pcl::device::sync ();
      }

      //build kd tree for vertex on the normal mask

      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      vector<float> contour_candidate;
      vector<float> contour_candidate_normal;
      vector<float> contour_curr;
      if (global_time_!=0 && 
          !icp_orig_ && icp_sgf_cpu_)
      {
          {
          ScopeTimeMicroSec time("|-sgf-kdtree-total"); //130~180ms

          int c=640;
          //mask.download(contour,c);
          contMsk_.download(contour, c); //zhangxaochen: ���� contMsk_, ��������Ч���ٽ�, ����cont-Vmap ѡ��ҲҪ������ //2016-3-27 00:23:04
          normal_mask.download(candidate,c);
          nmaps_g_prev_[0].download(normals,c);
          vmaps_g_prev_[0].download(vertexes,c);
          vmaps_curr_[0].download(vertexes_curr,c);

          //���ݵõ������ݣ���contour candidateȡ��������������kd tree
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
              if(contour[i]==255 && !pcl_isnan(vertexes_curr[i])) //zhangxaochen: �����ж� vmap ��Ӧλ�� not-nan //2016-3-27 00:21:35
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
                  normal_map.at<Vec3f>(i,j)[0]=normals[i*640+j];
                  normal_map.at<Vec3f>(i,j)[1]=normals[(i+480)*640+j];
                  normal_map.at<Vec3f>(i,j)[2]=normals[(i+480*2)*640+j];
                  //cout<<a[i*640+j]<<" ";
              }
              //cout<<endl;
          }
          printf("sgf:Contour_map: %d\n", countNonZero(Contour_map));
          Contour_map.setTo(128, contMskHost != 0); //sgf ���������ʻҡ����غ� ��
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

          {
          //ScopeTimeMicroSec t("kdtree.setInputCloud"); //~15ms
          kdtree.setInputCloud (cloud);
          }

          }//ScopeTimeMicroSec time("sgf-kdtree-total");
      }
      

      //can't perform more on first frame
      if (global_time_ == 0
          && regObjId_ == 0) //zhangxaochen: ������ kinfu.orig ����Դ, ����0ֱ֡�ӷ���, �����Ե�0֡Ϊ��������ϵ
      {
        Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

        //sunguofei
        if (hint)
        {
            Matrix3frm init_Rcam_1 = hint->rotation().matrix();
            Vector3f init_tcam_1 = hint->translation().matrix();
            //���ݵ�һ֡ʱ������ͬ�ĳ�ʼR��t������������ϵ֮��任��dR��dt
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
        
        //sunguofei
        {
            //���ݵõ�����������ϵ�µ���һ֡��vmap�����������̬�µ����ͼ
            device::generateDepth(device_Rcam_inv,device_tcam,vmaps_g_prev_[0],depths_prev_[0]);
            //���ﰴ��contour cue���ĵķ������޲�һ�����ͼ
            zc::inpaintGpu(depths_prev_[0], depths_prev_[0]);
            pcl::device::sync();
        }
        for (int i = 1; i < LEVELS; ++i)
            device::pyrDown (depths_prev_[i-1], depths_prev_[i]);

        //zhangxaochen: ��ע�͵�����һ֡�Ͳ����� prev
        //Mat nmaps_g_prev_0_host = zc::nmap2rgb(nmaps_g_prev_[0]); //��Ե���򲻺�
        //imshow("nmaps_g_prev_0_host", nmaps_g_prev_0_host);
        //computeNormalsEigen(vmaps_g_prev_[0], nmap_g_prev_choose_);

        //��0֡��cc_norm_prev_way_ == 0 or 1 or 2, ������ֱ�Ӹ�ֵ��
        if(this->cc_norm_prev_way_ == 0 || this->cc_norm_prev_way_ == 1 || this->cc_norm_prev_way_ == 2)
            nmap_g_prev_choose_ = nmaps_g_prev_[0];

        depth_raw.copyTo(depth_raw_prev_); //����raw, ����һ֡ʹ��

        ++global_time_;
        return (false);
        //return (true); //zhangxaochen: f0 Ҳ����true, ʹ�� f0 �� showScene //ȫ��ɫ, Ϊʲô? ��ʱ����
      }
      else if (global_time_ == 0){ //if regObjId_ == 1,2,3
          CV_Assert(synModelCloudPtr_->size() > 0);

          Matrix3frm init_Rcam = rmats_[0];
          Vector3f   init_tcam = tvecs_[0];
          Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
          float3& device_tcam = device_cast<float3>(init_tcam);

          if(regObjId_ == 1 || regObjId_ == 3){ //��dmat��TSDFͶ������׼, ���0֡, ҲҪ��cube��, ����ֱ�ӷ���TSDF
              TsdfVolume::Ptr pTsdfVolume = regObjId_ == 1 ? tsdf_volume_shadow_ : tsdf_volume_; //��ʵ��0ֱ֡���� tsdf_volume_ ����, ����������;
              Affine3f prevPose = this->getCameraPose();
              //pRaycaster_->run(*pTsdfVolume, prevPose); //����������
              //pRaycaster_->getVertexMap();
              float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());

              //��ô���׼�� vmap, nmap
              raycast (intr, device_Rcam, device_tcam, pTsdfVolume->getTsdfTruncDist(), device_volume_size, pTsdfVolume->data(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
              for (int i = 1; i < LEVELS; ++i)
              {
                  resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
                  resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
              }
              pcl::device::sync ();
          }
//           else if(regObjId_ == 2){ //ֱ��Ͷ�� cloud2depth, ��TSDF���޹ء�
//               Affine3f pose = this->getCameraPose();
//               DeviceArray2D<unsigned short> depFromCloud_device;
//               zc::cloud2depth(*synModelCloudPtr_, pose, intr, cols_, rows_, depFromCloud_device); //��
// 
// #if 0   //cube-dmat �ϲ���˫���˲�
//               device::bilateralFilter (depFromCloud_device, depths_curr_[0]);
//               if (max_icp_distance_ > 0)
//                   device::truncateDepth(depths_curr_[0], max_icp_distance_);
// #elif 1
//               depths_curr_[0] = depFromCloud_device;
// #endif
//               for (int i = 1; i < LEVELS; ++i)
//                   device::pyrDown (depths_curr_[i-1], depths_curr_[i]);
// 
//               for (int i = 0; i < LEVELS; ++i)
//               {
//                   device::createVMap (intr(i), depths_curr_[i], vmaps_g_prev_[i]);
//                   //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
//                   computeNormalsEigen (vmaps_g_prev_[i], nmaps_g_prev_[i]);
//                   device::tranformMaps (vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
//               }
//           }//regObjId_ == 1,2,3

          //���� vmaps_g_prev_ �Ƿ�������ȷ�� //2016-4-20 19:42:11
          /*Mat vm_g_prev()*/

          ++global_time_; //Ҫ+1, �����: ��ʵ����[0]�ڱ����poses.csv ��λ���� [1]
          //return (false); //������, ����������, ��ʵ����Ҫ������ cube-vmap/nmap��׼
      }//if global_time_ == 0 && regObjId_ != 0
      
      //�����ǲ��� global_time_ == 0��
      if(regObjId_ == 2){ //ֱ��Ͷ�� cloud2depth, ��TSDF���޹ء�
          Affine3f pose = this->getCameraPose();
          Matrix3frm Rrm = pose.linear(); //�൱�� rmats_ & tvecs_.back()
          Vector3f t = pose.translation();
          Mat33&  device_Rcam = device_cast<Mat33> (Rrm);
          float3& device_tcam = device_cast<float3>(t);

          DeviceArray2D<unsigned short> depFromCloud_device;
          zc::cloud2depth(*synModelCloudPtr_, pose, intr, cols_, rows_, depFromCloud_device); //��

#if 0   //cube-dmat �ϲ���˫���˲�
          device::bilateralFilter (depFromCloud_device, depths_curr_[0]);
          if (max_icp_distance_ > 0)
              device::truncateDepth(depths_curr_[0], max_icp_distance_);
#elif 1
          //depths_curr_[0] = depFromCloud_device;
          depths_prev_[0] = depFromCloud_device;
#endif
          for (int i = 1; i < LEVELS; ++i)
              //device::pyrDown (depths_curr_[i-1], depths_curr_[i]);
              device::pyrDown (depths_prev_[i-1], depths_prev_[i]);

          for (int i = 0; i < LEVELS; ++i)
          {
              //device::createVMap (intr(i), depths_curr_[i], vmaps_g_prev_[i]);
              device::createVMap (intr(i), depths_prev_[i], vmaps_g_prev_[i]);
              //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
              computeNormalsEigen (vmaps_g_prev_[i], nmaps_g_prev_[i]); //��ʱ vmaps_g_prev_, nmaps_g_prev_ ������ global, ������ת��
              //�趨 tranformMaps �� src, dst ��ͬû���⣿ //�ƺ�û���� 2016-5-25 17:11:31
              device::tranformMaps (vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
          }
      }//if(regObjId_ == 2)


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
#if 0
        //sunguofei
        //�����ⲿ�豸�õ��Ħ�R�ͦ�t��������һ֡�ĳ�ֵ
        Matrix3frm dr=hint->rotation().matrix();
        Vector3f dt=hint->translation().matrix();
        Rcurr=d_R*dr*Rprev;
        tcurr=d_R*dt+tprev;
        Rcurr_back=Rcurr;
        tcurr_back=tcurr;
#endif
        if(this->imud_){
            Matrix3f dR = hint->rotation(); //��ʱ hint �� delta��, ci->c(i-1), �����ҳ�
            //Rcurr = dR * Rprev; //�� 2014-IMU, �ҳ�
            Rcurr = Rprev * dR;
            tcurr = tprev; //t������

            Rcurr_back = Rcurr;
            tcurr_back = tcurr;
        }
      }
      else
      {
        Rcurr = Rprev; // tranform to global coo for ith camera pose
        tcurr = tprev;
        //Rcurr_back = init_Rcam_;
        //tcurr_back = init_tcam_;
      }
      const double condThresh = 1.5e2; //��1e2��Ϊ 1.5e2 //2016-7-14 00:21:58
          //INVALID_COND = 1e8; //�������� 2016-7-14 00:22:54
      double condNum = 0;//INVALID_COND;

      //zhangxaochen: ����ICP����֮ǰ���۲졿����׼src, dst, ���� regObjId_ ֮�� dst ���� Id �仯���ı� //2016-4-21 11:20:17
//       MapArr &vmap_g_prev_0 = vmaps_g_prev_[0]; //lv0 ���ֱ��� 640*480
//       MapArr &vmap_curr_0 = vmaps_curr_[0];
      DepthMap dmatPrev_device, dmatCurr_device;
      device::generateDepth(device_Rprev_inv, device_tprev, vmaps_g_prev_[0], dmatPrev_device); //vmaps_g_prev_ ����ͶӰΪ���ͼ, ���ڹ۲�
      //device::generateDepth(device_Rprev_inv, device_tprev, vmaps_curr_[0], dmatCurr_device); //ʵ�ʶ��� Rprev, tprev //��, vmaps_curr_ ���������ϵ�µ�, ��Ӧ�����κ����ȥת�� dmat
      dmatCurr_device = depths_curr_[0];
      Mat dmatPrevHost(dmatPrev_device.rows(), dmatPrev_device.cols(), CV_16U),
          dmatCurrHost(dmatCurr_device.rows(), dmatCurr_device.cols(), CV_16U);
      dmatPrev_device.download(dmatPrevHost.data, dmatPrev_device.colsBytes());
      dmatCurr_device.download(dmatCurrHost.data, dmatCurr_device.colsBytes());
      Mat dmPh8u, dmCh8u;
      dmatPrevHost.convertTo(dmPh8u, CV_8UC1, 1.*UCHAR_MAX/1e4);

      //�ƺ� dmatCurr ����Դ������, ��ӡ�۲������Сֵ: ���ѽ����
      //double dmin, dmax;
      //minMaxLoc(dmatCurrHost, &dmin, &dmax);
      //cout<<"dmatCurrHost, dmin, dmax: "<<dmin<<", "<<dmax<<endl;
      
      dmatCurrHost.convertTo(dmCh8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
      //����׼����ͼƴ�ӽ����
      Mat prevAndCurr8u;
      cv::hconcat(dmPh8u, dmCh8u, prevAndCurr8u);
      cv::line(prevAndCurr8u, Point(640, 0), Point(640, 480), 255); //��ɫ�ָ�����
      imshow("prevAndCurr8u", prevAndCurr8u);
      static int pacCnt = 0; //prev-and-curr-counter
      char buf[256];
      sprintf (buf, "./prev&curr-%06d.png", pacCnt);
      imwrite(buf, prevAndCurr8u);
      pacCnt++;

      illCondMat_ = false;
      {
        //ScopeTime time("icp-all"); //600~900ms
        for (int level_index = LEVELS-1; level_index>=0; --level_index)
        //for (int level_index = 0; level_index>=0; --level_index) //���� M2 ֱ�� reset ԭ��    //2016-5-28 16:34:08
        {
            if(callCnt==131){ //������� @monitor-0714-1
                               //����: 131֡Ư�Ʋ��ǽ������Ĵ� 2016-7-14 18:00:07
                level_index=0; //��������ײ�
                cout<<"+++++++++++++++setting level_index=0"<<endl;
            }

          if(hint && illCondMat_)
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
          //for (int iter = 0; iter < 5; ++iter) //imud ʱ, ���Լ��ٵ�������, �۲��Ƿ�����, �������: @monitor-0715-0-kf-imud-iter3 //2016-7-17 17:22:01
          {
              if(callCnt==131){ //������� @monitor-0714-1 //2016-7-15 01:08:22
                  Eigen::Quaternionf q (Rcurr);
                  cout<<"+++++++++++++++131::t+Rcurr: "<< tcurr[0] << "," << tcurr[1] << "," << tcurr[2] << "," << q.w () << "," << q.x () << "," << q.y ()<< ","  << q.z () << std::endl;
              }
            Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3& device_tcurr = device_cast<float3>(tcurr);

            //---------------zhangxaochen: pre-GCOO_2_CAMCOO //2016-5-26 15:53:47
            Affine3f pose; //tmp-pose
            pose.linear() = Rcurr;
            pose.translation() = tcurr;
//             cout<<"level_index, iter: "<<level_index<<", "<<iter<<endl
//                 <<pose.matrix()<<endl;

//#ifdef GCOO_2_CAMCOO //������ʦҪ��, �� v/n_g_prev ͶӰ�� cam_coo, ������ԭ���� v/n_curr ͶӰ�� global_coo //g2c
#if 0
            //����Ŀ����: ��ѭ��ÿ�ε���������� vmap_g_prev, nmap_g_prev   //2016-5-25 20:52:58
            if(regObjId_ == 2){
                //---------------1. �� R, tcurr �õ���ǰ�ӵ���������organized�ӵ���, ע��, ���겻��, ����ȡ�Ӽ�
                DeviceArray2D<unsigned short> depFromCloud_device; //tmp-dmat-device, ԭ�ߴ�, ������Ҫ pyrDown
                zc::cloud2depth(*synModelCloudPtr_, pose, intr, cols_, rows_, depFromCloud_device); //������ͶӰ���������ϵ, �õ����ͼ
                depths_prev_[0] = depFromCloud_device; //�����д depths_prev_, �ڱ����޸�����? �ݲ���� ��δ�����
                for (int i = 1; i <= level_index; ++i) //i ֻ�� level_index
                    device::pyrDown (depths_prev_[i-1], depths_prev_[i]);
            }
            DepthMap dmat_prev_pyr = depths_prev_[level_index]; //��ǰpyr-level �� dmat_prev-device
            Mat dmatPrevHost = Mat::zeros(dmat_prev_pyr.rows(), dmat_prev_pyr.cols(), CV_16UC1); //ȫ���0
            dmat_prev_pyr.download(dmatPrevHost.data, dmat_prev_pyr.colsBytes());
            Mat dmPh8u;
            dmatPrevHost.convertTo(dmPh8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
            imshow("each-iter-dmPH8u", dmPh8u);
            waitKey(01);

            device::createVMap(intr(level_index), dmat_prev_pyr, vmap_g_prev); //vmap_g_prev �ݲ��� global, ������ת��
            computeNormalsEigen(vmap_g_prev, nmap_g_prev);
            device::tranformMaps(vmap_g_prev, nmap_g_prev, device_Rcurr, device_tcurr, vmap_g_prev, nmap_g_prev); //�˴�ӦȷʵΪ global ��
                //��ʵ���� cam->global, ֱ�� .cu �� search() ʹ�� v/nprev_cc ����, ��Ϊ�˱�����֮ǰ�����߼���������, ��ת��һ���� global //2016-5-26 17:10:21

            nmap_g_prev_choose_ = nmap_g_prev; //Ϊ���� app �е��Ի���
#endif //GCOO_2_CAMCOO

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
            if(icp_orig_){
                estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
                    vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());

                //zhangxaochen: ���Ե��� g2c ����, R/t_prev ���� curr, ��ȥ�� e-c.cu �� g2c ����, ��֤����Դ�� cpp ���� cu, 
                //������ reset, ˵�������� cu, ������ cpp  //2016-5-29 00:37:09
                //Matrix3frm Rcurr_inv = Rcurr.inverse ();
                //Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
                //estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rcurr_inv, device_tcurr, intr (level_index),
                //                                                                                //���𣺡�-----------��
                //    vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
            }
            else if(icp_sgf_cpu_){
                {
                //ScopeTimeMicroSec time("|-icp_sgf_cpu_"); //35ms (*19=665ms)

                //sunguofei---contour cue
                vector<float> cores_v_curr;
                vector<float> cores_v_prev;
                vector<float> cores_n_prev;
                //�Ե�ǰ���ͼ�е�contour�ϵĵ�Ѱ�Ҷ�Ӧ�㣬����˳�������
                for (int i=0;i<contour_curr.size()/3;++i)
                {
                    pcl::PointXYZ searchpoint;
                    //ÿ�������ڵ�ǰ�������ϵ�µģ�����R_curr��t_curr��������ת������������ϵ�£���Ϊ��һ֡ģ�͵õ��ĵ㶼������������ϵ�µģ�
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
                            //�ҵ�����Ҫ��Ķ�Ӧ�㣬�����Ǵ���vector�У��±���ͬ�ĵ���һ���Ӧ��
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
                //�ҵ���Ӧ��ϵ֮�󣬽��䴫���±ߺ����У�����߽��м���
                //���ַ�����
                //1��ÿһ�������������contour���candidate��
                //2��ֻ����������������contour���candidate�㣬Ŀǰ�Ľ��˼·�������ҵ��Ķ�Ӧ��ϵ�浽maparr�У������͵�ǰvmap nmap��ͬ��ֱ�Ӵ���estimateCombined��
                //   ��Ҫ��������µ�����vmap_candidate,nmap_candidate,vmap_contour

                //Ϊ���ܹ��Ѷ�Ӧ�㰴��Ҫ�󴫽�ȥ����������
                //1����������������xyzxyz...���ĳ�xxx..yyy..zzz..
                //2�����Ե�ǰmap��������ȥ��������Ĳ���

                
                vector<float> cores_n_prev_new,cores_v_prev_new,cores_v_curr_new;
#if 0 //sgf ȥβ���� xx_new ��ʽ
                int contour_size=cores_v_curr.size()/3/int(vmap_curr.cols())*int(vmap_curr.cols());
                if(level_index == 0 && iter == iter_num - 1)
                    printf("contour_size: %d, %d\n", contour_size, cores_v_curr.size()/3);

                for (int k1=0;k1<3;++k1)
                {
                    for (int k2=0;k2<contour_size;++k2)
                    {
                        cores_v_curr_new.push_back(cores_v_curr[k2*3+k1]);
                        cores_v_prev_new.push_back(cores_v_prev[k2*3+k1]);
                        cores_n_prev_new.push_back(cores_n_prev[k2*3+k1]);
                    }
                }
#else //zhangxaochen: ����ʵ������xx_new.size()=0, ��Ϊcorresp<640�����, ����, ��Ϊȫ�ӽ�ȥ, ĩβ��qnan //2016-3-27 16:51:44
                int contSz = cores_v_curr.size()/3;
                int qnanNum = vmap_curr.cols() - contSz % vmap_curr.cols();
                const float qnan = numeric_limits<float>::quiet_NaN();

                for(int k1=0;k1<3;++k1){
                    for(int k2=0;k2<contSz;++k2){
                        cores_v_curr_new.push_back(cores_v_curr[k2*3+k1]);
                        cores_v_prev_new.push_back(cores_v_prev[k2*3+k1]);
                        cores_n_prev_new.push_back(cores_n_prev[k2*3+k1]);
                    }
                    //β�����qnan��
                    for(int k2=0;k2<qnanNum;++k2){
                        cores_v_curr_new.push_back(qnan);
                        cores_v_prev_new.push_back(qnan);
                        cores_n_prev_new.push_back(qnan);
                    }
                }
                if(level_index == 0 && iter == iter_num - 1)
                    cout<<"cores_v_curr_new.size()/3, cores_v_curr.size()/3: "<<cores_v_curr_new.size()/3<<", "<<cores_v_curr.size()/3<<endl;
#endif //���� xx_new ��ʽ

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
                        vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data (),this->contWeight_);
                }

                }//ScopeTimeMicroSec time("|-icp_sgf_cpu_"); //35ms (*19=665ms)
            }//else if(icp_sgf_cpu_)
            else if(icp_cc_inc_weight){
                zc::computeContours(depths_curr_[level_index], contMsk_);
                estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
                    vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data (), contMsk_, this->contWeight_);
            }

    #endif
            //checking nullspace
            double det = A.determinant ();

            //sunguofei
            Eigen::Matrix<double,6,6,Eigen::RowMajor>::EigenvaluesReturnType eigenvalues = A.eigenvalues();
            condNum=eigenvalues(0,0).real()/eigenvalues(5,0).real();
            //cout << "eigenvalues: " << eigenvalues << endl;
            condNum=sqrt(condNum);
            //cout<<"condition of A: "<<cond<<endl;
//             if (cond>100)
//             {
//                 Rcurr=Rcurr_back;tcurr=tcurr_back;
//                 break;
//             }

            if(callCnt % 10 == 0 || illCondMat_ //�� illCondMat_, ������ͬһ level_index �й�һ�鲻ͬ iter
                || 128<callCnt && callCnt<134 //������� @monitor-0714-1
                || callCnt<100 //��Ե�������, ���ƽ�����֡ 2016-7-17 21:25:24
                )
                cout<<"condNum: "<<condNum<<"; eigenvalues: "<<eigenvalues.transpose()<<"; det(A): "<<det<<endl;

            //if(hint && condNum > condThresh){
            if(hint){
                static int callCntFlag = callCnt; //��ǰһ�ֵ����ڵı��, �� condNum > condThresh ����, �Ա�һ�ֵ����ڳ������������Ϣ //2016-7-14 00:30:17
                if(callCntFlag == callCnt)
                    cout<<"condNum: "<<condNum<<"; eigenvalues: "<<eigenvalues.transpose()<<"; det(A): "<<det<<endl;

                if(condNum > condThresh){
                    if(callCntFlag != callCnt){
                        cout<<"illCondMat_ = true; level_index, iter:= "<<level_index<<", "<<iter<<endl
                            <<"condNum: "<<condNum<<"; eigenvalues: "<<eigenvalues.transpose()<<"; det(A): "<<det<<endl;
                    }

                    callCntFlag = callCnt;
                    //condNum = INVALID_COND;
                    illCondMat_ = true;
                    //break; //��Ȼ�����ߣ���һ��
                }
                else
                    illCondMat_ = false;
            }

            if (fabs (det) < 1e-15 || pcl_isnan (det))
            //if (fabs (det) < 1e-15 || pcl_isnan (det) || (hint && cond>INVALID_COND) )
            {
              if (pcl_isnan (det)) cout << "qnan" << endl;

               reset ();
               return (false);

              //sunguofei
              //���������ã����Ǿ����ⲿ�豸�õ���RT��Ϊ������
              //Rcurr=Rcurr_back;tcurr=tcurr_back;
              //cond=INVALID_COND;
              //illCondMat = true;
              //if(hint)
              //    break; //��Ȼ�����ߣ���һ��
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

//             cout<<"Rinc, tinc:"<<endl
//                 <<Rinc<<endl
//                 <<tinc<<endl;

            //compose
            tcurr = Rinc * tcurr + tinc;
            Rcurr = Rinc * Rcurr;
          }//for-iter
        }//for-level_index
      }

      //zhangxaochen: ����: csvDeltaRcurr_ ��������ȷ��
      if(70<callCnt && callCnt<80 || 335<callCnt && callCnt<345){ //������� @monitor-0708-0
          cout<<"R-prev & hint_init_back & curr:\n"
              <<Rprev<<endl<<endl
              <<Rcurr_back<<endl<<endl
              <<Rcurr<<endl<<endl;
      }
      if(128<callCnt && callCnt<134){ //������� @monitor-0714-1
          Eigen::Quaternionf q (Rcurr);
          cout<<"+++++++++++++++t+Rcurr: "<< tcurr[0] << "," << tcurr[1] << "," << tcurr[2] << "," << q.w () << "," << q.x () << "," << q.y ()<< ","  << q.z () << std::endl;
      }

      //save tranform
      //if (cond>1e5)
      if(hint && imud_ && illCondMat_)
      {
          rmats_.push_back (Rcurr_back);
          //tvecs_.push_back (tcurr_back);
          tvecs_.push_back (tcurr); //t���� ICP �Ľ��, ���� hint=true �� illCondMat_
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
#if 0   //kinfu ԭ����, �о�ʵ��������, ���? ��ʱ�滻�� //2016-5-9 01:12:07

      Matrix3frm Rcurr = rmats_[global_time_ - 1];
      Vector3f   tcurr = tvecs_[global_time_ - 1];
#elif 1 //�� hint, �����в���ͬʱ�� -csv_rt_hint -icp 0 ʱ��, ���ô˴�:
      Matrix3frm Rcurr = hint->rotation();
      Vector3f tcurr = hint->translation();
#endif

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

  if(illCondMat_){ //zhangxaochen //2016-7-12 17:08:11
      cout<<"illCondMat_=true, not integrating+++++++++++++++"<<endl;
      integrate = false;
  }

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
    raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmap_g_model_, nmap_g_model_);

    if(regObjId_ == 0 || regObjId_ == 3){ //=0, 3 ����ԭ tsdf_volume_ ��Ͷ��
        //raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
        vmaps_g_prev_[0] = vmap_g_model_;
        nmaps_g_prev_[0] = nmap_g_model_;
    }
    else if(regObjId_ == 1) //=1ʱ, ��Ӱ�� TSDF ��Ͷ��
        raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_shadow_->getTsdfTruncDist(), device_volume_size, tsdf_volume_shadow_->data(), vmaps_g_prev_[0], nmaps_g_prev_[0]);
    else if(regObjId_ == 2) //TODO, ��δʵ�� 2016-4-21 10:22:52
        ;
    for (int i = 1; i < LEVELS; ++i)
    {
      resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
      resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
    }
    pcl::device::sync ();
  }

  //zhangxaochen: ���Ի��� nmaps_curr_, nmaps_g_prev_, ���߿����д�
  //DeviceArray2D<NormalType> nmap2d; //�ȼ��� float8. 
  //getLastFrameNormals(nmap2d); //opencv���Ի��ƣ�Ҫ�� pcl::visualizer. �ݷ���

  //Mat nmaps_g_prev_0_host = zc::nmap2rgb(nmaps_g_prev_[0]); //��Ե���򲻺�
  //imshow("nmaps_g_prev_0_host", nmaps_g_prev_0_host);

  if(this->cc_norm_prev_way_ == 0){ //kinfu-orig.(raycast)
      nmap_g_prev_choose_ = nmaps_g_prev_[0];
  }
  else if(this->cc_norm_prev_way_ == 1){
      //���¼��� nmap_g ���̣�
      //volume->[raycast]->genDepth->[zc::inpaint]->inpaintDmat->[createVmap]->vmap_cam_coo->[transformVmap]->vmap_g->[computeNormalsEigen]->nmap_g
      Affine3f prevPose = this->getCameraPose();
      pRaycaster_->run(this->volume(), prevPose); //ǰ��ICP�Ѿ��õ� curr_R,t, �����ǵ� i ֡��̬, ���Ƕ�����һ֡��˵���� (i-1) //2016-5-12 16:45:24
#if 0   //�� pcc ���� p4/7 ���Ͻ�˵��, �õ���: inpainted depth image ^D'_(i-1), ̫����, �� inp ������?
      DepthMap genDepthPrev, genDepthPrevInp;
      pRaycaster_->generateDepthImage(genDepthPrev); //i-1 ����ӽ����ͼ
      //Mat genDepthPrevHost(genDepthPrev.rows(), genDepthPrev.cols(), CV_16UC1);
      //genDepthPrev.download(genDepthPrevHost.data, genDepthPrev.colsBytes());
      //Mat genDepthPrevHost8u;
      //genDepthPrevHost.convertTo(genDepthPrevHost8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
      //imshow("genDepthPrevHost8u", genDepthPrevHost8u);

      zc::inpaintGpu(genDepthPrev, genDepthPrevInp); //i-1 ���ͼ�޲� inpaint, �����: http://i.stack.imgur.com/vtZ3t.jpg
      //Mat genDepthPrevInpHost(genDepthPrevInp.rows(), genDepthPrevInp.cols(), CV_16UC1);
      //genDepthPrevInp.download(genDepthPrevInpHost.data, genDepthPrevInp.colsBytes()); //.step()=1536 �� .colsBytes()=1280
      //Mat genDepthPrevInpHost8u;
      //genDepthPrevInpHost.convertTo(genDepthPrevInpHost8u, CV_8UC1, 1*UCHAR_MAX/1e4);
      //imshow("genDepthPrevInpHost8u", genDepthPrevInpHost8u);

      MapArr vmap_prev_inp; //inpainted
      device::createVMap(intr(0), genDepthPrevInp, vmap_prev_inp); //i-1 �������ϵ����
      //���� inp-vmap �Ƿ���ȷ, �����д�
      //zc::test::testVmap(vmap_prev_inp, "vmap_prev_inp"); //OK, http://i.stack.imgur.com/HQodq.png

      Matrix3frm Rrm = prevPose.linear(); //Ҫ rm(row-major)�𣿱��룡���Ծ��飺 R(cm) �ȼ��� R(rm).inverse()
      Matrix3f Rcm = prevPose.linear();
      //assert(Rrm == Rcm); //���߼���ȷʵ���
      //assert(Rrm.isApprox(Rcm, 1e-8)); //ͬ�ϡ�

      Matrix3f Rinv = Rrm.inverse();      //
      Vector3f t = prevPose.translation();

      const Mat33 &device_Rrm = device_cast<const Mat33>(Rrm);
      const Mat33 &device_Rinv = device_cast<const Mat33>(Rinv); //�������� (Matrix3f R) ���� (Matrix3frm R), �������� Rinv ����ȷ����
      const float3 &device_t = device_cast<const float3>(t);

      MapArr vmap_g_prev_inp;
      zc::transformVmap(vmap_prev_inp, device_Rrm, device_t, vmap_g_prev_inp);
      //zc::test::testVmap(vmap_g_prev_inp, "vmap_g_prev_inp"); //OK, ��nmap_g_prev_eigen_

      MapArr nmap_g_prev_inp;
      //computeNormalsEigen(vmap_g_prev_inp, nmap_g_prev_choose_); //������ǰ�� nmap_g_prev_choose_ = nmaps_g_prev_[0]; ������д���ڴ�
      computeNormalsEigen(vmap_g_prev_inp, nmap_g_prev_inp);
      nmap_g_prev_choose_ = nmap_g_prev_inp;
#elif 01
      nmap_g_prev_choose_ = pRaycaster_->getNormalMap();
#endif

#if 0     //---------------��������̫�ԣ� ��˼·�� vmap_cam ->nmap_cam ->nmap_g //���ͬ�ϣ���������������Ե��³��*һ��*
      MapArr nmap_prev_inp;
      computeNormalsEigen(vmap_prev_inp, nmap_prev_inp);

      MapArr nmap_g_prev_inp2;

      float3 origTvec;
      origTvec.x = origTvec.y = origTvec.z = 0;
      zc::transformVmap(nmap_prev_inp, device_R, origTvec, nmap_g_prev_inp2);
      imshow("nmap_g_prev_inp2", zc::nmap2rgb(nmap_g_prev_inp2));
#endif
  }
  else if(this->cc_norm_prev_way_ == 2){
      //TODO: contour-cue ���� 2.3 Normal ���㷽�� @sgf
      Affine3f prevPose = this->getCameraPose();
      pRaycaster_->run(this->volume(), prevPose); //��ʱ��Ϊ i-1 �����̬, ��Ϊ i ֡��̬��û�����
      DepthMap genDepthPrev, genDepthPrevInp;
      pRaycaster_->generateDepthImage(genDepthPrev); //i-1 ����ӽ����ͼ
      zc::inpaintGpu(genDepthPrev, genDepthPrevInp); //i-1 ���ͼ�޲� inpaint
      Matrix3frm Rrm = prevPose.linear();
      const Mat33 &device_Rrm = device_cast<const Mat33>(Rrm);

      Mat genDprevInpHost=Mat::zeros(genDepthPrevInp.rows(),genDepthPrevInp.cols(),CV_16U);
      genDepthPrevInp.download(genDprevInpHost.data,genDprevInpHost.cols*genDprevInpHost.elemSize());

      Mat genDprevHost(genDepthPrev.rows(), genDepthPrev.cols(), CV_16U);
      genDepthPrev.download(genDprevHost.data, genDepthPrev.colsBytes());
      Mat genDprev8u, genDprevInp8u;
      genDprevHost.convertTo(genDprev8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
      genDprevInpHost.convertTo(genDprevInp8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
      imshow("genDprevInp8u", genDprevInp8u);

//       Mat kx,ky;
//       getDerivKernels(kx,ky,1,0,7);
//       cout<<"kx, ky:\n"
//           <<kx<<endl
//           <<ky<<endl;
      Mat grandient_x,grandient_y;
      Sobel(genDprevInpHost,grandient_x,CV_32F,1,0,7,1.0/1280);
      Sobel(genDprevInpHost,grandient_y,CV_32F,0,1,7,1.0/1280);
      //cout<<grandient_y<<endl;
      MapArr grandient_x_device,grandient_y_device;
      grandient_x_device.upload(grandient_x.data,grandient_x.cols*grandient_x.elemSize(),grandient_x.rows,grandient_x.cols);
      grandient_y_device.upload(grandient_y.data,grandient_y.cols*grandient_y.elemSize(),grandient_y.rows,grandient_y.cols);
      device::computeNormalsContourcue(intr(0),depths_prev_[0],grandient_x_device,grandient_y_device,prev_normals);
      vector<float> normals_curr_coo;
      int cols_normal;
      prev_normals.download(normals_curr_coo,cols_normal);
      //��ǰ�������ϵ�µķ���ת����������ϵ�µķ���
      float3 t_tmp;
      t_tmp.x=t_tmp.y=t_tmp.z=0;
      //const Mat33 &R_prev = device_cast<const Mat33>(rmats_[global_time_-1]);
      MapArr prev_normals_word_coo;
      zc::transformVmap(prev_normals,device_Rrm,t_tmp,prev_normals_word_coo);
      Mat prev_normal_show=zc::nmap2rgb(prev_normals_word_coo);
      imshow("prev_normal_show", prev_normal_show);

      nmap_g_prev_choose_ = prev_normals_word_coo;
  }

  //sunguofei
  {
      //���ݵõ�����������ϵ�µ���һ֡��vmap�����������̬�µ����ͼ
      //device::generateDepth(device_Rcurr_inv,device_tcurr,vmaps_g_prev_[0],depths_prev_[0]);
      device::generateDepth(device_Rcurr_inv,device_tcurr,vmap_g_model_,depths_prev_[0]); //model
      //���ﰴ��contour cue���ĵķ������޲�һ�����ͼ
      zc::inpaintGpu(depths_prev_[0], depths_prev_[0]);
      pcl::device::sync();
  }
  for (int i = 1; i < LEVELS; ++i)
      device::pyrDown (depths_prev_[i-1], depths_prev_[i]);

  Mat dcurrRawHost = Mat::zeros(depth_raw.rows(), depth_raw.cols(), CV_16UC1);
  depth_raw.download(dcurrRawHost.data, depth_raw.colsBytes());

  Mat gradu, gradv;
  Sobel(dcurrRawHost,gradu,CV_32F,1,0,7,1.0/1280);
  Sobel(dcurrRawHost,gradv,CV_32F,0,1,7,1.0/1280);

  MapArr gradu_device,gradv_device;
  gradu_device.upload(gradu.data, gradu.cols * gradu.elemSize(), gradu.rows, gradu.cols);
  gradv_device.upload(gradv.data, gradv.cols * gradv.elemSize(), gradv.rows, gradv.cols);

  //zc::align2dmapsGPU(depth_raw_prev_, intr, depth_raw, gradu_device, gradv_device);
  
  Mat dprevRawHost = Mat::zeros(depth_raw_prev_.rows(), depth_raw_prev_.cols(), CV_16UC1);
  depth_raw_prev_.download(dprevRawHost.data, depth_raw_prev_.colsBytes());
  //zc::align2dmapsCPU(dprevRawHost, intr, dcurrRawHost, gradu, gradv);

  depth_raw.copyTo(depth_raw_prev_); //����raw, ����һ֡ʹ��
  
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
  //generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
  //zhangxaochen: ���� regObjId_ ֮��, vmaps_g_prev_ ��������ı�, �ʴ˴���Ϊÿ������Ͷ�� TSDF
  //raycast (intr, device_Rcurr, device_tcurr, tsdf_volume_->getTsdfTruncDist(), device_volume_size, tsdf_volume_->data(), vmaps_g_prev_[0], nmaps_g_prev_[0]); //ȱ����Ҫintr, pose ����, ����
  generateImage (vmap_g_model_, nmap_g_model_, light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  //device::convert (vmaps_g_prev_[0], c);

  //zhangxaochen: vmaps_g_prev_ ��ģ��ͶӰvmap, ��Ϊ����ʾ���磨not���������ϵ��ǰ֡ curr
  Affine3f pose = this->getCameraPose();
  Matrix3frm Rrm = pose.linear();
  Vector3f t = pose.translation();
  const Mat33 &device_Rrm = device_cast<const Mat33>(Rrm);
  const float3 &device_t = device_cast<const float3>(t);

  MapArr vmap_g_curr;
  zc::transformVmap(vmaps_curr_[0], device_Rrm, device_t, vmap_g_curr);
  device::convert(vmap_g_curr, c);

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
