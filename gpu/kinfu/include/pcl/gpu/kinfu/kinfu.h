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

#ifndef PCL_KINFU_KINFUTRACKER_HPP_
#define PCL_KINFU_KINFUTRACKER_HPP_

#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/kinfu/pixel_rgb.h>
#include <pcl/gpu/kinfu/tsdf_volume.h>
#include <pcl/gpu/kinfu/color_volume.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <vector>

//zhangxaochen:
#include "contour_cue_impl.h"

//sunguofei---contour cue
#include <pcl/kdtree/kdtree_flann.h>

// Focal lengths of RGB camera
#define KINFU_DEFAULT_RGB_FOCAL_X 525.f
#define KINFU_DEFAULT_RGB_FOCAL_Y 525.f

// Focal lengths of depth (i.e. NIR) camera
#define KINFU_DEFAULT_DEPTH_FOCAL_X 585.f
#define KINFU_DEFAULT_DEPTH_FOCAL_Y 585.f

namespace pcl
{
  namespace gpu
  {
    /** \brief KinfuTracker class encapsulates implementation of Microsoft Kinect Fusion algorithm
      * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
      */
    class PCL_EXPORTS KinfuTracker
    {
      public:
        /** \brief Pixel type for rendered image. */
        typedef pcl::gpu::PixelRGB PixelRGB;

        typedef DeviceArray2D<PixelRGB> View;
        typedef DeviceArray2D<unsigned short> DepthMap;

        typedef pcl::PointXYZ PointType;
        typedef pcl::Normal NormalType;

        /** \brief Constructor
          * \param[in] rows height of depth image
          * \param[in] cols width of depth image
          */
        KinfuTracker (int rows = 480, int cols = 640);

        /** \brief Sets Depth camera intrinsics
          * \param[in] fx focal length x 
          * \param[in] fy focal length y
          * \param[in] cx principal point x
          * \param[in] cy principal point y
          */
        void
        setDepthIntrinsics (float fx, float fy, float cx = -1, float cy = -1);
        
        /** \brief Get Depth camera intrinsics
          * \param[out] fx focal length x 
          * \param[out] fy focal length y
          * \param[out] cx principal point x
          * \param[out] cy principal point y
          */
        void
        getDepthIntrinsics (float& fx, float& fy, float& cx, float& cy);
        

        /** \brief Sets initial camera pose relative to volume coordiante space
          * \param[in] pose Initial camera pose
          */
        void
        setInitalCameraPose (const Eigen::Affine3f& pose);
                        
		/** \brief Sets truncation threshold for depth image for ICP step only! This helps 
		  *  to filter measurements that are outside tsdf volume. Pass zero to disable the truncation.
          * \param[in] max_icp_distance Maximal distance, higher values are reset to zero (means no measurement). 
          */
        void
        setDepthTruncationForICP (float max_icp_distance = 0.f);

        /** \brief Sets ICP filtering parameters.
          * \param[in] distThreshold distance.
          * \param[in] sineOfAngle sine of angle between normals.
          */
        void
        setIcpCorespFilteringParams (float distThreshold, float sineOfAngle);
        
        /** \brief Sets integration threshold. TSDF volume is integrated iff a camera movement metric exceedes the threshold value. 
          * The metric represents the following: M = (rodrigues(Rotation).norm() + alpha*translation.norm())/2, where alpha = 1.f (hardcoded constant)
          * \param[in] threshold a value to compare with the metric. Suitable values are ~0.001          
          */
        void
        setCameraMovementThreshold(float threshold = 0.001f);

        /** \brief Performs initialization for color integration. Must be called before calling color integration. 
          * \param[in] max_weight max weighe for color integration. -1 means default weight.
          */
        void
        initColorIntegration(int max_weight = -1);

        /** \brief Returns cols passed to ctor */
        int
        cols ();

        /** \brief Returns rows passed to ctor */
        int
        rows ();

        /** \brief Processes next frame.
          * \param[in] depth next frame with values in millimeters
          * \param hint
          * \return true if can render 3D view.
          */
        bool operator() (const DepthMap& depth, Eigen::Affine3f* hint=NULL);

        /** \brief Processes next frame (both depth and color integration). Please call initColorIntegration before invpoking this.
          * \param[in] depth next depth frame with values in millimeters
          * \param[in] colors next RGB frame
          * \return true if can render 3D view.
          */
        bool operator() (const DepthMap& depth, const View& colors);

        /** \brief Returns camera pose at given time, default the last pose
          * \param[in] time Index of frame for which camera pose is returned.
          * \return camera pose
          */
        Eigen::Affine3f
        getCameraPose (int time = -1) const;

        /** \brief Returns number of poses including initial */
        size_t
        getNumberOfPoses () const;

        /** \brief Returns TSDF volume storage */
        const TsdfVolume& volume() const;

        /** \brief Returns TSDF volume storage */
        TsdfVolume& volume();

        /** \brief Returns color volume storage */
        const ColorVolume& colorVolume() const;

        /** \brief Returns color volume storage */
        ColorVolume& colorVolume();
        
        /** \brief Renders 3D scene to display to human
          * \param[out] view output array with image
          */
        void
        getImage (View& view) const;
        
        /** \brief Returns point cloud abserved from last camera pose
          * \param[out] cloud output array for points
          */
        void
        getLastFrameCloud (DeviceArray2D<PointType>& cloud) const;

        /** \brief Returns point cloud abserved from last camera pose
          * \param[out] normals output array for normals
          */
        void
        getLastFrameNormals (DeviceArray2D<NormalType>& normals) const;

        /** \brief Disables ICP forever */
        void disableIcp();

        //zhangxaochen:
        bool imud_;// = false; //2016-7-12 16:15:39
        bool illCondMat_;

        bool icp_orig_;// = true;          //default， kinfu 原来的实现
        bool icp_sgf_cpu_;// = false;      //孙国飞 cpu 上 kdTreeFlann 查找 contour-cue 对应点实现
        
        bool icp_cc_inc_weight;// = false; //仅增加 curr-depth 上找到的 contour-cue 的权重，仍用原 kinfu 对应点 search 算法
        float contWeight_;// = 1;

        //@contour-cue impl.: choose a computing normal method when getting CCCandidates
        //0(default): kinfu-orig.(raycast)
        //1: volume->[raycast]->genDepth->[zc::inpaint]->inpaintDmat->[createVmap]->vmap_cam_coo->[transformVmap]->vmap_g->[computeNormalsEigen]->nmap_g
        //2: contour-cue impl.
        int cc_norm_prev_way_; //= 0;

        RayCaster::Ptr pRaycaster_;

        //---------------虚拟立方体 2016-4-7 22:02:25
        PointCloud<PointXYZ>::Ptr synModelCloudPtr_;
        //棱边下标： //2016-4-10 17:55:56
        vector<int> edgeIdxVec_;
        //按下标加载的棱边点集, 以及其消隐后的子集
        PointCloud<PointXYZ>::Ptr edgeCloud_, edgeCloudVisible_;

        //+++++++++++++++配准目标物指定, 预计可以是:
        //0. kinfu.orig 原流程, 原TSDF投射为vmap, nmap;
        //1. 虚拟立方体cube的伴随【影子】TSDF投射得到的深度图; 优: kinfu::raycast 直接得到 vmap, nmap; 劣: 相当于压缩, edge会损失
        //2. 我自己实现的cube-cloud 直接投射 cloud2depth, 与TSDF【无关】; 优: 可保留edge; 劣: 相当于自己实现的消隐、压缩算法, 精度未对比测量;
        //3. cube预置到老TSDF, 与depth共同形成模型, 再投射为深度图; 优: 虚实公摊权重, 似乎更合理; 劣: 并不单独考虑cube-edge
        int regObjId_;
        TsdfVolume::Ptr tsdf_volume_shadow_; //当 regObjId_==1时, 才初始化

        zc::MaskMap getContMask(){
            return contMsk_;
        }

        zc::MaskMap getContCorrespMask(){
            return contCorrespMsk_;
        }

        //zhangxaochen:
        DeviceArray2D<float> getNmapGprev(){
            return nmap_g_prev_choose_;
        }

      private:
        
        /** \brief Number of pyramid levels */
        enum { LEVELS = 3 };

        /** \brief ICP Correspondences  map type */
        typedef DeviceArray2D<int> CorespMap;

        /** \brief Vertex or Normal Map type */
        typedef DeviceArray2D<float> MapArr;
        
        typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
        typedef Eigen::Vector3f Vector3f;

        /** \brief Height of input depth image. */
        int rows_;
        /** \brief Width of input depth image. */
        int cols_;
        /** \brief Frame counter */
        int global_time_;

        /** \brief Truncation threshold for depth image for ICP step */
        float max_icp_distance_;

        /** \brief Intrinsic parameters of depth camera. */
        float fx_, fy_, cx_, cy_;

        /** \brief Tsdf volume container. */
        TsdfVolume::Ptr tsdf_volume_;
        ColorVolume::Ptr color_volume_;
                
        /** \brief Initial camera rotation in volume coo space. */
        Matrix3frm init_Rcam_;

        /** \brief Initial camera position in volume coo space. */
        Vector3f   init_tcam_;

        /** \brief array with IPC iteration numbers for each pyramid level */
        int icp_iterations_[LEVELS];
        /** \brief distance threshold in correspondences filtering */
        float  distThres_;
        /** \brief angle threshold in correspondences filtering. Represents max sine of angle between normals. */
        float angleThres_;
        
        /** \brief Depth pyramid. */
        std::vector<DepthMap> depths_curr_;
        /** \brief Vertex maps pyramid for current frame in global coordinate space. */
        std::vector<MapArr> vmaps_g_curr_;
        /** \brief Normal maps pyramid for current frame in global coordinate space. */
        std::vector<MapArr> nmaps_g_curr_;

        //sunguofei
        /** \brief Normal map of previous frame computed from another method in contour cue. */
        MapArr prev_normals;

        /** \brief Vertex maps pyramid for previous frame in global coordinate space. */
        std::vector<MapArr> vmaps_g_prev_;
        /** \brief Normal maps pyramid for previous frame in global coordinate space. */
        std::vector<MapArr> nmaps_g_prev_;

        //sunguofei
        /** \brief Normal maps pyramid for previous frame in global coordinate space. Another implement */
        std::vector<MapArr> nmaps_g_prev_contourcue;
        /** \brief previous Depth pyramid. */
        std::vector<DepthMap> depths_prev_;        

        /** \brief Vertex maps pyramid for current frame in current coordinate space. */
        std::vector<MapArr> vmaps_curr_;
        /** \brief Normal maps pyramid for current frame in current coordinate space. */
        std::vector<MapArr> nmaps_curr_;

        /** \brief Array of buffers with ICP correspondences for each pyramid level. */
        std::vector<CorespMap> coresps_;
        
        /** \brief Buffer for storing scaled depth image */
        DeviceArray2D<float> depthRawScaled_;
        
        /** \brief Temporary buffer for ICP */
        DeviceArray2D<double> gbuf_;
        /** \brief Buffer to store MLS matrix. */
        DeviceArray<double> sumbuf_;

        /** \brief Array of camera rotation matrices for each moment of time. */
        std::vector<Matrix3frm> rmats_;
        
        /** \brief Array of camera translations for each moment of time. */
        std::vector<Vector3f> tvecs_;

        /** \brief Camera movement threshold. TSDF is integrated iff a camera movement metric exceedes some value. */
        float integration_metric_threshold_;

        /** \brief ICP step is completelly disabled. Inly integratio now */
        bool disable_icp_;
        
        /** \brief Allocates all GPU internal buffers.
          * \param[in] rows_arg
          * \param[in] cols_arg          
          */
        void
        allocateBufffers (int rows_arg, int cols_arg);

        /** \brief Performs the tracker reset to initial  state. It's used if case of camera tracking fail.
          */
        void
        reset ();

        //sunguofei
        Eigen::Matrix3f d_R;
        Eigen::Vector3f d_t;

        //zhangxaochen:
        zc::MaskMap contMsk_;
        zc::MaskMap contCorrespMsk_; //contour correspondence candidates
        MapArr nmap_g_prev_choose_; //现在觉得乱, 不如用 nmaps_g_prev_[0], 但是懒得改了 //2016-5-12 16:58:19
        
        //vmaps_g_prev_, nmaps_g_prev_ 改作随 regObjId_ 变化, 故用新变量作为投射TSDF做 V(i-1) N(i-1): //2016-4-21 16:24:42
        MapArr vmap_g_model_, nmap_g_model_;
        DepthMap dmatInp_;

        DepthMap depth_raw_prev_; //前一帧原始深度图    //2016-6-21 21:50:32

public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };
  }
};

#endif /* PCL_KINFU_KINFUTRACKER_HPP_ */
