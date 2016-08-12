/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
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
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */


#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <vector>

#include <pcl/console/parse.h>

#include <boost/filesystem.hpp>

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/exceptions.h>

#include "openni_capture.h"
#include <pcl/visualization/point_cloud_color_handlers.h>
#include "evaluation.h"

#include <pcl/common/angles.h>

#include "tsdf_volume.h"
#include "tsdf_volume.hpp"

#include "camera_pose.h"

#ifdef HAVE_OPENCV  
  #include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
//zhangxaochen:
//using namespace cv;
#include "contour_cue_impl.h"
#include "syn_model_impl.h" //2016-4-5 04:20:53
#include "zcUtility.h"
#include "test_cuda.h"
#include "test_ns.h"

//#include "video_recorder.h"
#endif
typedef pcl::ScopeTime ScopeTimeT;

#include "../src/internal.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace pcl
{
  namespace gpu
  {
    void paint3DView (const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
    void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
  }

  namespace visualization
  {
    //////////////////////////////////////////////////////////////////////////////////////
    /** \brief RGB handler class for colors. Uses the data present in the "rgb" or "rgba"
      * fields from an additional cloud as the color at each point.
      * \author Anatoly Baksheev
      * \ingroup visualization
      */
    template <typename PointT>
    class PointCloudColorHandlerRGBCloud : public PointCloudColorHandler<PointT>
    {
      using PointCloudColorHandler<PointT>::capable_;
      using PointCloudColorHandler<PointT>::cloud_;

      typedef typename PointCloudColorHandler<PointT>::PointCloud::ConstPtr PointCloudConstPtr;
      typedef typename pcl::PointCloud<RGB>::ConstPtr RgbCloudConstPtr;

      public:
        typedef boost::shared_ptr<PointCloudColorHandlerRGBCloud<PointT> > Ptr;
        typedef boost::shared_ptr<const PointCloudColorHandlerRGBCloud<PointT> > ConstPtr;
        
        /** \brief Constructor. */
        PointCloudColorHandlerRGBCloud (const PointCloudConstPtr& cloud, const RgbCloudConstPtr& colors)
          : rgb_ (colors)
        {
          cloud_  = cloud;
          capable_ = true;
        }
              
        /** \brief Obtain the actual color for the input dataset as vtk scalars.
          * \param[out] scalars the output scalars containing the color for the dataset
          * \return true if the operation was successful (the handler is capable and 
          * the input cloud was given as a valid pointer), false otherwise
          */
        virtual bool
        getColor (vtkSmartPointer<vtkDataArray> &scalars) const
        {
          if (!capable_ || !cloud_)
            return (false);
         
          if (!scalars)
            scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
          scalars->SetNumberOfComponents (3);
            
          vtkIdType nr_points = vtkIdType (cloud_->points.size ());
          reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples (nr_points);
          unsigned char* colors = reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->GetPointer (0);
            
          // Color every point
          if (nr_points != int (rgb_->points.size ()))
            std::fill (colors, colors + nr_points * 3, static_cast<unsigned char> (0xFF));
          else
            for (vtkIdType cp = 0; cp < nr_points; ++cp)
            {
              int idx = cp * 3;
              colors[idx + 0] = rgb_->points[cp].r;
              colors[idx + 1] = rgb_->points[cp].g;
              colors[idx + 2] = rgb_->points[cp].b;
            }
          return (true);
        }

      private:
        virtual std::string 
        getFieldName () const { return ("additional rgb"); }
        virtual std::string 
        getName () const { return ("PointCloudColorHandlerRGBCloud"); }
        
        RgbCloudConstPtr rgb_;
    };
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
vector<string> getPcdFilesInDir(const string& directory)
{
  namespace fs = boost::filesystem;
  fs::path dir(directory);
 
  std::cout << "path: " << directory << std::endl;
  if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
    PCL_THROW_EXCEPTION (pcl::IOException, "No valid PCD directory given!\n");
    
  vector<string> result;
  fs::directory_iterator pos(dir);
  fs::directory_iterator end;           

  for(; pos != end ; ++pos)
    if (fs::is_regular_file(pos->status()) )
      if (fs::extension(*pos) == ".pcd")
      {
#if BOOST_FILESYSTEM_VERSION == 3
        result.push_back (pos->path ().string ());
#else
        result.push_back (pos->path ());
#endif
        cout << "added: " << result.back() << endl;
      }
    
  return result;  
}

namespace zc{
//#define _ZC_DBG_LOG 01
#ifdef _ZC_DBG_LOG
#define dbgPrint(format, ...) printf(format, ## __VA_ARGS__)
#define dbgPrintln(format, ...) printf(format"\n", ## __VA_ARGS__)
#else
#define dbgPrint
#define dbgPrintln
#endif

    //@author zhangxaochen
    //@brief get file names in *dir* with extension *ext*
    //@param dir file directory
    //@param ext file extension
    vector<string> getFnamesInDir(const string &dir, const string &ext){
        namespace fs = boost::filesystem;
        //fake dir:
        //cout << fs::is_directory("a/b/c") <<endl; //false

        //fs::path dirPath(dir); //no need
        cout << "path: " << dir << endl
            << "ext: " << ext <<endl;

        if(dir.empty() || !fs::exists(dir) || !fs::is_directory(dir)) //其实 is_directory 包含了 exists: http://stackoverflow.com/questions/2203919/boostfilesystem-exists-on-directory-path-fails-but-is-directory-is-ok
            PCL_THROW_EXCEPTION (pcl::IOException, "ZC: No valid directory given!\n");

        vector<string> res;
        fs::directory_iterator pos(dir),
            end;

        for(; pos != end; ++pos){
            if(fs::is_regular_file(pos->status()) && fs::extension(*pos) == ext){
#if BOOST_FILESYSTEM_VERSION == 3
                res.push_back(pos->path().string());
#else
                res.push_back(pos->path());
#endif
            }
        }

        if(res.empty())
            PCL_THROW_EXCEPTION(pcl::IOException, "ZC: no *" + ext + " files in current directory!\n");
        return res;
    }//getFnamesInDir

    using namespace cv;
    //@brief cv::erode 的弱化版: 规则为, px 的 radius 邻域内有零值, px=0; 否则不动
    //void erode0(InputArray src, OutputArray dst, int radius = 0){
    void erode0(Mat srcMat, Mat dstMat, int radius = 0){
        //Mat srcMat = src.getMat(); //之前的 InputArray 方式
        //dst.create(src.size(), src.type());
        //Mat dstMat = dst.getMat();
        
        //for(size_t i = 0; i < srcMat.rows; i++){
        //    for(size_t j = 0; j < srcMat.cols; j++){
        //        bool found0 = false;
        //        //遍历邻域：
        //        for(size_t i0 = min(0, i-radius); i0 < max(srcMat.rows, i+radius) && !found0; i0++){
        //            for(size_t j0 = min(0, j-radius); j0 < max(srcMat.cols, j+radius) && !found0; j0++){
        //                if(srcMat.at<ushort>(i0, j0) == 0)
        //                    found0 = true;
        //            }
        //        }

        //        //dstMat.at
        //    }
        //}

        Mat mask = srcMat > 0; //获取二值图蒙版
        //腐蚀二值图：
        int erosion_type = MORPH_RECT;
        Mat eroElement = getStructuringElement( erosion_type,
            Size( 2*radius + 1, 2*radius+1 ),
            Point( radius, radius ) );

        erode(mask, mask, eroElement);
        
        //srcMat.copyTo(dstMat, mask); //若 dstMat 原本==src, 则此句无 erode 效果, 弃用
        srcMat.copyTo(dstMat);
        dstMat.setTo(0, mask==0);
    }//erode0

}//namespace zc

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch
{          
  enum { EACH = 33 };
  SampledScopeTime(int& time_ms) : time_ms_(time_ms) {}
  ~SampledScopeTime()
  {
    static int i_ = 0;
    static boost::posix_time::ptime starttime_ = boost::posix_time::microsec_clock::local_time();
    time_ms_ += getTime ();
    if (i_ % EACH == 0 && i_)
    {
      boost::posix_time::ptime endtime_ = boost::posix_time::microsec_clock::local_time();
      cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )"
           << "( real: " << 1000.f * EACH / (endtime_-starttime_).total_milliseconds() << "fps )"  << endl;
      time_ms_ = 0;
      starttime_ = endtime_;
    }
    ++i_;
  }
private:    
  int& time_ms_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f 
getViewerPose (visualization::PCLVisualizer& viewer)
{
  Eigen::Affine3f pose = viewer.getViewerPose();
  Eigen::Matrix3f rotation = pose.linear();

  Matrix3f axis_reorder;  
  axis_reorder << 0,  0,  1,
                 -1,  0,  0,
                  0, -1,  0;

  rotation = rotation * axis_reorder;
  pose.linear() = rotation;
  return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CloudT> void
writeCloudFile (int format, const CloudT& cloud);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<RGB>& colors)
{    
  typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());
    
  pcl::copyPointCloud (points, *merged_ptr);      
  for (size_t i = 0; i < colors.size (); ++i)
    merged_ptr->points[i].rgba = colors.points[i].rgba;
      
  return merged_ptr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
{ 
  if (triangles.empty())
      return boost::shared_ptr<pcl::PolygonMesh>();

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width  = (int)triangles.size();
  cloud.height = 1;
  triangles.download(cloud.points);

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() ); 
  pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);
      
  mesh_ptr->polygons.resize (triangles.size() / 3);
  for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
  {
    pcl::Vertices v;
    v.vertices.push_back(i*3+0);
    v.vertices.push_back(i*3+2);
    v.vertices.push_back(i*3+1);              
    mesh_ptr->polygons[i] = v;
  }    
  return mesh_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CurrentFrameCloudView
{
  CurrentFrameCloudView() : cloud_device_ (480, 640), cloud_viewer_ ("Frame Cloud Viewer")
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

    cloud_viewer_.setBackgroundColor (0, 0, 0.15);
    cloud_viewer_.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 1);
    cloud_viewer_.addCoordinateSystem (1.0, "global");
    cloud_viewer_.initCameraParameters ();
    //cloud_viewer_.setPosition (0, 500);
    cloud_viewer_.setPosition (1000, 250);
    cloud_viewer_.setSize (640, 480);
    cloud_viewer_.setCameraClipDistances (0.01, 10.01);
  }

  void
  show (const KinfuTracker& kinfu)
  {
    kinfu.getLastFrameCloud (cloud_device_);

    int c;
    cloud_device_.download (cloud_ptr_->points, c);
    cloud_ptr_->width = cloud_device_.cols ();
    cloud_ptr_->height = cloud_device_.rows ();
    cloud_ptr_->is_dense = false;

    cloud_viewer_.removeAllPointClouds ();
    cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
    cloud_viewer_.spinOnce ();
  }

  void
  setViewerPose (const Eigen::Affine3f& viewer_pose) {
    ::setViewerPose (cloud_viewer_, viewer_pose);
  }

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  DeviceArray2D<PointXYZ> cloud_device_;
  visualization::PCLVisualizer cloud_viewer_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
  ImageView(int viz) : viz_(viz), paint_image_ (false), accumulate_views_ (false)
  {
    if (viz_)
    {
        viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
        viewerDepth_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);

        viewerScene_->setWindowTitle ("View3D from ray tracing");
        viewerScene_->setPosition (0, 0);
        viewerDepth_->setWindowTitle ("Kinect Depth stream");
        viewerDepth_->setPosition (640, 0);
        //viewerColor_.setWindowTitle ("Kinect RGB stream");
    }
  }

  void
  showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool registration, Eigen::Affine3f* pose_ptr = 0)
  {
    if (pose_ptr)
    {
        raycaster_ptr_->run(kinfu.volume(), *pose_ptr);
        raycaster_ptr_->generateSceneView(view_device_);
    }
    else
      kinfu.getImage (view_device_);

    if (paint_image_ && registration && !pose_ptr)
    {
      colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
      paint3DView (colors_device_, view_device_);
    }


    int cols;
    view_device_.download (view_host_, cols);
    if (viz_)
        viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());    

    //viewerColor_.showRGBImage ((unsigned char*)&rgb24.data, rgb24.cols, rgb24.rows);
    //cout<<"rgb24.cols: "<<rgb24.cols<<", "<<rgb24.rows<<endl;

#ifdef HAVE_OPENCV
    if (accumulate_views_)
    {
      views_.push_back (cv::Mat ());
      cv::cvtColor (cv::Mat (480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back (), CV_RGB2GRAY);
      //cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
    }
#endif
  }

  void
  showDepth (const PtrStepSz<const unsigned short>& depth) 
  { 
     if (viz_)
       viewerDepth_->showShortImage (depth.data, depth.cols, depth.rows, 0, 5000, true); 
  }
  
  void
  showGeneratedDepth (KinfuTracker& kinfu, const Eigen::Affine3f& pose)
  {            
    raycaster_ptr_->run(kinfu.volume(), pose);
    raycaster_ptr_->generateDepthImage(generated_depth_);    

    int c;
    vector<unsigned short> data;
    generated_depth_.download(data, c);

    if (viz_)
        //viewerDepth_->showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
        viewerGdepth_->showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
  }

  void
  toggleImagePaint()
  {
    paint_image_ = !paint_image_;
    cout << "Paint image: " << (paint_image_ ? "On   (requires registration mode)" : "Off") << endl;
  }

  int viz_;
  bool paint_image_;
  bool accumulate_views_;

  visualization::ImageViewer::Ptr viewerScene_;
  visualization::ImageViewer::Ptr viewerDepth_;
  //visualization::ImageViewer viewerColor_;

  //zhangxaochen:
  visualization::ImageViewer::Ptr viewerGdepth_; //generated-depth-viewer

  KinfuTracker::View view_device_;
  KinfuTracker::View colors_device_;
  vector<KinfuTracker::PixelRGB> view_host_;

  RayCaster::Ptr raycaster_ptr_;

  KinfuTracker::DepthMap generated_depth_;
  
#ifdef HAVE_OPENCV
  vector<cv::Mat> views_;
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView
{
  enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

  SceneCloudView(int viz) : viz_(viz), extraction_mode_ (GPU_Connected6), compute_normals_ (false), valid_combined_ (false), cube_added_(false)
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
    normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
    combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
    point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);

    if (viz_)
    {
        cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("Scene Cloud Viewer") );

        cloud_viewer_->setBackgroundColor (0, 0, 0);
        cloud_viewer_->addCoordinateSystem (1.0, "global");
        cloud_viewer_->initCameraParameters ();
        cloud_viewer_->setPosition (0, 500);
        cloud_viewer_->setSize (640, 480);
        cloud_viewer_->setCameraClipDistances (0.01, 10.01);

        cloud_viewer_->addText ("H: print help", 2, 15, 20, 34, 135, 246);
    }
  }

  void
  show (KinfuTracker& kinfu, bool integrate_colors)
  {
    viewer_pose_ = kinfu.getCameraPose();

    ScopeTimeT time ("PointCloud Extraction");
    cout << "\nGetting cloud... " << flush;

    valid_combined_ = false;

    if (extraction_mode_ != GPU_Connected6)     // So use CPU
    {
      kinfu.volume().fetchCloudHost (*cloud_ptr_, extraction_mode_ == CPU_Connected26);
    }
    else
    {
      DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_);             

      dbgPrintln("ZC: compute_normals_: %d", compute_normals_);
      if (compute_normals_)
      {
        kinfu.volume().fetchNormals (extracted, normals_device_);
        pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
        combined_device_.download (combined_ptr_->points);
        combined_ptr_->width = (int)combined_ptr_->points.size ();
        combined_ptr_->height = 1;

        valid_combined_ = true;
      }
      else
      {
        extracted.download (cloud_ptr_->points);
        cloud_ptr_->width = (int)cloud_ptr_->points.size ();
        cloud_ptr_->height = 1;
      }

      if (integrate_colors)
      {
        kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
        point_colors_device_.download(point_colors_ptr_->points);
        point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
        point_colors_ptr_->height = 1;
      }
      else
        point_colors_ptr_->points.clear();
    }
    size_t points_size = valid_combined_ ? combined_ptr_->points.size () : cloud_ptr_->points.size ();
    cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

    if (viz_)
    {
        cloud_viewer_->removeAllPointClouds ();
        if (valid_combined_)
        {
          visualization::PointCloudColorHandlerRGBCloud<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
          cloud_viewer_->addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
          cloud_viewer_->addPointCloudNormals<PointNormal>(combined_ptr_, 5); //原来 level==50
        }
        else
        {
          visualization::PointCloudColorHandlerRGBCloud<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
          cloud_viewer_->addPointCloud<PointXYZ> (cloud_ptr_, rgb);
        }
    }
  }

  void
  toggleCube(const Eigen::Vector3f& size)
  {
      if (!viz_)
          return;

      if (cube_added_)
          cloud_viewer_->removeShape("cube");
      else
        cloud_viewer_->addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

      cube_added_ = !cube_added_;
  }

  void
  toggleExtractionMode ()
  {
    extraction_mode_ = (extraction_mode_ + 1) % 3;

    switch (extraction_mode_)
    {
    case 0: cout << "Cloud exctraction mode: GPU, Connected-6" << endl; break;
    case 1: cout << "Cloud exctraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
    case 2: cout << "Cloud exctraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
    }
    ;
  }

  void
  toggleNormals ()
  {
    compute_normals_ = !compute_normals_;
    cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
  }

  void
  clearClouds (bool print_message = false)
  {
    if (!viz_)
        return;

    cloud_viewer_->removeAllPointClouds ();
    cloud_ptr_->points.clear ();
    normals_ptr_->points.clear ();    
    if (print_message)
      cout << "Clouds/Meshes were cleared" << endl;
  }

  void
  showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
  {
    if (!viz_)
       return;

    ScopeTimeT time ("Mesh Extraction");
    cout << "\nGetting mesh... " << flush;

    if (!marching_cubes_)
      marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

    DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);    
    mesh_ptr_ = convertToMesh(triangles_device);
    
    cloud_viewer_->removeAllPointClouds ();
    if (mesh_ptr_)
      cloud_viewer_->addPolygonMesh(*mesh_ptr_);
    
    cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
  }
    
  int viz_;
  int extraction_mode_;
  bool compute_normals_;
  bool valid_combined_;
  bool cube_added_;

  Eigen::Affine3f viewer_pose_;

  visualization::PCLVisualizer::Ptr cloud_viewer_;

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  PointCloud<Normal>::Ptr normals_ptr_;

  DeviceArray<PointXYZ> cloud_buffer_device_;
  DeviceArray<Normal> normals_device_;

  PointCloud<PointNormal>::Ptr combined_ptr_;
  DeviceArray<PointNormal> combined_device_;  

  DeviceArray<RGB> point_colors_device_; 
  PointCloud<RGB>::Ptr point_colors_ptr_;

  MarchingCubes::Ptr marching_cubes_;
  DeviceArray<PointXYZ> triangles_buffer_device_;

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuApp
{
  enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };
  
  KinFuApp(pcl::Grabber& source, float vsz, int icp, int viz, boost::shared_ptr<CameraPoseProcessor> pose_processor=boost::shared_ptr<CameraPoseProcessor> () ) : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), independent_camera_ (false),
      registration_ (false), integrate_colors_ (false), pcd_source_ (false), focal_length_(-1.f), capture_ (source), scene_cloud_view_(viz), image_view_(viz), time_ms_(0), icp_(icp), viz_(viz), pose_processor_ (pose_processor)
      ,png_source_(false), fid_(0), isReadOn_(false), dmatErodeRadius_(0)
      ,edgeViewer_("edgeViewer")
  {    
    //Init Kinfu Tracker
    Eigen::Vector3f volume_size = Vector3f::Constant (vsz/*meters*/);    
    kinfu_.volume().setSize (volume_size);

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
    Eigen::Vector3f t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);
    //zhangxaochen: 减小volume尺寸后, 初始相机位置也要改    //2016-3-17 10:41:14
    //t = Vector3f (volume_size(0)/2, volume_size(1)/2, -0.5);

    Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

    kinfu_.setInitalCameraPose (pose);
    kinfu_.volume().setTsdfTruncDist (0.030f/*meters*/);    
    kinfu_.setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
    //kinfu_.setDepthTruncationForICP(5.f/*meters*/);
    kinfu_.setCameraMovementThreshold(0.001f);

    if (!icp)
      kinfu_.disableIcp();

    //Init KinfuApp            
    tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr (new pcl::PointCloud<pcl::PointXYZI>);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols ()) );

    if (viz_)
    {
        scene_cloud_view_.cloud_viewer_->registerKeyboardCallback (keyboard_callback, (void*)this);
        image_view_.viewerScene_->registerKeyboardCallback (keyboard_callback, (void*)this);
        image_view_.viewerDepth_->registerKeyboardCallback (keyboard_callback, (void*)this);

        scene_cloud_view_.toggleCube(volume_size);
    }

    //虚拟立方体可视化  //2016-4-10 21:54:54
    edgeViewer_.setBackgroundColor (0, 0, 0);
    edgeViewer_.addCoordinateSystem();
    edgeViewer_.initCameraParameters();
    edgeViewer_.setSize(640, 480);
    edgeViewer_.setCameraClipDistances (0.01, 10.01);
  }//KinFuApp-ctor

  ~KinFuApp()
  {
    if (evaluation_ptr_)
      evaluation_ptr_->saveAllPoses(kinfu_);
  }

  void
  initCurrentFrameView ()
  {
    current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView ());
    current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
    current_frame_cloud_view_->setViewerPose (kinfu_.getCameraPose ());
  }

  //zhangxaochen:
  void initGenDepthView(bool viz){
      if(!viz)
          return;
      image_view_.viewerGdepth_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
      image_view_.viewerGdepth_->setWindowTitle("model-generated-depth");
      image_view_.viewerGdepth_->setPosition(1000, 100);
      image_view_.viewerGdepth_->registerKeyboardCallback(keyboard_callback, this);
  }//initGenDepthView

  void
  initRegistration ()
  {        
    registration_ = capture_.providesCallback<pcl::ONIGrabber::sig_cb_openni_image_depth_image> ();
    cout << "Registration mode: " << (registration_ ? "On" : "Off (not supported by source)") << endl;
    if (registration_)
      kinfu_.setDepthIntrinsics(KINFU_DEFAULT_RGB_FOCAL_X, KINFU_DEFAULT_RGB_FOCAL_Y);
  }
  
  void
  setDepthIntrinsics(std::vector<float> depth_intrinsics)
  {
    float fx = depth_intrinsics[0];
    float fy = depth_intrinsics[1];
    
    if (depth_intrinsics.size() == 4)
    {
        float cx = depth_intrinsics[2];
        float cy = depth_intrinsics[3];
        kinfu_.setDepthIntrinsics(fx, fy, cx, cy);
        //zhangxaochen: pRaycaster_ 是我专用, 要注意设定内参    //2016-5-12 11:51:54
        kinfu_.pRaycaster_->setIntrinsics(fx, fy, cx, cy);
        cout << "Depth intrinsics changed to fx="<< fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << endl;
    }
    else {
        kinfu_.setDepthIntrinsics(fx, fy);
        //zhangxaochen: pRaycaster_ 是我专用, 要注意设定内参    //2016-5-12 11:51:54
        kinfu_.pRaycaster_->setIntrinsics(fx, fy);
        cout << "Depth intrinsics changed to fx="<< fx << " fy=" << fy << endl;
    }
  }

  void 
  toggleColorIntegration()
  {
    if(registration_)
    {
      const int max_color_integration_weight = 2;
      kinfu_.initColorIntegration(max_color_integration_weight);
      integrate_colors_ = true;      
    }    
    cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
  }
  
  void 
  enableTruncationScaling()
  {
    kinfu_.volume().setTsdfTruncDist (kinfu_.volume().getSize()(0) / 100.0f);
  }

  void
  toggleIndependentCamera()
  {
    independent_camera_ = !independent_camera_;
    cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
  }
  
  void
  toggleEvaluationMode(const string& eval_folder, const string& match_file = string())
  {
    evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );
    if (!match_file.empty())
        evaluation_ptr_->setMatchFile(match_file);

    kinfu_.setDepthIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols (), 
        evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy) );
  }
  
  void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data)
  //sunguofei   //zhangxaochen: 不要随意改接口
  //void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data,vector<double> R_t)
  {        
    bool has_image = false;

    //sunguofei
    Eigen::Affine3f hint;//=(Eigen::Affine3f *)0;

    //zhangxaochen:
    const vector<float> &R_t = this->csvRtCurrRow_;
    if(this->png_source_ && this->csv_rt_hint_){
        kinfu_.imud_ = this->imud_;

        if(!this->imud_){ //若无 imud, hint 是 csv 中当前行的值
            Eigen::Matrix3f M_tmp;
            Eigen::Vector3f T_tmp;
            for (int i=0;i<3;++i)
            {
                T_tmp(i,0)=R_t[i]/1000; //mm->m
            }
            for (int i=0;i<9;++i)
            {
                M_tmp(i/3,i%3)=R_t[i+3];
            }
            //cout<<"M_tmp:"<<M_tmp<<endl; //正确！ 说明上面赋值法并不需要 row-major
            cout<<"determinant of rotation: "<<M_tmp.determinant()<<endl;
            hint.linear()=M_tmp;
            hint.translation()=T_tmp;
        }
        else{ //若 imud, hint 是 i->(i-1) 的 delta 量 (忽略 tvec部分)
            hint.linear() = this->csvDeltaRcurr_;
        }
    }

    if (has_data)
    {
        //zhangxaochen: 用opencv 绘制相机轨迹  //2016-5-26 11:02:15
        Affine3f poseBA = kinfu_.getCameraPose(); //BeforeAlign
        Vector3f tvecBA = poseBA.translation();

      depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
      if (integrate_colors_)
          image_view_.colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
    
      {
        SampledScopeTime fps(time_ms_);
    
        //run kinfu algorithm
        if (integrate_colors_)
          has_image = kinfu_ (depth_device_, image_view_.colors_device_);
        else{
          //has_image = kinfu_ (depth_device_,&hint);                  
          if(csv_rt_hint_){
              has_image = kinfu_ (depth_device_, &hint);                  
          }
          else{
              has_image = kinfu_ (depth_device_);                  
          }
        }
      }

      //zhangxaochen: 用opencv 绘制相机轨迹  //2016-5-26 11:02:15
      Affine3f poseAA = kinfu_.getCameraPose();
      Vector3f tvecAA = poseAA.translation();
      using namespace cv;
      static Mat tmat8u(480, 640, CV_8UC1, Scalar(0));
      Vector3f tcenter(1.5, 1.5, 0.4); //估计了一个旋转中心点, 逻辑是 0.4-(-0.3)=0.7, 大约是最近有效距离

      //暂时投影到 xz 平面, 舍弃y
      int scaleCoeff = 150;
      int pxXba = (tvecBA[0]-tcenter[0])*scaleCoeff+320,
          pxYba = (tvecBA[2]-tcenter[2])*scaleCoeff+240,
          pxXaa = (tvecAA[0]-tcenter[0])*scaleCoeff+320,
          pxYaa = (tvecAA[2]-tcenter[2])*scaleCoeff+240;

      line(tmat8u, Point(pxXba, pxYba), Point(pxXaa, pxYaa), 255);
      imshow("camera-trajectory", tmat8u);

      // process camera pose
      if (pose_processor_)
      {
        pose_processor_->processPose (kinfu_.getCameraPose ());
      }

      image_view_.showDepth (depth);
      //image_view_.showGeneratedDepth(kinfu_, kinfu_.getCameraPose());
      if(image_view_.viewerGdepth_ != nullptr)
          image_view_.showGeneratedDepth(kinfu_, kinfu_.getCameraPose());
    }

    if (scan_)
    {
      scan_ = false;
      scene_cloud_view_.show (kinfu_, integrate_colors_);
                    
      if (scan_volume_)
      {                
        cout << "Downloading TSDF volume from device ... " << flush;
        kinfu_.volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
        tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_.volume().getSize ());
        cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;
                
        cout << "Converting volume to TSDF cloud ... " << flush;
        tsdf_volume_.convertToTsdfCloud (tsdf_cloud_ptr_);
        cout << "done [" << tsdf_cloud_ptr_->size () << " points]" << endl << endl;        
      }
      else
        cout << "[!] tsdf volume download is disabled" << endl << endl;
    }

    if (scan_mesh_)
    {
        scan_mesh_ = false;
        scene_cloud_view_.showMesh(kinfu_, integrate_colors_);
    }
     
    if (viz_ && has_image)
    {
      Eigen::Affine3f viewer_pose = getViewerPose(*scene_cloud_view_.cloud_viewer_);
      image_view_.showScene (kinfu_, rgb24, registration_, independent_camera_ ? &viewer_pose : 0);
      //image_view_.showScene (kinfu_, rgb24, registration_, &viewer_pose); //zhangxaochen
    }    

    if (current_frame_cloud_view_)
      current_frame_cloud_view_->show (kinfu_);    
      
    if (viz_ && !independent_camera_)
      setViewerPose (*scene_cloud_view_.cloud_viewer_, kinfu_.getCameraPose());

    //zhangxaochen: 可视化虚拟立方体棱边(消隐前后) //2016-4-10 21:40:57
    if(kinfu_.edgeCloud_->size() > 0){
        //edgeViewer_.updatePointCloud(kinfu_.edgeCloud_, "edge_all");
        //edgeViewer_.updatePointCloud(kinfu_.edgeCloudVisible_, "edge_visible");
        edgeViewer_.removeAllPointClouds();

        const char *edgeAllId = "edge_all";
        const char *edgeVisibleId = "edge_visible";
#if 0   //V1, 着色方案1, 白色, 失败
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0); //样例
        visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
            //cgreen(kinfu_.edgeCloud_, 0, 255, 0),
            cgreen(0, 255, 0),
            cred(255, 0, 0);

        edgeViewer_.addPointCloud(kinfu_.edgeCloud_, cgreen, edgeAllId);
        edgeViewer_.addPointCloud(kinfu_.edgeCloudVisible_, cred, edgeVisibleId);
#elif 1 //V2, 着色方案2, 预先转换点云类型
        PointCloud<PointXYZRGB>::Ptr edgeCloudRgb(new PointCloud<PointXYZRGB>);
        copyPointCloud(*kinfu_.edgeCloud_, *edgeCloudRgb);
        visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cgreen(edgeCloudRgb, 0, 255, 0);
        edgeViewer_.addPointCloud(edgeCloudRgb, cgreen, edgeAllId);

        PointCloud<PointXYZRGB>::Ptr edgeCloudVisRgb(new PointCloud<PointXYZRGB>);
        copyPointCloud(*kinfu_.edgeCloudVisible_, *edgeCloudVisRgb);
        visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cred(edgeCloudVisRgb, 255, 0, 0);
        edgeViewer_.addPointCloud(edgeCloudVisRgb, cred, edgeVisibleId);
#endif
        edgeViewer_.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, edgeAllId);
        edgeViewer_.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, edgeVisibleId);

        ::setViewerPose(edgeViewer_, kinfu_.getCameraPose());
        edgeViewer_.spinOnce();
    }
  }
  
  void source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)  
  {
    //zhangxaochen:
    dbgPrint("--source_cb1_device\n\t--fid, time, baseline?: %u, %lu, %f\n", depth_wrapper->getFrameID(), depth_wrapper->getTimeStamp(), depth_wrapper->getBaseline());
//     cout << "--source_cb1_device" << endl
//          << "fid, time, baseline?: " << depth_wrapper->getFrameID() << ", " << depth_wrapper->getTimeStamp() << ", " << depth_wrapper->getBaseline() << endl;
    //cout << "\t--threadId: " << boost::this_thread::get_id() << endl;
    {
      //boost::mutex::scoped_try_lock lock(data_ready_mutex_);
      //改用 阻塞锁, 不要 try-lock: 
      //2015-9-7 18:13:59   读真实硬件可以做到不跳帧, 读 -pcd 导致阻塞, 暂时放弃. -pcd 解决跳帧本质上由 -pcd_fps < 0 控制
      boost::mutex::scoped_lock lock(data_ready_mutex_);

      if (exit_ || !lock){
          dbgPrint("\tif (exit_ || !lock)\n");
          //cout << "\tif (exit_ || !lock)" << endl;
          return;
      }
      dbgPrint("\t--lock~~\n");
      //cout << "\tlock~~" <<endl;

      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];     
    }
    dbgPrint("\t--notify_one~\n");
    //cout << "notify_one~" << endl;
    data_ready_cond_.notify_one();
    
  }

  void source_cb2_device(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
  {
      //zhangxaochen:
      dbgPrint("--source_cb2_device\n");
      //cout << "--source_cb2_device" << endl;
    {
      boost::mutex::scoped_try_lock lock(data_ready_mutex_);
      if (exit_ || !lock)
          return;
                  
      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];      
      
      rgb24_.cols = image_wrapper->getWidth();
      rgb24_.rows = image_wrapper->getHeight();
      rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 

      source_image_data_.resize(rgb24_.cols * rgb24_.rows);
      image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
      rgb24_.data = &source_image_data_[0];           
    }
    data_ready_cond_.notify_one();
  }


   void source_cb1_oni(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)  
  {        
    {
      boost::mutex::scoped_lock lock(data_ready_mutex_);
      if (exit_)
          return;
      
      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];     
    }
    data_ready_cond_.notify_one();
  }

  void source_cb2_oni(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
  {
    {
      boost::mutex::scoped_lock lock(data_ready_mutex_);
      if (exit_)
          return;
                  
      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];      
      
      rgb24_.cols = image_wrapper->getWidth();
      rgb24_.rows = image_wrapper->getHeight();
      rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 

      source_image_data_.resize(rgb24_.cols * rgb24_.rows);
      image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
      rgb24_.data = &source_image_data_[0];           
    }
    data_ready_cond_.notify_one();
  }

  void
  source_cb3 (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr & DC3)
  {
      //zhangxaochen:
      dbgPrint("--source_cb3\n");
      //cout << "--source_cb3" << endl;
    {
      boost::mutex::scoped_try_lock lock(data_ready_mutex_);
      if (exit_ || !lock)
        return;
      int width  = DC3->width;
      int height = DC3->height;
      depth_.cols = width;
      depth_.rows = height;
      depth_.step = depth_.cols * depth_.elemSize ();
      source_depth_data_.resize (depth_.cols * depth_.rows);

      rgb24_.cols = width;
      rgb24_.rows = height;
      rgb24_.step = rgb24_.cols * rgb24_.elemSize ();
      source_image_data_.resize (rgb24_.cols * rgb24_.rows);

      unsigned char *rgb    = (unsigned char *)  &source_image_data_[0];
      unsigned short *depth = (unsigned short *) &source_depth_data_[0];

      for (int i=0; i < width*height; i++) 
      {
        PointXYZRGBA pt = DC3->at (i);
        rgb[3*i +0] = pt.r;
        rgb[3*i +1] = pt.g;
        rgb[3*i +2] = pt.b;
        depth[i]    = pt.z/0.001;
      }
      rgb24_.data = &source_image_data_[0];
      depth_.data = &source_depth_data_[0];
    }
    data_ready_cond_.notify_one ();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  startMainLoop (bool triggered_capture)
  {   
    using namespace openni_wrapper;
    typedef boost::shared_ptr<DepthImage> DepthImagePtr;
    typedef boost::shared_ptr<Image> ImagePtr;
        
    //zhangxaochen:
//#if 0 //mutex-lock 获取 & 处理数据异步
    if(!this->png_source_){ //mutex-lock 获取 & 处理数据异步

        boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1_dev = boost::bind (&KinFuApp::source_cb2_device, this, _1, _2, _3);
        boost::function<void (const DepthImagePtr&)> func2_dev = boost::bind (&KinFuApp::source_cb1_device, this, _1);

        boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1_oni = boost::bind (&KinFuApp::source_cb2_oni, this, _1, _2, _3);
        boost::function<void (const DepthImagePtr&)> func2_oni = boost::bind (&KinFuApp::source_cb1_oni, this, _1);

        bool is_oni = dynamic_cast<pcl::ONIGrabber*>(&capture_) != 0;
        boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1 = is_oni ? func1_oni : func1_dev;
        boost::function<void (const DepthImagePtr&)> func2 = is_oni ? func2_oni : func2_dev;

        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&) > func3 = boost::bind (&KinFuApp::source_cb3, this, _1);

        bool need_colors = integrate_colors_ || registration_;
        if ( pcd_source_ && !capture_.providesCallback<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)>() )
        {
            std::cout << "grabber doesn't provide pcl::PointCloud<pcl::PointXYZRGBA> callback !\n";
        }
        boost::signals2::connection c = pcd_source_? capture_.registerCallback (func3) : need_colors ? capture_.registerCallback (func1) : capture_.registerCallback (func2);

        {
            boost::unique_lock<boost::mutex> lock(data_ready_mutex_);

            if (!triggered_capture)
                capture_.start (); // Start stream

            bool scene_view_not_stopped= viz_ ? !scene_cloud_view_.cloud_viewer_->wasStopped () : true;
            bool image_view_not_stopped= viz_ ? !image_view_.viewerScene_->wasStopped () : true;

            while (!exit_ && scene_view_not_stopped && image_view_not_stopped)
            { 
                dbgPrintln("mainloop-while, locking~");
                //cout << "threadId: " << boost::this_thread::get_id() << endl;

                //cout << "mainloop-while, locking~" <<endl;
                //zc: trigger 速度可能较慢, 
                if (triggered_capture){
                    capture_.start(); // Triggers new frame
                    //cout << "capture_.start" << endl;
                    dbgPrint("capture_.start\n");
                }
                dbgPrint("timed_wait~\n");
#if 0 //改用阻塞式 wait, 不是跳帧, 是为了与 source_cb1_device 双射输出
                data_ready_cond_.wait(lock);
                bool has_data = true; //really?
#elif 01 //改用 timed_wait, 但长时间 ( >10s)
                bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(3000));        
#else
                bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(100));        
#endif //wait & timed_wait
                static int fid = 0;
                dbgPrint("has_data: %d; %d\n", has_data, fid);
                //cout<<"has_data: "<<has_data<<"; "
                //    <<fid<<endl;
                fid++;


                try { this->execute (depth_, rgb24_, has_data); }
                catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
                catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

                if (viz_)
                    scene_cloud_view_.cloud_viewer_->spinOnce (3);
            }

            if (!triggered_capture)     
                capture_.stop (); // Stop stream
        }
        c.disconnect();
    }
//#elif 1 //用 for-loop, 同步获取 & 处理数据
    else{ //this->png_source_ == true, 用 for-loop, 同步获取 & 处理数据
#ifdef HAVE_OPENCV
        dbgPrintln("---------------HAVE_OPENCV");
        size_t pngVecSz = pngFnames_.size();
        for(size_t i = 0; i < pngVecSz; i++){
            string &fn = pngFnames_[i];
            if(synthetic_RT_.size() > 0){
                //sunguofei
#if 0
                //当前读进来的RT已经是dR dt了
                vector<double> rt=synthetic_RT_[i];
#else
                //sgf: 当前读进来的RT是原始的RT，并不是dR dt
                //zhangxaochen: 若要求 dR (是i->(i-1)), 下面全错了
//                 vector<double> rt;
//                 if (i==0)
//                 {
//                     rt=synthetic_RT_[i];
//                 }
//                 else
//                 {
//                     rt.resize(12,0);
//                     vector<double> rt0=synthetic_RT_[i-1];
//                     vector<double> rt1=synthetic_RT_[i];
//                     for (int j=0;j<3;++j)
//                     {
//                         rt[j]=rt1[j]-rt0[j];
//                     }
//                     for (int j=0;j<3;++j)
//                     {
//                         for (int k=0;k<3;++k)
//                         {
//                             rt[3+j*3+k]+=rt1[3+j*3+0]*rt0[3+k*3+0]+rt1[3+j*3+1]*rt0[3+k*3+1]+rt1[3+j*3+2]*rt0[3+k*3+2];
//                         }
//                     }
//                 }
#endif
                //this->csvRtCurrRow_ = rt; //大概是错的, 因为此处SGF的 rt 是 dR, dt //2016-5-9 00:52:04
                //this->csvRtCurrRow_ = synthetic_RT[i]; //就表示*直接*读到的一行
                //目前 OpencvCalib 生成一组去畸变的dmat-vec, 但是因为程序内是 WaitOneUpdateAll(rgb-img), 所以 dmat 序号跳帧, 所以此处改为以 dmat 文件名所示序号选择 csv 哪一行:	//2016-5-14 11:21:32
                size_t pos = fn.find_last_of('.') - 4;
                //dmat 在 oni 中实际对应的 frame ID, 起始为 1, 因为 -raw_frames/NiViewer 中的起始值都是1, 尽管 dg.GetFrameID 起始值是 0
                int dfid = std::stoi(fn.substr(pos, 4));
                int csvRowIdx = dfid - 1; //csv 矩阵的行号, 从0开始 (>=0)
                CV_Assert(csvRowIdx >= 0);

                this->csvRtCurrRow_ = synthetic_RT_[csvRowIdx];
                if(this->imud_){
                    typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
                    this->csvRcurr_ = Matrix3frm(csvRtCurrRow_.data() + 3); //注意: +3 是因为 vector12 前 3 项是 tvec //【必须】是 Matrix3frm
                    if(csvRowIdx == 0)
                        this->csvRprev_ = Matrix3f::Identity();
                    else
                        this->csvRprev_ = Matrix3frm(synthetic_RT_[csvRowIdx - 1].data() + 3); //注意: 【必须】是 Matrix3frm, 因为约定 syntheticRT 是按 row-major 存储 //2016-7-13 17:37:34

                    this->csvDeltaRcurr_ = csvRprev_.inverse() * csvRcurr_; //若 R 为 c->g(之前认知), 则此为 ci->c(i-1), 用于右乘, 与2014-IMU 一致
                    //this->csvDeltaRcurr_ = csvRcurr_.inverse() * csvRprev_; //感觉错, 反而对 //因为上面误用 Matrix3f(csvRtCurrRow_.data() + 3) (col-major)
                    //this->csvDeltaRcurr_ = csvRcurr_ * csvRprev_.inverse(); //若按 Rcurr = Rinc * Rcurr; 则 delta=Ri*R(i-1)', 用于左乘, 但在 c->g的假设下无实际物理意义
                }
            }

            printf("%s\n", fn.c_str());

            using namespace cv;
            Mat dmat = imread(fn, IMREAD_UNCHANGED);
            if(this->dmatErodeRadius_ > 0){
                Mat dmat8u;
                dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
                imshow("dmat.orig", dmat8u);
                //尝试腐蚀边缘, 以消除深度图边缘(不连续性)噪声  //2016-5-10 10:28:38
                int erosion_type = MORPH_RECT;
                //int erosion_size = 2;
                Mat eroElement = getStructuringElement( erosion_type,
                    //Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                    //Point( erosion_size, erosion_size ) );
                    Size( 2*dmatErodeRadius_ + 1, 2*dmatErodeRadius_+1 ),
                    Point( dmatErodeRadius_, dmatErodeRadius_ ) );

                //erode(dmat, dmat, eroElement); //这样不行, cv::erode 原理是在 kernel 范围内选取极小值, 会导致非边缘区域像素也被改变;
                zc::erode0(dmat, dmat, dmatErodeRadius_);
                dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
                imshow("dmatErode", dmat8u);
            }

            depth_.cols = dmat.cols;
            depth_.rows = dmat.rows;
            depth_.step = depth_.cols * depth_.elemSize();
            depth_.data = (ushort*)dmat.data;

            bool has_data = true; //fake flag;
            //try { this->execute (depth_, rgb24_, has_data, rt); }//@sunguofei, 不要随意改接口
            try { this->execute (depth_, rgb24_, has_data); }
            catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
            catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

            if(this->debug_level1_){
                Mat dmat8u;
                dmat.convertTo(dmat8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
                putText(dmat8u, "dmat8u", Point(0, 30), FONT_HERSHEY_PLAIN, 2, 255);
                putText(dmat8u, "fid: "+to_string((long long) i), Point(0, 50), FONT_HERSHEY_PLAIN, 2, 255);
                //imshow("dmat8u", dmat8u);

                Mat inpDmat = zc::inpaintCpu<ushort>(dmat),
                    inpDmat8u,
                    contMskShow;
                inpDmat.convertTo(inpDmat8u, CV_8UC1, 1.*UCHAR_MAX/1e4);
                contMskShow = inpDmat8u.clone();

                putText(inpDmat8u, "inpDmat8u", Point(0, 30), FONT_HERSHEY_PLAIN, 2, 255);
                putText(inpDmat8u, "fid: "+to_string((long long) i), Point(0, 50), FONT_HERSHEY_PLAIN, 2, 255);
                //imshow("inpDmat8u", inpDmat8u);
                //dmat8u.push_back(inpDmat8u); //==vconcat
                hconcat(dmat8u, inpDmat8u, dmat8u);
                //pyrDown(dmat8u, dmat8u);
                resize(dmat8u, dmat8u, Size(dmat8u.cols/2, dmat8u.rows/2));
                imshow("dmat8u", dmat8u);

                zc::MaskMap contMskDevice = kinfu_.getContMask();
                Mat contMskHost(contMskDevice.rows(), contMskDevice.cols(), CV_8UC1);
                //contMskDevice.download(contMskHost.data, contMskDevice.step()); //step -> 1024, 应为 640. 不知原因
                contMskDevice.download(contMskHost.data, contMskDevice.cols());
                contMskShow.setTo(UCHAR_MAX, contMskHost);
                imshow("contMskShow", contMskShow);
                printf("zc:contMskHost: %d\n", countNonZero(contMskHost));
                if(this->debug_level2_){
                    static int cmskCnt = 0;
                    char buf[4096];
                    sprintf(buf, "./cmsk-%06d.png", (int)cmskCnt);
                    cv::imwrite(buf, contMskShow);
                    cmskCnt++;
                }

                //contour-correspondence-candidate mask debug show
                zc::MaskMap cccDevice = kinfu_.getContCorrespMask();
                if(cccDevice.cols() > 0 && cccDevice.rows() > 0){
                    Mat cccHost(cccDevice.rows(), cccDevice.cols(), CV_8UC1);
                    cccDevice.download(cccHost.data, cccHost.cols * cccHost.elemSize());
                    imshow("cccHost", cccHost);

                    if(this->debug_level2_){
                        static int cccCnt = 0;
                        char buf[4096];
                        sprintf (buf, "./ccc-%06d.png", (int)cccCnt);
                        cv::imwrite (buf, cccHost);
                        cccCnt++;
                    }
                }

                DeviceArray2D<float> nmap_g_prev = kinfu_.getNmapGprev();
                if(nmap_g_prev.cols() > 0){ //若有数据
                    Mat nmap_g_prev_host = zc::nmap2rgb(nmap_g_prev);
                    imshow("nmap_g_prev_host", nmap_g_prev_host);

                    if(this->debug_level2_){
                        static int nmapCnt = 0;
                        char buf[4096];
                        sprintf (buf, "./nmap_g_prev-%06d.png", (int)nmapCnt);
                        cv::imwrite (buf, nmap_g_prev_host);
                        nmapCnt++;
                    }

                    zc::Image nmapColor = zc::renderNmap2(nmap_g_prev);
                    Mat nmapColorHost(nmapColor.rows(), nmapColor.cols(), CV_8UC3);
                    nmapColor.download(nmapColorHost.data, nmapColorHost.cols * nmapColorHost.elemSize());
                    imshow("nmapColorHost", nmapColorHost);
                }
            }
            int key = waitKey(this->png_fps_ > 0 && this->isReadOn_ ? int(1e3 / png_fps_) : 0);
            if(key==27) //Esc
                break;
            else if(key==' ')
                this->isReadOn_ = !this->isReadOn_;
        }//for-pngFnames_
    }//else //this->png_source_ == true
#endif //HAVE_OPENCV
//#endif //sync VS. async

  }//startMainLoop

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  writeCloud (int format) const
  {      
    const SceneCloudView& view = scene_cloud_view_;

    // Points to export are either in cloud_ptr_ or combined_ptr_.
    // If none have points, we have nothing to export.
    if (view.cloud_ptr_->points.empty () && view.combined_ptr_->points.empty ())
    {
      cout << "Not writing cloud: Cloud is empty" << endl;
    }
    else
    {
      if(view.point_colors_ptr_->points.empty()) // no colors
      {
        if (view.valid_combined_)
          writeCloudFile (format, view.combined_ptr_);
        else
          writeCloudFile (format, view.cloud_ptr_);
      }
      else
      {        
        if (view.valid_combined_)
          writeCloudFile (format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
        else
          writeCloudFile (format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  writeMesh(int format) const
  {
    if (scene_cloud_view_.mesh_ptr_) 
      writePolygonMeshFile(format, *scene_cloud_view_.mesh_ptr_);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  printHelp ()
  {
    cout << endl;
    cout << "KinFu app hotkeys" << endl;
    cout << "=================" << endl;
    cout << "    H    : print this help" << endl;
    cout << "   Esc   : exit" << endl;
    cout << "    T    : take cloud" << endl;
    cout << "    A    : take mesh" << endl;
    cout << "    M    : toggle cloud exctraction mode" << endl;
    cout << "    N    : toggle normals exctraction" << endl;
    cout << "    I    : toggle independent camera mode" << endl;
    cout << "    B    : toggle volume bounds" << endl;
    cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
    cout << "    C    : clear clouds" << endl;    
    cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
    cout << "    7,8  : save mesh to PLY, VTK" << endl;
    cout << "   X, V  : TSDF volume utility" << endl;
    cout << endl;
  }  

  bool exit_;
  bool scan_;
  bool scan_mesh_;
  bool scan_volume_;

  bool independent_camera_;

  bool registration_;
  bool integrate_colors_;
  bool pcd_source_;
  float focal_length_;
  
  pcl::Grabber& capture_;
  KinfuTracker kinfu_;

  SceneCloudView scene_cloud_view_;
  ImageView image_view_;
  boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;

  KinfuTracker::DepthMap depth_device_;

  pcl::TSDFVolume<float, short> tsdf_volume_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

  Evaluation::Ptr evaluation_ptr_;
  
  boost::mutex data_ready_mutex_;
  boost::condition_variable data_ready_cond_;
 
  std::vector<KinfuTracker::PixelRGB> source_image_data_;
  std::vector<unsigned short> source_depth_data_;
  PtrStepSz<const unsigned short> depth_;
  PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;

  int time_ms_;
  int icp_, viz_;

  boost::shared_ptr<CameraPoseProcessor> pose_processor_;

  //zhangxaochen:
  int fid_;
  bool png_source_;
  string pngDir_;
  vector<string> pngFnames_;
  bool isReadOn_;   //之前一直没用, 现在启用, 做空格暂停控制键 2016-3-26 15:46:37
  int png_fps_;
  bool hasRtCsv_; //是否（在 pngDir_）存在 {R, t} csv 描述文件
  bool csv_rt_hint_; //是否用 csv {R, t} 做初值？（不一定用）
  bool show_gdepth_; //show-generated-depth. 是否显示当前时刻 (模型, 视角) 对应的深度图
  bool debug_level1_; //是否download并显示一些中间调试窗口？运行时效率相关
  bool debug_level2_; //if true, imwrite 中间结果到文件, 且 debug_level1_=true (包含关系)
  bool imud_; //是否 csv_rt_hint_ 存的是 IMUD-fusion 所需的 Rt, 其特点: 只有 rmat 有用, t=(0,0,0)

  vector<float> csvRtCurrRow_; //csv 文件读到的当前一行
  //string icp_impl_str_;
  Matrix3f csvRcurr_, csvRprev_, csvDeltaRcurr_; //仅在 imud_=true 时才初始化, 不必写 row-major //2016-7-12 15:48:50

  //虚拟立方体棱边可视化  //2016-4-10 21:49:08
  visualization::PCLVisualizer edgeViewer_;
  
  int dmatErodeRadius_; //深度图边缘不连续性噪声, 会导致"棱边钝化", 设定腐蚀半径, 默认0   //2016-5-10 11:11:16

  //sunguofei
  vector<vector<float>> synthetic_RT_;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  static void
  keyboard_callback (const visualization::KeyboardEvent &e, void *cookie)
  {
    KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

    int key = e.getKeyCode ();

    if (e.keyUp ())    
      switch (key)
      {
      case 27: app->exit_ = true; break;
      case (int)'t': case (int)'T': app->scan_ = true; break;
      case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
      case (int)'h': case (int)'H': app->printHelp (); break;
      case (int)'m': case (int)'M': app->scene_cloud_view_.toggleExtractionMode (); break;
      case (int)'n': case (int)'N': app->scene_cloud_view_.toggleNormals (); break;      
      case (int)'c': case (int)'C': app->scene_cloud_view_.clearClouds (true); break;
      case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
      case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kinfu_.volume().getSize()); break;
      case (int)'7': case (int)'8': app->writeMesh (key - (int)'0'); break;
      case (int)'1': case (int)'2': case (int)'3': app->writeCloud (key - (int)'0'); break;      
      case '*': app->image_view_.toggleImagePaint (); break;

      case (int)'x': case (int)'X':
        app->scan_volume_ = !app->scan_volume_;
        cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
        break;
      case (int)'v': case (int)'V':
        cout << "Saving TSDF volume to tsdf_volume.dat ... " << flush;
        app->tsdf_volume_.save ("tsdf_volume.dat", true);
        cout << "done [" << app->tsdf_volume_.size () << " voxels]" << endl;
        cout << "Saving TSDF volume cloud to tsdf_cloud.pcd ... " << flush;
        pcl::io::savePCDFile<pcl::PointXYZI> ("tsdf_cloud.pcd", *app->tsdf_cloud_ptr_, true);
        cout << "done [" << app->tsdf_cloud_ptr_->size () << " points]" << endl;
        break;

        //zhangxaochen:
      case (int)' ':
          cout << "ZC: read a frame" << endl;
          app->isReadOn_ = !app->isReadOn_;
          break;

      default:
        break;
      }    
  }//keyboard_callback
};//struct KinFuApp

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename CloudPtr> void
writeCloudFile (int format, const CloudPtr& cloud_prt)
{
  if (format == KinFuApp::PCD_BIN)
  {
    cout << "Saving point cloud to 'cloud_bin.pcd' (binary)... " << flush;
    pcl::io::savePCDFile ("cloud_bin.pcd", *cloud_prt, true);
  }
  else
  if (format == KinFuApp::PCD_ASCII)
  {
    cout << "Saving point cloud to 'cloud.pcd' (ASCII)... " << flush;
    pcl::io::savePCDFile ("cloud.pcd", *cloud_prt, false);
  }
  else   /* if (format == KinFuApp::PLY) */
  {
    cout << "Saving point cloud to 'cloud.ply' (ASCII)... " << flush;
    pcl::io::savePLYFileASCII ("cloud.ply", *cloud_prt);
  
  }
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh)
{
  if (format == KinFuApp::MESH_PLY)
  {
    cout << "Saving mesh to to 'mesh.ply'... " << flush;
    pcl::io::savePLYFile("mesh.ply", mesh);		
  }
  else /* if (format == KinFuApp::MESH_VTK) */
  {
    cout << "Saving mesh to to 'mesh.vtk'... " << flush;
    pcl::io::saveVTKFile("mesh.vtk", mesh);    
  }  
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
print_cli_help ()
{
  cout << "\nKinFu parameters:" << endl;
  cout << "    --help, -h                              : print this message" << endl;  
  cout << "    --registration, -r                      : try to enable registration (source needs to support this)" << endl;
  cout << "    --current-cloud, -cc                    : show current frame cloud" << endl;
  cout << "    --save-views, -sv                       : accumulate scene view and save in the end ( Requires OpenCV. Will cause 'bad_alloc' after some time )" << endl;  
  cout << "    --integrate-colors, -ic                 : enable color integration mode (allows to get cloud with colors)" << endl;   
  cout << "    --scale-truncation, -st                 : scale the truncation distance and raycaster based on the volume size" << endl;
  cout << "    -volume_size <size_in_meters>           : define integration volume size" << endl;
  cout << "    --depth-intrinsics <fx>,<fy>[,<cx>,<cy> : set the intrinsics of the depth camera" << endl;
  cout << "    -save_pose <pose_file.csv>              : write tracked camera positions to the specified file" << endl;
  cout << "Valid depth data sources:" << endl; 
  cout << "    -dev <device> (default), -oni <oni_file>, -pcd <pcd_file or directory>" << endl;
  cout << "";
  cout << " For RGBD benchmark (Requires OpenCV):" << endl; 
  cout << "    -eval <eval_folder> [-match_file <associations_file_in_the_folder>]" << endl;
    
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
main (int argc, char* argv[])
{  
  if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
    return print_cli_help ();
  
  int device = 0;
  pc::parse_argument (argc, argv, "-gpu", device);
  pcl::gpu::setDevice (device);
  pcl::gpu::printShortCudaDeviceInfo (device);

//  if (checkIfPreFermiGPU(device))
//    return cout << endl << "Kinfu is supported only for Fermi and Kepler arhitectures. It is not even compiled for pre-Fermi by default. Exiting..." << endl, 1;
  
  boost::shared_ptr<pcl::Grabber> capture;
  
  bool triggered_capture = false;
  bool pcd_input = false;
  
  std::string eval_folder, match_file, openni_device, oni_file, pcd_dir;
  //zhangxaochen:
  std::string png_dir;
  vector<string> pngFnames;

  //sunguofei
  vector<vector<float>> R_t;

  int png_fps = 15;

  try
  {    
    if (pc::parse_argument (argc, argv, "-dev", openni_device) > 0)
    {
      capture.reset (new pcl::OpenNIGrabber (openni_device));
    }
    else if (pc::parse_argument (argc, argv, "-oni", oni_file) > 0)
    {
      triggered_capture = true;
      bool repeat = false; // Only run ONI file once
      capture.reset (new pcl::ONIGrabber (oni_file, repeat, ! triggered_capture));
    }
    else if (pc::parse_argument (argc, argv, "-pcd", pcd_dir) > 0)
    {
      float fps_pcd = 15.0f;
      pc::parse_argument (argc, argv, "-pcd_fps", fps_pcd);

      vector<string> pcd_files = getPcdFilesInDir(pcd_dir);    

      // Sort the read files by name
      sort (pcd_files.begin (), pcd_files.end ());
      capture.reset (new pcl::PCDGrabber<pcl::PointXYZRGBA> (pcd_files, fps_pcd, false));
      //zhangxaochen:
      //capture.reset (new pcl::PCDGrabber<pcl::PointXYZI> (pcd_files, fps_pcd, false));
      triggered_capture = true;
      pcd_input = true;
    }
    //zhangxaochen:
    else if (pc::parse_argument (argc, argv, "-png-dir", png_dir) > 0)
    {
        pngFnames = zc::getFnamesInDir(png_dir, ".png");
        if(0 == pngFnames.size()){
            cout << "No PNG files found in folder: " << png_dir << endl;
            return -1;
        }

        std::sort(pngFnames.begin(), pngFnames.end());

        //sunguofei
        ifstream synthetic_rt;
        string synRtFn = "syntheticRT.txt"; //不带路径, 意味着必须放在 png_dir 目录下; 默认值xxx
        pc::parse_argument(argc, argv, "-synRtFn", synRtFn); //尝试从命令行参数读取,替换默认值
        string path_rt=png_dir+"/"+synRtFn;
        cout<<path_rt<<endl;
        //if(!boost::filesystem::exists(path_rt))
        //    PCL_THROW_EXCEPTION (pcl::IOException, "file does not exist");

        if(boost::filesystem::exists(path_rt)){
            synthetic_rt.open(path_rt);
            //for (int i=0;i<pngFnames.size();++i)
            int i = 0; //用 while 替代 for, 针对当 png 是截取一部分, 而 syntheticRT.txt 是整个时   //2016-7-17 22:16:04
            while(1)
            {
                if(!synthetic_rt.good()){
                    cout<<"fstream synthetic_rt NOT good! i= "<<i<<endl;
                    break;
                }
                vector<float> rt;
                for (int j=0;j<12;++j)
                {
                    float tmp;
                    synthetic_rt>>tmp;
                    rt.push_back(tmp);
                }
                R_t.push_back(rt);
                i++;
            }
        }
        pc::parse_argument(argc, argv, "-png-fps", png_fps);
    }
    else if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
    {
      //init data source latter
      pc::parse_argument (argc, argv, "-match_file", match_file);
    }
    else
    {
      capture.reset( new pcl::OpenNIGrabber() );
        
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224932.oni", true, ! triggered_capture) );
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/reg20111229-180846.oni, true, ! triggered_capture) );    
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("/media/Main/onis/20111013-224932.oni", true, ! triggered_capture) );
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224551.oni", true, ! triggered_capture) );
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224719.oni", true, ! triggered_capture) );    
    }
  }
  catch (const pcl::PCLException& /*e*/) { return cout << "Can't open depth source" << endl, -1; }

  float volume_size = 3.f;
  pc::parse_argument (argc, argv, "-volume_size", volume_size);

  int icp = 1, visualization = 1;
  std::vector<float> depth_intrinsics;
  pc::parse_argument (argc, argv, "--icp", icp);
  pc::parse_argument (argc, argv, "--viz", visualization);
        
  std::string camera_pose_file;
  boost::shared_ptr<CameraPoseProcessor> pose_processor;
  if (pc::parse_argument (argc, argv, "-save_pose", camera_pose_file) && camera_pose_file.size () > 0)
  {
    pose_processor.reset (new CameraPoseWriter (camera_pose_file));
  }

  KinFuApp app (*capture, volume_size, icp, visualization, pose_processor);
  //zhangxaochen:
  app.pngFnames_ = pngFnames;
  app.png_fps_ = png_fps;
  if(!png_dir.empty()){
      app.pngDir_ = png_dir;
      app.png_source_ = true;
  }

  //启用 hint(目前 png_dir -> syntheticRT.txt) 做 ICP 初值
  app.csv_rt_hint_ = pc::find_switch (argc, argv, "-csv_rt_hint");
  app.imud_ = pc::find_switch(argc, argv, "-imud");

  app.debug_level1_ = pc::find_switch(argc, argv, "-dbg1");
  app.debug_level2_ = pc::find_switch(argc, argv, "-dbg2");
  if(app.debug_level2_){
      app.debug_level1_=true;
  }

  if(pc::find_switch(argc, argv, "--gen-depth") || pc::find_switch(argc, argv, "-gd")){
      app.show_gdepth_ = true; //似乎多余。暂时放着
      app.initGenDepthView(visualization);
  }

  string choose_icp_impl = "";
  if(pc::parse_argument(argc, argv, "-icp_impl", choose_icp_impl) > 0){
      if(choose_icp_impl == ""){ //default， kinfu 原来的实现
          app.kinfu_.icp_orig_ = true;
      }
      else{
          app.kinfu_.icp_orig_ = false;
          if(choose_icp_impl == "sgf_cpu"){ //孙国飞 cpu 上 kdTreeFlann 查找 contour-cue 对应点实现, 
              app.kinfu_.icp_sgf_cpu_ = true;
          }
          else if(choose_icp_impl == "cc_inc_weight"){ //仅增加 curr-depth 上找到的 contour-cue 的权重，仍用原 kinfu 对应点 search 算法
              //其实没增加 weight， 用的默认值 1；请用参数 -cc_inc_weight
              app.kinfu_.icp_sgf_cpu_ = false;
              app.kinfu_.icp_cc_inc_weight = true;
          }
      }
  }

  float contWeight = 1;
  if(pc::parse_argument(argc, argv, "-cc_inc_weight", contWeight) > 0 || pc::parse_argument(argc, argv, "-ccw", contWeight) > 0){
      //app.kinfu_.icp_sgf_cpu_ = false;
      //app.kinfu_.icp_cc_inc_weight = true;

      app.kinfu_.contWeight_ = contWeight;
  }

  //@contour-cue impl.: choose a computing normal method when getting CCCandidates
  //0(default): kinfu-orig.(raycast)
  //1: volume->[raycast]->genDepth->[zc::inpaint]->inpaintDmat->[createVmap]->vmap_cam_coo->[transformVmap]->vmap_g->[computeNormalsEigen]->nmap_g
  //2: contour-cue impl.
  int cc_norm_way = 0;
  if(pc::parse_argument(argc, argv, "-cc_norm_prev", cc_norm_way) > 0){
      app.kinfu_.cc_norm_prev_way_ = cc_norm_way;
  }

  //sunguofei
  app.synthetic_RT_=R_t;

  if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
    app.toggleEvaluationMode(eval_folder, match_file);

  if (pc::find_switch (argc, argv, "--current-cloud") || pc::find_switch (argc, argv, "-cc"))
    app.initCurrentFrameView ();

  if (pc::find_switch (argc, argv, "--save-views") || pc::find_switch (argc, argv, "-sv"))
    app.image_view_.accumulate_views_ = true;  //will cause bad alloc after some time  
  
  if (pc::find_switch (argc, argv, "--registration") || pc::find_switch (argc, argv, "-r"))
  {
    if (pcd_input) 
    {
      app.pcd_source_   = true;
      app.registration_ = true; // since pcd provides registered rgbd
    } 
    else 
    {
      app.initRegistration ();
    }
  }
  if (pc::find_switch (argc, argv, "--integrate-colors") || pc::find_switch (argc, argv, "-ic"))
    app.toggleColorIntegration();

  if (pc::find_switch (argc, argv, "--scale-truncation") || pc::find_switch (argc, argv, "-st"))
    app.enableTruncationScaling();
  
  if (pc::parse_x_arguments (argc, argv, "--depth-intrinsics", depth_intrinsics) > 0)
  {
    if ((depth_intrinsics.size() == 4) || (depth_intrinsics.size() == 2))
    {
       app.setDepthIntrinsics(depth_intrinsics);
    }
    else
    {
        pc::print_error("Depth intrinsics must be given on the form fx,fy[,cx,cy].\n");
        return -1;
    }   
  }
  //+++++++++++++++配准目标物指定, 预计可以是:
  //0. kinfu.orig 原流程, 原TSDF投射为vmap, nmap;
  //1. 虚拟立方体cube的伴随【影子】TSDF投射得到的深度图; 优: kinfu::raycast 直接得到 vmap, nmap; 劣: 相当于压缩, edge会损失
  //2. 我自己实现的cube-cloud 直接投射 cloud2depth, 与TSDF【无关】; 优: 可保留edge; 劣: 相当于自己实现的消隐、压缩算法, 精度未对比测量;
  //3. cube预置到老TSDF, 与depth共同形成模型, 再投射为深度图; 优: 虚实公摊权重, 似乎更合理; 劣: 并不单独考虑cube-edge
  int regObjId = 0;
  if(pc::parse_argument(argc, argv, "-regObj", regObjId) > 0){
      app.kinfu_.regObjId_ = regObjId;
  }//if "-regObj"

  //+++++++++++++++增加虚拟立方体做第0帧 2016-4-5 00:37:36
  int synF0weight = 1;
  pc::parse_argument(argc, argv, "-synF0wei", synF0weight);

  string synModelFn; //使用已知模型做配准, 作为第0帧! 当前用虚拟立方体, 如 zcCube.ply
  PointCloud<PointXYZ>::Ptr synModelCloudPtr(new PointCloud<PointXYZ>);
  if(pc::parse_argument(argc, argv, "-synF0", synModelFn) > 0){
      if(boost::ends_with(synModelFn, ".pcd")){
          io::loadPCDFile(synModelFn, *synModelCloudPtr);
      }
      else if(boost::ends_with(synModelFn, ".ply")){
          io::loadPLYFile(synModelFn, *synModelCloudPtr);
      }

      app.kinfu_.synModelCloudPtr_ = synModelCloudPtr;

#if 0
      //---------------测试写GPU内存
      zc::test_write_gpu_mem_in_syn_cu();//√
      return 0;

      //---------------测试命名空间 namespace zc:
      zc::foo_in_syn_cpp();//√, pcl_gpu_kinfu_debug.lib, .dll 里能搜到
      foo_in_cc_cpp();//√, 同上
      zc::foo_in_cc_cu();//√, contour_cue_impl.cu 这个就好好的, 为什么?
      zctmp::foo_test_cuda();//忘了加 PCL_EXPORTS   //√
      foo_test_no_ns();//√
      zc::foo_in_syn_cu();//×, syn_model_impl.cu 怎么也不行 答: dllexport, √
      zctmp::foo_test_ns_cpp(); //√
#endif

#if 01   //---------------测试 cloud2depth, 移到 kinfu_.operator() 开头:   //2016-4-8 12:19:47
      KinfuTracker kinfu = app.kinfu_;
      float fx, fy, cx, cy;
      kinfu.getDepthIntrinsics(fx, fy, cx, cy);
      pcl::device::Intr intr (fx, fy, cx, cy);
      Affine3f pose = kinfu.getCameraPose();

      cv::Mat depFromCloud;
      {
      zc::ScopeTimeMicroSec time("zc::cloud2depth"); //1mm.pcd, 11.8ms
      zc::cloud2depth(*synModelCloudPtr, pose, intr, 640, 480, depFromCloud);
      }
      {
      zc::ScopeTimeMicroSec time("zc::cloud2depthCPU"); //1mm.pcd, 91ms
      //zc::cloud2depthCPU(*synModelCloudPtr, pose, intr, 640, 480, depFromCloud);//√
      }
      cv::Mat dfc8u(depFromCloud.rows, depFromCloud.cols, CV_8UC1);
      double dmin, dmax;
      minMaxLoc(depFromCloud, &dmin, &dmax);
      depFromCloud.convertTo(dfc8u, CV_8UC1, 255./(dmax-dmin), -dmin*255./(dmax-dmin));
      cv::imshow("dfc8u", dfc8u);
      cv::waitKey(0);
      return 0;
#endif

      //pcl::device::initVolume(tsdf_volume.data()); //可写, √
      if(synModelCloudPtr->size() > 0){
          if(regObjId == 1){ //影子TSDF
              TsdfVolume &tsdf_volume = *app.kinfu_.tsdf_volume_shadow_;
              zc::cloud2tsdf(*synModelCloudPtr, synF0weight, tsdf_volume);
          }
          else if(regObjId == 3){ //原TSDF
              TsdfVolume &tsdf_volume = app.kinfu_.volume();
              zc::cloud2tsdf(*synModelCloudPtr, synF0weight, tsdf_volume);
              vector<float> tsdfHost; //仅用于统计!=0值个数, 检验 zc::cloud2tsdf 是否正确实现
              tsdf_volume.downloadTsdf(tsdfHost);
              int tsdfNoZeroCnt = 0;
              for(size_t i =0; i< tsdfHost.size(); i++){
                  if(tsdfHost[i] != 0)
                      tsdfNoZeroCnt++;
              }
              cout<<"tsdfNoZeroCnt: "<<tsdfNoZeroCnt<<endl; //16494

              ImageView &tview = app.image_view_; //temp 局部变量
              
              app.scene_cloud_view_.show(app.kinfu_, false);
              //app.scene_cloud_view_.cloud_viewer_->spin();
              while(!app.scene_cloud_view_.cloud_viewer_->wasStopped()){
                  //要坐标转换：
                  //Affine3f &pose = ::getViewerPose(*app.scene_cloud_view_.cloud_viewer_);
                  //上面结果不对, 不知道转到哪里去了, 尝试自己转换一下:
                  Affine3f &pose = app.scene_cloud_view_.cloud_viewer_->getViewerPose();
                  Matrix3f axis_reorder; //注意可能是 column-major
                  //axis_reorder << 0,  0,  1,
                  //                 -1,  0,  0,
                  //                  0, -1,  0;
                  axis_reorder << -1,  0,  0, //也不行
                                    0,  -1,  0,
                                    0,   0, -1;
                  //pose.linear() = pose.linear() * axis_reorder;

                  //tview.raycaster_ptr_->run(tsdf_volume, )
                  //tview.raycaster_ptr_->run(tsdf_volume, Affine3f::Identity());
                  //tview.raycaster_ptr_->run(tsdf_volume, app.kinfu_.getCameraPose());
                  tview.raycaster_ptr_->run(tsdf_volume, pose);
                  tview.raycaster_ptr_->generateSceneView(tview.view_device_); //下面仿照 showScene 写的
                  int cols;
                  tview.view_device_.download (tview.view_host_, cols);
                  //坐标系翻转 XY, 目前仍不懂坐标怎么回事？
                  vector<PixelRGB> &imgVec = tview.view_host_;
                  for(size_t i = 0; i < imgVec.size() / 2; i++){
                      PixelRGB tmpPx = imgVec[i];
                      int tailPos = imgVec.size() - 1 - i;
                      imgVec[i] = imgVec[tailPos];
                      imgVec[tailPos] = tmpPx;
                  }

                  tview.viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&tview.view_host_[0]), tview.view_device_.cols (), tview.view_device_.rows ());
                  app.scene_cloud_view_.cloud_viewer_->spinOnce();

              }

          }

      }
  }//if "-synF0"

  //+++++++++++++++加载棱边下标txt文件: //2016-4-10 17:57:53
  string edgeIdxFn;
  if(pc::parse_argument(argc, argv, "-edgeIdx", edgeIdxFn) > 0){
      //vector<int> edgeIdxVec;
      vector<int> &edgeIdxVec = app.kinfu_.edgeIdxVec_;

      ifstream fin(edgeIdxFn);
      while(!fin.eof()){
          int v;
          fin>>v;
          edgeIdxVec.push_back(v);
      }

      //直接在这里提取边框点云
//       ExtractIndices<PointXYZ> extractInd;
//       extractInd.setInputCloud(synModelCloudPtr);
//       extractInd.setIndices(boost::make_shared<vector<int>>(edgeIdxVec));
//       extractInd.filter(*app.kinfu_.edgeCloud_);

  }//if "-edgeIdx"

  //zhangxaochen:   //2016-5-10 15:13:04
  if(pc::parse_argument(argc, argv, "-erodeRad", app.dmatErodeRadius_) > 0 && app.dmatErodeRadius_ < 0)
      app.dmatErodeRadius_ = 0;

  // executing
  try { 
      //zhangxaochen:
#if 01
      app.startMainLoop (triggered_capture); 
#elif 01    //use png-dir
      //legacy
#endif
  }
  catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
  catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
  catch (const std::exception& /*e*/) { cout << "Exception" << endl; }

#ifdef HAVE_OPENCV
  for (size_t t = 0; t < app.image_view_.views_.size (); ++t)
  {
    if (t == 0)
    {
      cout << "Saving depth map of first view." << endl;
      cv::imwrite ("./depthmap_1stview.png", app.image_view_.views_[0]);
      cout << "Saving sequence of (" << app.image_view_.views_.size () << ") views." << endl;
    }
    char buf[4096];
    sprintf (buf, "./%06d.png", (int)t);
    cv::imwrite (buf, app.image_view_.views_[t]);
    printf ("writing: %s\n", buf);
  }
#endif

  return 0;
}
