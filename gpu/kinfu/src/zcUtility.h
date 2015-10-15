#ifndef _ZC_UTILITY_H_
#define _ZC_UTILITY_H_
#pragma once

#include <pcl/common/time.h>

namespace zc{
using namespace cv;

namespace test{
    //@brief 测试 inpaint 算法实现正确性：1. CPU & GPU 结果一致; 2. GPU 更高效
    void testInpaintImplCpuAndGpu(const DepthMap &src, bool debugDraw = false);

    //@brief 测试 vmap 是否正确, 通过 nmap opencv 绘制观察. 防止某些 vmap 坐标转换 etc. 之后出错.
    //@param[in] vmap, vertex map
    //@param[in] winName, opencv imshow window name
    void testVmap(const MapArr &vmap, const char *winName);
}//namespace test

class ScopeTimeMicroSec : public StopWatch{
    std::string title_;
public:
    ScopeTimeMicroSec(const char *title=""):
      title_(title){}

      ~ScopeTimeMicroSec(){
          double val = this->getTimeMicros();
          std::cerr << title_ << " took " << val << " micro seconds.\n";
      }

      //@brief return execution time in micro-seconds
      //@issue&fix move function body to .cpp file to avoid http://stackoverflow.com/questions/11540962/tell-nvcc-to-not-preprocess-host-code-to-avoid-boost-compiler-redefinition
      //traceback: @sgf's-PC: http://codepad.org/3M0tgmrb
      double getTimeMicros();

};//class ScopeTimeMillis

//@brief normal-map(SOA内存结构)转换为opencv-Mat, 用于调试显示. 策略改进过程见: http://www.evernote.com/l/AY8AtmAVfmZBvKb96aRiC4VW_wQp59sAUzE/
//@param nmap, 法向图, 必须为 SOA 内存结构(因为用到device::convert)
//@param debugDraw, 是否 imshow
//@return cv::Mat of type CV_32FC3
PCL_EXPORTS Mat nmap2rgb(const MapArr &nmap, bool debugDraw = false);

}//namespace zc

#endif //_ZC_UTILITY_H_
