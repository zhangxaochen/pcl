#ifndef _ZC_UTILITY_H_
#define _ZC_UTILITY_H_
#pragma once

#include <pcl/common/time.h>

namespace zc{
using namespace cv;

namespace test{
    void testInpaintImplCpuAndGpu(const DepthMap &src, bool debugDraw = false);

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

//@brief normal-map(SOA�ڴ�ṹ)ת��Ϊopencv-Mat, ���ڵ�����ʾ. ���ԸĽ����̼�: http://www.evernote.com/l/AY8AtmAVfmZBvKb96aRiC4VW_wQp59sAUzE/
//@param nmap, ����ͼ, ����Ϊ SOA �ڴ�ṹ(��Ϊ�õ�device::convert)
//@param debugDraw, �Ƿ� imshow
//@return cv::Mat of type CV_32FC3
Mat nmap2rgb(const MapArr &nmap, bool debugDraw = false);

}//namespace zc

#endif //_ZC_UTILITY_H_
