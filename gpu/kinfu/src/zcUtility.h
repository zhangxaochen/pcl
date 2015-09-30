#ifndef _ZC_UTILITY_H_
#define _ZC_UTILITY_H_
#pragma once

#include <pcl/common/time.h>

namespace zc{
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

}//namespace zc

#endif //_ZC_UTILITY_H_
