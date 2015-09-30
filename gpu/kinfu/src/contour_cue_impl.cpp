#include "contour_cue_impl.h"

using namespace pcl;

//do-something
//2015-9-27 21:10:57 currently do NOTHING


void zc::test::testInpaintImplCpuAndGpu(const DepthMap &src, bool debugDraw /*= false*/ ){
    //ver CPU:
    Mat inpDmatCpu;
    size_t hostStep;
    {
        ScopeTimeMicroSec time("inpaintCpu");
        Mat dmatHost(src.rows(), src.cols(), CV_16UC1);
        hostStep = dmatHost.cols * dmatHost.elemSize();
        src.download(dmatHost.data, hostStep);
        {
            ScopeTimeMicroSec time("\t|-inpaintCpu-core");
            inpDmatCpu = inpaintCpu<ushort>(dmatHost);
        }
    }
    
    //ver GPU:
    Mat inpDmatGpu(src.rows(), src.cols(), CV_16UC1);
    {
        ScopeTimeMicroSec time("inpDmatGpu");
        DepthMap inpDmatGPUdev;
        {
            ScopeTimeMicroSec time("\t|-inpDmatGpu-core");
            inpaintGpu(src, inpDmatGPUdev);
        }
        inpDmatGPUdev.download(inpDmatGpu.data, hostStep);
    }
    //check if CPU & GPU impl. are identical:
    CV_Assert(countNonZero(inpDmatCpu != inpDmatGpu) == 0); 

    if(debugDraw){
        Mat tmp8u;
        inpDmatGpu.convertTo(tmp8u, CV_8UC1, 1. * UCHAR_MAX / 1e4);
        imshow("testInpaintImplCpuAndGpu", tmp8u);
    }
}//testInpaintImplCpuAndGpu
