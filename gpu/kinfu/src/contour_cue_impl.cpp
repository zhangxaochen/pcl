#include "contour_cue_impl.h"
#include "zcUtility.h"

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

double zc::ScopeTimeMicroSec::getTimeMicros(){
    boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time();
    return static_cast<double>((end_time - start_time_).total_microseconds());
}//getTimeMicros

cv::Mat zc::nmap2rgb(const MapArr &nmap, bool debugDraw /*= false*/){
    DeviceArray2D<float3> nmap2d;
    device::convert(nmap, nmap2d);
    Mat nmaps_curr_host(nmap2d.rows(), nmap2d.cols(), CV_32FC3);
    //cout<<"nmap2d.step: "<<nmap2d.step()<<", "<<nmap2d.colsBytes()
    //    <<", "<<nmap2d.rows()<<"x"<<nmap2d.cols()<<"; "<<nmap2d.elem_step()<<endl; //7680, 7680, 480x640; 640
    nmap2d.download(nmaps_curr_host.data, nmaps_curr_host.cols * nmaps_curr_host.elemSize());
    Mat cn[3];
    cv::split(nmaps_curr_host, cn);
    //Mat negMsk = (cn[0] < 0) & (cn[1] < 0) & (cn[2] < 0); //标记全负mask, 并不好
    Mat negMsk = Mat::ones(nmaps_curr_host.size(), CV_8UC1) > 0;//*255;
    float magThresh = 0.5f;
    for(size_t i = 0; i < 3; i++){
        //negMsk &= (cn[i] > 0 & cn[i] < magThresh) | (cn[i] < 0 & cn[i] < -magThresh); //标记： 若很负，或微正。逻辑错：万一像素某通道微负，其余很负，则漏标记
        negMsk &= (abs(cn[i]) > magThresh & cn[i] < 0) | (abs(cn[i]) <= magThresh); //标记： 若很负，或绝对值很小
        cout<<"negMsk.at<uchar>(111, 111): "<<(int)negMsk.at<uchar>(111, 111)<<endl;
    }
    //nmaps_curr_host = cv::abs(nmaps_curr_host); //全取绝对值并不对！
    Mat(-nmaps_curr_host).copyTo(nmaps_curr_host, negMsk); //对某些不易显示的法向取反

    if(debugDraw)
        imshow("nmaps_curr_host", nmaps_curr_host);
    //cout<<"nmaps_curr_host [260x160], [260x260]: "<<endl
    //    <<nmaps_curr_host(Rect(260, 160, 3, 3))<<endl
    //    <<nmaps_curr_host(Rect(260, 260, 3, 3))<<endl
    //    <<nmaps_curr_host(Rect(50, 400, 3, 3))<<endl
    //    <<nmaps_curr_host(Rect(100, 400, 3, 3))<<endl
    //    <<nmaps_curr_host(Rect(20, 287, 3, 3))<<endl;

    return nmaps_curr_host;
}//nmap2rgb
