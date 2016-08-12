//#include "test_ns.h"
//#include "pcl/pcl_exports.h"

namespace zctmp{
//__global__
//void foo_test_cuda_kernel(){}

__declspec(dllexport) void foo_test_ns_cpp(){}

}//namespace zctmp

//void zctmp::foo_test_ns_cpp(){}

__declspec(dllexport) void foo_test_no_ns_cpp(){}
