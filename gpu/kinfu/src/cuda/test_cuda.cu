#include "test_cuda.h"

namespace zctmp{
__global__
void foo_test_cuda_kernel(){}

void foo_test_cuda(){}

}//namespace zctmp

//void zctmp::foo_test_cuda(){}

void foo_test_no_ns(){}
