//@author zhangxaochen  //2016-4-7 14:51:42
#ifndef _TEST_NS_H_
#define _TEST_NS_H_

//#include "pcl/pcl_exports.h"

namespace zctmp{

/*__declspec(dllexport)*/ void foo_test_ns_cpp();

}//namespace zctmp

/*__declspec(dllexport)*/ void foo_test_no_ns_cpp();

#endif _TEST_NS_H_
