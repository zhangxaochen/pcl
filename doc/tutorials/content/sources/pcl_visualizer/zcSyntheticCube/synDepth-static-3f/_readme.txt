zhangxaochen:	//2016-6-22 15:51:09
1. 拷贝自 synDepth-rt-3f, 只用 f0 第一张图片, 复制多份, 当做相机一直不动情形下采集的数据


+++++++++++++++原 synDepth-rt-3f/_readme.txt 内容:
zhangxaochen:	2016-6-15 10:37:52
1. zcSyntheticCube 命令行:
-rotStep 1 -pt_step 1 -ptsz 1 -cube_param 1500,1520,400,399,300,250 -load "E:\Github\pcl\_build.vc10\bin\cloud_bin.pcd"
因为是手动拖动, 仅仅命令行不可能重现这组数据

2. 手动拖动, R+t 都有, 每次移动量微小, 确保 icp 能够成功
	其实 kinfu 中 icp 是 projectIve data association + 距离&角度滤波, 不是严格的 最近点做对应点 ICP

3. 命令行没指定相机内参, 使用代码中默认值: 525.5,525.5,320,240

