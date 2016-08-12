zhangxaochen:	//第一次生成的虚拟立方体,
命令行:
-rotStep 1 -pt_step 0.5 -ptsz 1 -cube_param 1500,1520,400,305,280,250 -load "E:\Github\pcl\doc\tutorials\content\sources\pcl_visualizer\zcSyntheticCube\zcCube-5mm.ply"
读取姿态配置文件: //文件名写死了, 不是参数
zcCubePose.csv
尺寸: 
305,280,250

缺陷是：
末端边缘棱边， 因为循环末尾判定时，浮点误差(这么说不准确)

---------------之后改成
1. 单独分别生成棱边, 面片
2. 保存棱边描述配置文件 zcCubeEdgeIdx-...txt
