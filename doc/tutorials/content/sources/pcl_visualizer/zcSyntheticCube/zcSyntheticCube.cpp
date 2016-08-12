//@author zhangxaochen
//@brief 1. �������������(ע��߶�����ʵ������"��"��?); 2. ���ɸ�������ӽǵ����ͼ

#include <iostream>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <time.h>

#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>

#include <boost/algorithm/string/predicate.hpp>

//#include "syn_model_impl.h"
//#include "cuda_runtime_api.h"

#include <opencv2/opencv.hpp>
using namespace cv;

using namespace std;
using namespace pcl;
using namespace visualization;
using namespace Eigen;

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;

bool isSavePCD_ = false;
bool isAddViewerCoo_ = false;

//���̿���������ת:
Affine3f cubePose_; //��ת�����̬����
float rotStep_ = 1.f; //һ����ת�ĽǶ�ֵ
bool isTransformCube_ = false;
const char *poseConfFn_ = "zcCubePose.csv";
const char *edgeIdxConfFn = "zcCubeEdgeIdx.txt";

//���̿�������ƽ��:
float tranStep_ = 10.f; //һ��ƽ�Ƶľ��� mm(��Ĭ�� 10mm=1cm)
const float m2mm = 1000;

//�������ͼ��ʾ������ //2016-6-12 20:57:45
bool isUpdateDepth_ = false;
bool isSaveDepth_ = false;
cv::Mat depFromCloud_;
int dmapCnt_ = 0; //���ͼpng ����������, ctrl+x ������
Affine3f pose_viewer_;
Affine3f pose_f0_;
ofstream fout_camPose_;

//������ @E:\Github\pcl\gpu\kinfu\src\internal.h
struct Intr
{
    float fx, fy, cx, cy;
    Intr () {}
    Intr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    Intr operator()(int level_index) const
    { 
        int div = 1 << level_index; 
        return (Intr (fx / div, fy / div, cx / div, cy / div));
    }
};

//������ @E:\Github\pcl\gpu\kinfu\src\syn_model_impl.cpp
//@param[in] pose, cam->global, �����̬
void cloud2depthCPU(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, Mat &depOut){
    //��������ͼ�����ڴ�: CPU
    depOut = Mat::zeros(imHeight, imWidth, CV_16UC1);
    Matrix3frm Rrm = pose.linear();
    Vector3f t = pose.translation();
    Matrix3frm Rrm_inv = Rrm.inverse(); //inv �Ƿ񻹱����� Matrix3frm�� //�����̬������ת

    int sz = cloud.size();
    for(int i = 0; i < sz; i++){
        //---------------���涼�Ǵ� cloud2depthKernel �������, GPU���� bug, ���� CPU �ϲ���:
        PointXYZ pt4 = cloud[i];
        Vector3f pi(pt4.x, pt4.y, pt4.z);

        Vector3f pi2cam = pi - t; //pi �� camera ԭ������
        Vector3f vcam = Rrm_inv * pi2cam; //pi world->camera

        int u = int((vcam.x() * intr.fx) / vcam.z() + intr.cx); //����ͼ������ (u,v), Ӧ���� m->pixel? ���Ǻ����أ�Ū����� ��: ȷʵ m, ������� (u, v) Ҳû��
        int v = int((vcam.y() * intr.fy) / vcam.z() + intr.cy);

        //������ͼ��Χ�ڣ�
        if(u < 0 || u > imWidth-1 ||
            v < 0 || v > imHeight-1)
            continue;

        const int MAX_DEPTH = 10000;
        float z = vcam.z() * 1000;
        ushort oldz = depOut.at<ushort>(v, u);
        if(0 < z && z < MAX_DEPTH && //��z���ֵ����
            (0 == oldz || oldz > z))//�Ҵ�����δ����, ��z��������ֵС
            depOut.at<ushort>(v, u) = (ushort)z; //�˴����ص��� float->ushort �ض�
    }//for
}//cloud2depthCPU

//������ @E:\Github\pcl\gpu\kinfu\tools\kinfu_app_sim.cpp
Eigen::Affine3f getViewerPose (visualization::PCLVisualizer& viewer){
    Eigen::Affine3f pose = viewer.getViewerPose();
    Eigen::Matrix3f rotation = pose.linear();

    Matrix3f axis_reorder;  
    axis_reorder << 0,  0,  1,
        -1,  0,  0,
        0, -1,  0;

    rotation = rotation * axis_reorder;
    pose.linear() = rotation;
    return pose;
}//getViewerPose

void processPose (ofstream &out_stream, const Eigen::Affine3f &pose){
    if (out_stream.good ())
    {
        // convert 3x4 affine transformation to quaternion and write to file
        Eigen::Quaternionf q (pose.rotation ());
        Eigen::Vector3f t (pose.translation ());
        // write translation , quaternion in a row
        out_stream << t[0] << "," << t[1] << "," << t[2]
        << "," << q.w () << "," << q.x ()
            << "," << q.y ()<< ","  << q.z () << std::endl;
    }
}//processPose

void printUsage(const char *progName){
    cout<<"Usage: "<<progName<<" [options]\n"
        <<"Options:\n"
        <<"---------------\n"
        <<"-cube_param <cenX,cenY,cenZ,lenX,lenY,lenZ> //scale: mm; param: [comma] separated, no [space]; cenX->centroid coo X, lenX->length of side X, etc.\n"
        <<"-coo_s <scale> //[OPTIONAL] scale of the coordinate axes\n"
        <<"-ptsz <PointSize> //[OPTIONAL] size of points\n"
        <<"-pt_step <step> //step(interval) between 2 points in *mm*"
        <<"-rotStep <step> //*DEGREE* of a rotate step\n"
        <<"-load <fileName> //pcd|ply fileName to load and show"
        <<"\n";
}//printUsage

inline bool fileExists(const std::string& name){
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0); 
}//fileExists

//@brief load 4x4 (R,t) *pose* if *fname* exists; otherwise return false.
bool loadPose(Affine3f &pose, const char *fname){
    cout<<"loadPose() loading: "<<fname<<endl;
    if(!fileExists(fname)){
        cout<<fname<<" does NOT exist, do nothing!"<<endl;
        return false;
    }
    ifstream fin(fname);
    int rows = 4,
        cols = 4;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float e;
            fin>>e;
            pose(i,j) = e;
        }
    }
    return true;
}//loadPose

//@brief save the *pose* to file *fname*, will overwrite the file if it already exists
void savePose(const Affine3f &pose, const char *fname){
    cout<<"savePose() saving:\n"<<pose.matrix()<<"to file: "<<fname<<endl;
    if(fileExists(fname))
        cout<<"overwriting the file!!!"<<endl;
    ofstream fout;
    fout.open(fname, ios::out);
    //Matrix3f rot = pose.rotation();
    //Vector3f t = pose.translation();

    //fout<<pose(0,0)<<", "<<pose(0,1)<<", "<<pose(0,2)<<", "<<pose(0,3)<<endl;
    fout<<pose.matrix()<<endl;
}//savePose

//@brief ����ĳidxVec���ļ�, ����߽�㼯����; û�ж�Ӧ load ����
void saveIdxFile(const char *fname, const vector<int> &idxVec){
    cout<<"saveIdxFile() saving "<<idxVec.size()<<" idx to file: "<<fname<<endl;
    if(fileExists(fname))
        cout<<"overwriting the file!!!"<<endl;
    ofstream fout;
    fout.open(fname, ios::out);

    int vecSz = idxVec.size();
    for(int i = 0; i < vecSz; i++){
        fout<<idxVec[i]<<endl;
    }
    fout<<endl;
}//saveIdxFile

void kbCallback(const KeyboardEvent &e, void *cookie){
    cout<<"kbCallback, getKeyCode, getKeySym: "<<int(e.getKeyCode())<<", "<<e.getKeySym()<<endl;
    //if(e.getKeyCode() == 's'){ //���µ��𶼴���
    //if(e.getKeySym() == "sa"){ //����, sym �����ܽ��ܶ���ַ�, �� getKeyCode ûɶ����. �����Խ��������ַ� Control_L, etc
    if(e.keyUp()){
        if(19 == e.getKeyCode()){ //upֻ��һ��, down�����������
                                   //ctrl+s
            cout<<"save pcd file"<<endl;
            isSavePCD_ = true;
        }
        if(1 == e.getKeyCode()){ //ctrl+a, ������������Կ���
            cout<<"isAddViewerCoo_"<<endl;
            isAddViewerCoo_ = true;
        }
        if(4 == e.getKeyCode()){ //ctrl+d, �������ͼ //2016-6-12 18:42:31
            cout<<"isUpdateDepth_"<<endl;
            isUpdateDepth_ = true;
        }
        if(23 == e.getKeyCode()){ //ctrl+w, �������ͼΪ 16bit png //2016-6-12 20:45:47
            cout<<"isSaveDepth_"<<endl;
            isSaveDepth_ = true;
        }
        if(24 == e.getKeyCode()){ //ctrl+x, ���� png ����������
            cout<<"~~~reset dmapCnt_ = 0;"<<endl;
            dmapCnt_ = 0;
        }

        //---------------���̿�����ת
        if('a' == e.getKeyCode()){//����, ��Y��
            cubePose_.rotate(AngleAxisf(rotStep_, Vector3f::UnitY()));
            isTransformCube_ = true;
        }
        else if('d' == e.getKeyCode()){
            cubePose_.rotate(AngleAxisf(-rotStep_, Vector3f::UnitY()));
            isTransformCube_ = true;
        }
        else if('w' == e.getKeyCode()){//����, ��X��
            cubePose_.rotate(AngleAxisf(rotStep_, Vector3f::UnitX()));
            isTransformCube_ = true;
        }
        else if('s' == e.getKeyCode()){
            cubePose_.rotate(AngleAxisf(-rotStep_, Vector3f::UnitX()));
            isTransformCube_ = true;
        }
        else if('z' == e.getKeyCode()){//��Z��
            cubePose_.rotate(AngleAxisf(rotStep_, Vector3f::UnitZ()));
            isTransformCube_ = true;
        }
        else if('x' == e.getKeyCode()){
            cubePose_.rotate(AngleAxisf(-rotStep_, Vector3f::UnitZ()));
            isTransformCube_ = true;
        }
        //---------------ƽ�ƣ�
        if("Left" == e.getKeySym()){//��, ��X��
            //cubePose_.translation()<<-tranStep_,0,0;
            cubePose_.translate(Vector3f(-tranStep_,0,0));
            isTransformCube_ = true;
        }
        if("Right" == e.getKeySym()){//��
            //cubePose_.translation()<<tranStep_,0,0;
            cubePose_.translate(Vector3f(tranStep_,0,0));
            isTransformCube_ = true;
        }
        if("Up" == e.getKeySym()){//��, ��Y��
            //cubePose_.translation()<<0,tranStep_,0;
            cubePose_.translate(Vector3f(0,tranStep_,0));
            isTransformCube_ = true;
        }
        if("Down" == e.getKeySym()){//��
            //cubePose_.translation()<<0,-tranStep_,0;
            cubePose_.translate(Vector3f(0,-tranStep_,0));
            isTransformCube_ = true;
        }
        if("Prior" == e.getKeySym()){//PageUp, +, ��Z��
            //cubePose_.translation()<<0,0,tranStep_;
            cubePose_.translate(Vector3f(0,0,tranStep_));
            isTransformCube_ = true;
        }
        if("Next" == e.getKeySym()){//PageDown, -
            //cubePose_.translation()<<0,0,-tranStep_;
            cubePose_.translate(Vector3f(0,0,-tranStep_));
            isTransformCube_ = true;
        }

        //---------------����任������̬�������ļ�
        if('p' == e.getKeyCode()){
            savePose(cubePose_, poseConfFn_);
        }
    }//if(e.keyUp())
//     if("Control_L" == e.getKeySym() && e.keyDown()){//ctrl-L ����
//         
//     }//if(e.keyDown())
}//kbCallback

//@brief �Լ����� pose ��������������
//@param[in] pose, ��ʾ��������̬
void zcAddCoordSystem(PCLVisualizer &viewer, double scale = 1.0, const Affine3f& pose = Affine3f::Identity()){
    Matrix3f rot = pose.rotation();
    Vector3f position = pose.translation();
    PointXYZ pt0;// = position; //cannot convert from 'Eigen::Vector3f' to 'pcl::PointXYZ'
    pt0.getVector3fMap() = position; //http://stackoverflow.com/questions/17129018
    

    //pose���м�Ϊ��������;
    Vector3f 
    /*PointXYZ*/ ax = rot.col(0) * scale,
             ay = rot.col(1) * scale,
             az = rot.col(2) * scale;
    
    PointXYZ ptx, pty, ptz;
    ptx.getVector3fMap() = position + ax;
    pty.getVector3fMap() = position + ay;
    ptz.getVector3fMap() = position + az;

    static int callCnt = -1; //һ�ν�����, ���ô˺����Ĵ���    //2016-6-14 17:04:26
    callCnt++;
    string callCntStr = to_string((long long)callCnt);

#if 0   //V1, addArrow, �Ŵ�������ͻ��, ����, ����
    //viewer.addArrow(position, ax, 1, 0, 0, "axisX");
    //viewer.addArrow(position, ay, 0, 1, 0, "axisY");
    //viewer.addArrow(position, az, 0, 0, 1, "axisZ");
    viewer.addArrow(pt0, ptx, 1, 0, 0, false, string("zcAxisX"));
    viewer.addArrow(pt0, pty, 0, 1, 0, false, string("zcAxisY"));
    viewer.addArrow(pt0, ptz, 0, 0, 1, false, string("zcAxisZ"));
    //���� display_length, һ��Ҫ string, ���� char*; ����ʾ˫���ͷ; 
    //viewer.addArrow(PointXYZ(-1,-1,-1), PointXYZ(2,1,3), 0, 1, 0, string("testArrow"));
#elif 1 //V2, 
    viewer.addLine(pt0, ptx, 1, 0, 0, "zcAxisX"+callCntStr);
    viewer.addLine(pt0, pty, 0, 1, 0, "zcAxisY"+callCntStr);
    viewer.addLine(pt0, ptz, 0, 0, 1, "zcAxisZ"+callCntStr);
#endif
}//zcAddCoordSystem

void main(int argc, char **argv){
    //---------------�趨תΪ���ͼ���õ��ڲ�, ��λ: ���� //2016-6-12 20:49:44
    Intr intr(525.5f, 525.5f, 320, 240); //Ĭ��ֵ, ע�����ͼ�ߴ�̶� 640*480
    vector<float> intr4;
    if(console::parse_x_arguments(argc, argv, "-intr", intr4) > 0){
        intr.fx = intr4[0];
        intr.fy = intr4[1];
        intr.cx = intr4[2];
        intr.cy = intr4[3];
    }

    //---------------��������������, ָ��(-x,-y,-z)��������, ָ�������
    vector<float> cenParams; //corner
    float cnrX, cnrY, cnrZ, lenX, lenY, lenZ,
        cenX, cenY, cenZ;//����, mass centroid
    float cooScale = 1.f;
    float ptsz = 1.f;
    if(console::parse_x_arguments(argc, argv, "-cube_param", cenParams) > 0){
        cenX = cenParams[0] / m2mm;
        cenY = cenParams[1] / m2mm;
        cenZ = cenParams[2] / m2mm;

        lenX = cenParams[3] / m2mm;
        lenY = cenParams[4] / m2mm;
        lenZ = cenParams[5] / m2mm;

//         cnrX = cenX - lenX / 2;
//         cnrY = cenY - lenY / 2;
//         cnrZ = cenZ - lenZ / 2;
    }
    else{
        printUsage(argv[0]);
        return;
    }

    console::parse_argument(argc, argv, "-coo_s", cooScale);
    console::parse_argument(argc, argv, "-ptsz", ptsz);
    console::parse_argument(argc, argv, "-rotStep", rotStep_);
    //תΪ����ֵ:
    rotStep_ = M_PI * rotStep_ / 180;
    console::parse_argument(argc, argv, "-tranStep", tranStep_);
    //תΪ��λ�ף�
    tranStep_ /= m2mm;

    string fnLoad;
    console::parse_argument(argc, argv, "-load", fnLoad);

    //---------------���������ļ��е���̬, �����������ļ�, ������ָ����ƽ�ƾ�ʧЧ
    cubePose_ = Matrix4f::Identity();
    if(loadPose(cubePose_, poseConfFn_)){//�����������ļ�
        cenX = cubePose_(0,3);
        cenY = cubePose_(1,3);
        cenZ = cubePose_(2,3);
    }


    typedef PointXYZRGB PtTYPE;
    typedef PointCloud<PtTYPE> CloudTYPE;

    //---------------������Ҫ���صĵ����ļ�:
    PointCloud<PointXYZ>::Ptr cloud_loaded_ptr(new PointCloud<PointXYZ>);
    if(boost::ends_with(fnLoad, ".pcd")){
        io::loadPCDFile(fnLoad, *cloud_loaded_ptr);
    }
    else if(boost::ends_with(fnLoad, ".ply")){
        io::loadPLYFile(fnLoad, *cloud_loaded_ptr);
    }
    
    //---------------�������������:
    //��ʽԭʼcube, ������ԭ��:
    CloudTYPE::Ptr cloud_ptr(new CloudTYPE);
    //ʩ�ӱ任��cube, ������cenXYZ:
    CloudTYPE::Ptr cloud_ptr_trans(new CloudTYPE);

    //1. �ߴ�Լ30cm, ���ܶ� 50����; ����ֱ��ÿ�� 5mm һ����?
    //2. ע�����Ͻ��� (-x, +y, +z), ����������������Զ��
    //3. ��z�ṹ��, ����������, ���ĸ�����
    float pt_step = 5; //mm
    console::parse_argument(argc, argv, "-pt_step", pt_step); //�����в������� mm
    pt_step /= m2mm;

    //�����洢��������Ӽ�������
    vector<int> idxVec;
    int idx=0; //ȫ���±�

    //cnrX = cnrY = cnrZ = 0;
    //��������:
    for(float z = -lenZ/2; z <= lenZ/2; z+=lenZ){ //����z, ����2���ᵼ�¸�����
        //���������, ���⸡������ĩ����߲���ȷ. �˴��� 4*2��
        for(float x = -lenX/2; x <= lenX/2; x+=lenX){//��X
            for(float y = -lenY/2; y <= lenY/2; y+=pt_step){//��Y��
                PtTYPE pt(0,0,255); //blue
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cloud_ptr->points.push_back(pt);
                idxVec.push_back(idx);

                idx++;
            }
        }
        for(float y = -lenY/2; y <= lenY/2; y+=lenY){//��Y
            for(float x = -lenX/2; x <= lenX/2; x+=pt_step){//��X��
                PtTYPE pt(0,0,255); //blue
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cloud_ptr->points.push_back(pt);
                idxVec.push_back(idx);

                idx++;
            }
        }


        //������ƽ��:
        for(float x = -lenX/2; x <= lenX/2; x+=pt_step){
            for(float y = -lenY/2; y <= lenY/2; y+=pt_step){
                //PtTYPE pt(x,y,z); //֮ǰ PointXYZ
                PtTYPE pt(0,0,255); //blue
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cloud_ptr->points.push_back(pt);

                idx++;
            }
        }
    }

    //����������+ʣ�����:
    for(float x = -lenX/2; x <= lenX/2; x+=lenX){//����x, ��������
        //���滹ʣ�������:
        for(float y = -lenY/2; y <= lenY/2; y+=lenY){//��Y
            for(float z = -lenZ/2; z <= lenZ/2; z+=pt_step){//��Z��
                PtTYPE pt(255,0,0); //red
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cloud_ptr->points.push_back(pt);
                idxVec.push_back(idx);

                idx++;
            }
        }

        //������������:
        for(float z = -lenZ/2+pt_step; z <= lenZ/2; z+=pt_step){
            for(float y = -lenY/2; y <= lenY/2; y+=pt_step){
                //PtTYPE pt(x,y,z); //֮ǰ PointXYZ
                PtTYPE pt(255,0,0); //red
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cloud_ptr->points.push_back(pt);

                idx++;
            }
        }
    }

    //����������
    for(float y = -lenY/2; y <= lenY/2; y+=lenY){//����y, ��������
        for(float x = -lenX/2+pt_step; x <= lenX/2; x+=pt_step){
            for(float z = -lenZ/2+pt_step; z <= lenZ/2; z+=pt_step){
                //PtTYPE pt(x,y,z); //֮ǰ PointXYZ
                PtTYPE pt(0,255,0); //green
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cloud_ptr->points.push_back(pt);

                idx++;
            }
        }
    }

    //---------------���ӻ���ʾ:
    PCLVisualizer viewer;
    //viewer.setBackgroundColor(0,0,0); //Ĭ��

    //1. Ԥ������ʵ���ƣ����Ա�
    int vp1;
    viewer.createViewPort(0, 0, 0.5, 1, vp1);
    viewer.addPointCloud(cloud_loaded_ptr, "cloud_loaded", vp1);
    
    //2. ���ɵ�������ƣ��Ѿ��任��
    //���� handler:
    //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_ptr);
    //viewer->addPointCloud<pcl::PointXYZRGB> (cloud_ptr, rgb, "cube");
    const char *zcCubeId = "zcCube";
    //viewer.addPointCloud(cloud_ptr, zcCubeId);
    //��Ϊ���� R,t ֮��ĵ��ƣ�
    transformPointCloud(*cloud_ptr, *cloud_ptr_trans, cubePose_);
    viewer.addPointCloud(cloud_ptr_trans, zcCubeId, vp1);
    viewer.setPointCloudRenderingProperties(PCL_VISUALIZER_POINT_SIZE, ptsz, zcCubeId);

    //3. ��ȡ�����Ӽ����
    CloudTYPE::Ptr cloudEdgePtr(new CloudTYPE);
    IndicesPtr indPtr = boost::make_shared<vector<int>>(idxVec);
    //     PointIndices indices;
    //     indices.indices = idxVec;
    ExtractIndices<PtTYPE> extractInd;
    extractInd.setInputCloud(cloud_ptr_trans);
    extractInd.setIndices(indPtr);
    extractInd.filter(*cloudEdgePtr);
    int vp2;
    viewer.createViewPort(0.5, 0, 1, 1, vp2);
    const char *edgeCloudId = "edgeCloud";
    viewer.addPointCloud(cloudEdgePtr, edgeCloudId, vp2);

    viewer.addCoordinateSystem(cooScale);
    //viewer.addCoordinateSystem(cooScale, cubePose_); //���������R,t������ʲô��? //��� ������
    //viewer.addCoordinateSystem(cooScale, cubePose_, "1");
    zcAddCoordSystem(viewer, 1, cubePose_); //��
    viewer.initCameraParameters();

    viewer.registerKeyboardCallback(kbCallback);
    
    while(!viewer.wasStopped()){
        viewer.spinOnce();
        //viewer.getCameraParameters(argc, argv);
        Affine3f pose = viewer.getViewerPose();
        Quaternionf qrot(pose.rotation());
        Vector3f t3(pose.translation());
        Vector4f t4(t3[0], t3[1], t3[2], 0);
        if(isSavePCD_){
            string fn = "zcCube";
            io::savePCDFileBinary(fn+".pcd", *cloud_ptr_trans);
            io::savePLYFileASCII(fn+".ply", *cloud_ptr_trans);

            //io::savePLYFile(fn+"2.ply", *cloud_ptr, t4, qrot, false, true); //��, PointCloud 
            //sensor_msgs::PointCloud2 pc2;
            //pcl::toROSMsg(*cloud_ptr, pc2);
            //io::savePLYFile(fn+"2.ply", pc2, t4, qrot, false, true); //plyĩβ����һ��, ����meshlab��ʾû����
            cout<<"write: "+fn+" finished."<<endl;

            //---------------����������ļ�, Ҫ�ڼ��� ctrl+s ������ű���
            saveIdxFile(edgeIdxConfFn, idxVec);

            isSavePCD_ = false;
        }
        if(isAddViewerCoo_){
            cout<<pose.matrix()<<endl;
            //pose.linear() = Matrix3f::Identity(); //ԭʼ����ϵɶ��? ��Ĭ��������һ������
            Matrix3f mt; //test, ����xy��ת����, z��ת�������, addCoordinateSystem ��bug?
            mt = AngleAxisf(0.5*M_PI, Vector3f::UnitZ());
            cout<<"mt:\n"<<mt<<endl;
            mt = AngleAxisf(-0.5*M_PI, Vector3f::UnitX());
            cout<<"mt:\n"<<mt<<endl;
            mt = AngleAxisf(0.5*M_PI, Vector3f::UnitY());
            cout<<"mt:\n"<<mt<<endl;
//             mt = AngleAxisf(-0.5*M_PI, Vector3f::UnitX()) * AngleAxisf(0.5*M_PI, Vector3f::UnitZ());
//             cout<<"mt:\n"<<mt<<endl;
            //pose.linear() = mt;

            //viewer.addCoordinateSystem(cooScale, pose);

            Affine3f good_pose = ::getViewerPose(viewer);
            viewer.addCoordinateSystem(cooScale, good_pose);
            zcAddCoordSystem(viewer, 1, good_pose); //���ڶԱ��ж��Ƿ� addCoordinateSystem ���� bug //2016-6-14 17:01:11

            static int cubeCnt = 0;
            //viewer.addCube(t3, qrot, lenX, lenY, lenZ, "cube-"+to_string((long long)cubeCnt));
            viewer.addCube(pose.translation(), Quaternionf(pose.rotation()), lenX, lenY, lenZ, "cube-"+to_string((long long)cubeCnt));
            cubeCnt++;
            isAddViewerCoo_ = false;
        }//if(isAddViewerCoo_)
        if(isTransformCube_){
            isTransformCube_ = false;
            //cubePose_.translation()<<cenX,cenY,cenZ; //���������ļ���, ����
            transformPointCloud(*cloud_ptr, *cloud_ptr_trans, cubePose_);
//             viewer.removePointCloud(zcCubeId);
//             viewer.addPointCloud(cloud_ptr, zcCubeId);
            viewer.updatePointCloud(cloud_ptr_trans, zcCubeId);

            //Ҳ������ȡ��ߵ����Ӽ���
            clock_t begt = clock();
            extractInd.filter(*cloudEdgePtr);
            cout<<"extractInd.filter, time: "<<clock()-begt<<endl;

            viewer.updatePointCloud(cloudEdgePtr, edgeCloudId);

        }
        if(isUpdateDepth_){
            isUpdateDepth_ = false;

            //���ͼ��ʾ
            PointCloud<PointXYZ> cloud_xyz;
            copyPointCloud(*cloud_ptr_trans, cloud_xyz); //xyzrgb->xyz
            //Affine3f pose_viewer = Affine3f::Identity(); //���� �ӵ���̬, 
            pose_viewer_ = ::getViewerPose(viewer);
            cout<<"pose_viewer_:\n"<<pose_viewer_.matrix()<<endl;

            cloud2depthCPU(cloud_xyz, pose_viewer_, intr, 640, 480, depFromCloud_);//��

            cv::Mat dfc8u(depFromCloud_.rows, depFromCloud_.cols, CV_8UC1);
            double dmin, dmax;
            minMaxLoc(depFromCloud_, &dmin, &dmax);
            depFromCloud_.convertTo(dfc8u, CV_8UC1, 255./(dmax-dmin), -dmin*255./(dmax-dmin));
            cv::imshow("dfc8u", dfc8u);
            //cv::waitKey(0);
        }
        if(isSaveDepth_){
            isSaveDepth_ = false;

            //���ͼ�洢
            char buf[111];
            sprintf (buf, "./synDepth-%04d.png", (int)dmapCnt_);
            imwrite(buf, depFromCloud_);

            //��̬�ļ��洢
            if(0 == dmapCnt_){ //���� ctrl+x ���ú�ĵ�һ��
                fout_camPose_.close();
                fout_camPose_.open("synDepthPose.csv", ios::out);
                pose_f0_ = pose_viewer_; //δ���õõ�
            }

            processPose(fout_camPose_, pose_viewer_);

            dmapCnt_++;
        }
    }//while
    //viewer.spin();
}//main
