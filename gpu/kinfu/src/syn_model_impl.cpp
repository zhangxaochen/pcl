//@author zhangxaochen  //2016-4-5 04:17:49
//@brief ��Ӧ cu �к����İ�װ����, cloud2tsdf, etc


#include <iostream>
#include <fstream>
//#include <Eigen/Core>

#include <pcl/common/centroid.h>


#include "syn_model_impl.h" //Ǩ�Ƶ� zcUtility.h ��
#include "zcUtility.h"

//#include "cuda/device.hpp" //��, ��Ҫ�ڷ� cu �ļ�������

#define SZ_6DOF 6 //�ݶ��½�������������

namespace zc{

using namespace std;
// using namespace pcl;
// using namespace pcl::gpu;
using namespace pcl::device;
// using namespace cv;
// using namespace Eigen;

//+++++++++++++++�� cuda/device.hpp �������� utils, ����CPU��

float3 make_float3(float x, float y, float z){
    float3 res;
    res.x = x;
    res.y = y;
    res.z = z;
    return res;
}

float dot(const float3& v1, const float3& v2)
{
    return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
}

float3 operator* (const Mat33& m, const float3& vec)
{
//     float3 res;
//     res.x = dot (m.data[0], vec);
//     res.y = dot (m.data[1], vec);
//     res.z = dot (m.data[2], vec);
//     return res;
    return make_float3 (dot (m.data[0], vec), dot (m.data[1], vec), dot (m.data[2], vec));
}

float3&
    operator+=(float3& vec, const float& v)
{
    vec.x += v;  vec.y += v;  vec.z += v; return vec;
}

float3
    operator+(const float3& v1, const float3& v2)
{
    return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

float3&
    operator*=(float3& vec, const float& v)
{
    vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
}

float3
    operator-(const float3& v1, const float3& v2)
{
    return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

float3
    operator*(const float3& v1, const float& v)
{
    return make_float3(v1.x * v, v1.y * v, v1.z * v);
}


void cloud2tsdf(const PointCloud<PointXYZ> &cloud, int weight, TsdfVolume &tsdf_volume){

    int cols = cloud.size(), //unorganized cloud
        rows = 1;

    DeviceArray<PointXYZ> cloud_device;
    cloud_device.create(cols);
    cloud_device.upload(cloud.points);
    const float3& cell_size = device_cast<const float3>(tsdf_volume.getVoxelSize()); //voxel ����ߴ�
    Eigen::Vector4f centroid; //��������
    compute3DCentroid(cloud, centroid);
    cout<<"cloud2tsdf(), centroid: "<<centroid<<endl;
    float3 pcen;
    pcen.x = centroid.x();
    pcen.y = centroid.y();
    pcen.z = centroid.z();

    //int weight = 1; //����ݶ���, �ο� MAX_WEIGHT=128 @tsdf_volume.cu
                      //Ȩ�ظ�������
    cloud2tsdf((DeviceArray<float4>&)cloud_device, pcen, cell_size, tsdf_volume.getTsdfTruncDist(), weight, tsdf_volume.data());

    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
    
//     int tsdfKrnlCntHost;
//     cudaMemcpyFromSymbol(&tsdfKrnlCntHost, &tsdfKrnlCnt, sizeof(int), 0);
}//cloud2tsdf

void cloud2depth(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, DeviceArray2D<ushort> &depOutDevice){
    depOutDevice.create(imHeight, imWidth);
    int cols = cloud.size(), //cloud �����Ƿ� organized, һ����Ϊ unorganized
        rows = 1;

    DeviceArray<PointXYZ> cloud_device;
    cloud_device.create(cols);
    cloud_device.upload(cloud.points);

    Matrix3frm Rrm = pose.linear();
    Vector3f t = pose.translation();
    Matrix3frm Rrm_inv = Rrm.inverse(); //inv �Ƿ񻹱����� Matrix3frm��
    const Mat33 &device_Rrm_inv = device_cast<const Mat33>(Rrm_inv); //ע�⴫����������̬������ת
    const float3 &device_t = device_cast<const float3>(t);
    cloud2depth((DeviceArray<float4>&)cloud_device, device_Rrm_inv, device_t, intr, depOutDevice);
}//cloud2depth

void cloud2depth(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, Mat &depOut){
    DeviceArray2D<ushort> depOut_device;
#if 0 //����, Ǩ�Ƶ����ذ� cloud2depth, �������
    //GPU:
    depOut_device.create(imHeight, imWidth);
    //depOut_device.upload(depOut.data, ); //��, ֻӦdownload!

    int cols = cloud.size(), //cloud �����Ƿ� organized, һ����Ϊ unorganized
        rows = 1;

    DeviceArray<PointXYZ> cloud_device;
    cloud_device.create(cols);
    cloud_device.upload(cloud.points);

    Matrix3frm Rrm = pose.linear();
    Vector3f t = pose.translation();
    Matrix3frm Rrm_inv = Rrm.inverse(); //inv �Ƿ񻹱����� Matrix3frm��
    const Mat33 &device_Rrm_inv = device_cast<const Mat33>(Rrm_inv); //ע�⴫����������̬������ת
    const float3 &device_t = device_cast<const float3>(t);
    cloud2depth((DeviceArray<float4>&)cloud_device, device_Rrm_inv, device_t, intr, depOut_device);
#elif 1 //�������ذ�
    cloud2depth(cloud, pose, intr, imWidth, imHeight, depOut_device);
#endif

    //��������ͼ�����ڴ�: CPU
    depOut = Mat::zeros(imHeight, imWidth, CV_16UC1); //ȫ���0
    //depOut_device.download(depOut.data, depOut.cols * depOut.elemSize());
    depOut_device.download(depOut.data, depOut.cols * depOut.elemSize());
}//cloud2depth

void cloud2depthCPU(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, int imWidth, int imHeight, Mat &depOut){
    //��������ͼ�����ڴ�: CPU
    depOut = Mat::zeros(imHeight, imWidth, CV_16UC1);
    Matrix3frm Rrm = pose.linear();
    Vector3f t = pose.translation();
    Matrix3frm Rrm_inv = Rrm.inverse(); //inv �Ƿ񻹱����� Matrix3frm��
    const Mat33 &device_Rrm_inv = device_cast<const Mat33>(Rrm_inv); //ע�⴫����������̬������ת
    const float3 &device_t = device_cast<const float3>(t);

    int sz = cloud.size();
    for(int i = 0; i < sz; i++){
        //---------------���涼�Ǵ� cloud2depthKernel �������, GPU���� bug, ���� CPU �ϲ���:
        PointXYZ pt4 = cloud[i];
        float3 pi;
        pi.x = pt4.x; 
        pi.y = pt4.y;
        pi.z = pt4.z;

        float3 pi2cam = pi - device_t; //pi �� camera ԭ������
        float3 vcam = device_Rrm_inv * pi2cam; //pi world->camera

        int u = int((vcam.x * intr.fx) / vcam.z + intr.cx); //����ͼ������ (u,v), Ӧ���� m->pixel? ���Ǻ����أ�Ū����� ��: ȷʵ m, ������� (u, v) Ҳû��
        int v = int((vcam.y * intr.fy) / vcam.z + intr.cy);

        //������ͼ��Χ�ڣ�
        if(u < 0 || u > imWidth-1 ||
           v < 0 || v > imHeight-1)
           continue;

        const int MAX_DEPTH = 10000;
        float z = vcam.z * 1000;
        ushort oldz = depOut.at<ushort>(v, u);
        if(0 < z && z < MAX_DEPTH && //��z���ֵ����
           (0 == oldz || oldz > z))//�Ҵ�����δ����, ��z��������ֵС
           depOut.at<ushort>(v, u) = (ushort)z; //�˴����ص��� float->ushort �ض�
    }//for
}//cloud2depthCPU

void raycastCloudSubset(const PointCloud<PointXYZ> &cloud, const Affine3f &pose, const Intr &intr, const Mat &dmat, PointCloud<PointXYZ> &outCloud){
    int cols = cloud.size(), //cloud �����Ƿ� organized, һ����Ϊ unorganized
        rows = 1;

    DeviceArray<PointXYZ> cloud_device;
    cloud_device.create(cols);
    cloud_device.upload(cloud.points);

    Matrix3frm Rrm = pose.linear();
    Vector3f t = pose.translation();
    Matrix3frm Rrm_inv = Rrm.inverse(); //inv �Ƿ񻹱����� Matrix3frm��
    const Mat33 &device_Rrm_inv = device_cast<const Mat33>(Rrm_inv); //ע�⴫����������̬������ת
    const float3 &device_t = device_cast<const float3>(t);
    DeviceArray2D<ushort> dmatGpu;
    //dmatGpu.create(dmat.rows, dmat.cols);
    dmatGpu.upload(dmat.data, dmat.cols * sizeof(ushort), dmat.rows, dmat.cols);

    //vector<bool> outIdxVec(cols, false); //default: false //��, @duyu: vector<bool>��stl�в���bool, ����data[xx]������ֵ
    vector<int> outIdxVec(cols, 0); //default: 0
    DeviceArray<int> outIdxGpu;
    outIdxGpu.upload(outIdxVec);
    raycastCloudSubset((DeviceArray<float4>&)cloud_device, device_Rrm_inv, device_t, intr, dmatGpu, outIdxGpu);
    outIdxGpu.download(outIdxVec);

    //���������ĵ�, ��䵽�������:
    for(int i = 0; i < cols; i++){
        if(0 != outIdxVec[i]){
            outCloud.push_back(cloud[i]);
        }
    }
}//raycastCloudSubset

void align2dmapsGPU(const DepthMap &srcDmat, const Intr &intr, const DepthMap &dstDmat, const MapArr &dstGradu, const MapArr &dstGradv){
    int imWidth = srcDmat.cols(),
        imHeight = srcDmat.rows();

    MapArr vmap_g/*, vmap_c*/;
    DeviceArray2D<float4> vmap_g_aos;
    createVMap(intr, srcDmat, vmap_g); //srcDmat ת���������ϵ����, ͬʱ�ڴ˺�����Ҳ��Ϊ��������ϵ //vmap ����(m)�߶� @computeVmapKernel

    device::convert(vmap_g, vmap_g_aos);
    int elemNum = vmap_g_aos.rows() * vmap_g_aos.cols();

//     Mat33 rmat_g2c; //global2camera, �������� Tc
//     rmat_g2c.data[0].x = 1;
//     rmat_g2c.data[1].y = 1;
//     rmat_g2c.data[2].z = 1;
//     float3 tvec_g2c;
//     tvec_g2c.x = tvec_g2c.y = tvec_g2c.z = 0; //(R0,t0)=(Eye, 0)
    Matrix3frm Rrm_g2c = Matrix3f::Identity();
    Vector3f tvec_g2c(0,0,0);

    vector<float> errVec;
    for(size_t i = 0; i < 40; i++){
        cout<<"+++++++++++++++iter-cnt@align2dmapsGPU= "<<i<<endl;
        //transformVmap(vmap_g, rmat_g2c, tvec_g2c, vmap_c);
        
        Mat33&  device_Rrm_g2c = device_cast<Mat33> (Rrm_g2c);
        float3& device_tvec_g2c = device_cast<float3>(tvec_g2c);
        
        Matrix3frm Rrm_c2g = Rrm_g2c.inverse();
        Vector3f tvec_c2g = -Rrm_c2g * tvec_g2c;
        Mat33 &device_Rrm_c2g = device_cast<Mat33>(Rrm_c2g);
        float3 &device_tvec_c2g = device_cast<float3>(tvec_c2g);

        DepthMap srcDmatNew; //ÿ�ε�����Ͷ������һ�����ͼ
        cloud2depth(DeviceArray<float4>(vmap_g_aos.ptr(), elemNum), device_Rrm_g2c, device_tvec_c2g, intr, imWidth, imHeight, srcDmatNew);

        float err = twoDmatsMSE(srcDmatNew, dstDmat);
        cout<<"err: "<<err<<endl;
        errVec.push_back(err);

        float dposeBuf[SZ_6DOF]; //һ�ε������ delta pose 6dof
        clock_t begt = clock();
        align2dmapsOnce(srcDmatNew, intr, device_Rrm_c2g, device_tvec_c2g, dstDmat, dstGradu, dstGradv, dposeBuf);
        cout<<"alignOnce, time= "<<clock()-begt<<endl;

        //��ΪҪ�ݶ��½�������Ҫȡ����
        for(size_t i=0; i<SZ_6DOF; i++)
            dposeBuf[i] = -dposeBuf[i];
//         for(size_t i=0; i<2; i++) //�� rot(a, b, r) ���� //�����, ���� 2016-6-30 15:47:26
//             dposeBuf[i] *= err/10;

        float alpha = dposeBuf[0];
        float beta  = dposeBuf[1];
        float gamma = dposeBuf[2];
        Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
        Vector3f tinc(dposeBuf[3], dposeBuf[4], dposeBuf[5]);

        tvec_g2c = Rinc * tvec_g2c + tinc;
        //tvec_g2c += tinc; //������Ϊ Rinc ��������С, ��������ʵ�ʽ��Ҳûɶ��ͬ
        Rrm_g2c = Rinc * Rrm_g2c;
    }

    ofstream errFout("zcGdErr.csv", ios::app);
    cout<<"errVec=[";
    for(int i = 0; i < errVec.size()-1; i++){
        cout<<errVec[i]<<", ";
        errFout<<errVec[i]<<", ";
    }
    cout<<errVec.back()<<']'<<endl;
    errFout<<errVec.back()<<endl;
    errFout.close();
}//align2dmaps

typedef PointXYZ PtType;
typedef PointCloud<PointXYZ> CloudType;
//@brief �� cv::Mat ���� pcl::PointCloud ����
//@ref: http://stackoverflow.com/questions/32521043/how-to-convert-cvmat-to-pclpointcloud
//@param[in] dmat �洢���� ushort, ���׳߶����ֵ, 
//@return ����ֵ PointCloud �Ҵ洢�����׳߶ȵ���, �Ǳ�׼��λ
CloudType::Ptr cvMat2PointCloud(const Mat &dmat, const Intr &intr){
    CV_Assert(dmat.type() == CV_16UC1);

    int imWidth = dmat.cols,
        imHeight = dmat.rows;

    //�����صĵ���:
    CloudType::Ptr pCloud(new CloudType);
    pCloud->points.reserve(imWidth * imHeight); //ע��: �� resize, ��������� push_back, ���� "[]"

    float fx_inv = 1 / intr.fx,
        fy_inv = 1 / intr.fy;
    float mm2m = 0.001;

    ushort *pDat = (ushort*)dmat.data;
    for(int i = 0; i < imHeight; i++){
        for(int j = 0; j < imWidth; j++){
            ushort z = *pDat;
            PtType pt;
            pt.z = z * mm2m; //ת������(m)�߶�
            pt.x = pt.z * (j - intr.cx) * fx_inv;
            pt.y = pt.z * (i - intr.cy) * fy_inv;
            pCloud->push_back(pt);

            ++pDat;
        }
    }

    pCloud->width = imWidth; //��ʾ�������, ����������, ��Ϊ push_back �����趨
    pCloud->height = imHeight;

    return pCloud;
}//cvMat2PointCloud

void align2dmapsCPU(const Mat &srcDmat, const Intr &intr, const Mat &dstDmat, const Mat &dstGradu, const Mat &dstGradv){
    int imWidth = srcDmat.cols,
        imHeight = srcDmat.rows;

    float fu = intr.fx,
          fv = intr.fy,
          fu_inv = 1 / fu,
          fv_inv = 1 / fv,
          cu = intr.cx,
          cv = intr.cy;

    //src ���ɵ���, ���Ӧ���������ϵ��Ϊ��������ϵ:
    CloudType::Ptr srcCloud = cvMat2PointCloud(srcDmat, intr);

    Matrix3f R_g2c = Matrix3f::Identity(); //(R,t) ��ֵ ==(I, 0)
    Vector3f tvec_g2c(0,0,0);

    for(int iter = 0; iter < 20; iter++){
        cout<<"---------------iter-cnt@align2dmapsCPU== "<<iter<<endl;
        //�ݶ��½���һ�ε�����
        Affine3f pose; //�����̬, ע���� c2g
        pose.linear() = R_g2c.inverse();
        pose.translation() = -R_g2c.inverse() * tvec_g2c;
        cout<<"pose:=\n"<<pose.matrix()<<endl;

        Mat srcDmatNew; //src->cloud(ȫ��)->��ͶӰ���ͼ
        cloud2depthCPU(*srcCloud, pose, intr, imWidth, imHeight, srcDmatNew);
//         srcDmatNew = srcDmat.clone(); //���Ծ���ԭsrc, ���÷�ͶӰ����-ͶӰ���ͼ����, //�� zc-h(uv) ��Ϊ��, �������, ����ʵ��һ�����ݵ���û̫�����
//         srcDmatNew += 5; //�������ֵ���Ӧ�Z, 

        //�Ƶ� 
//         Mat sdn8u;
//         srcDmatNew.convertTo(sdn8u, CV_8UC1, 255/1e4);
//         Mat dd8u;
//         dstDmat.convertTo(dd8u, CV_8UC1, 255/1e4);
// 
//         imshow("srcDmatNew", srcDmatNew);
//         imshow("sdn8u", sdn8u);
//         imshow("dstDmat", dstDmat);
//         imshow("dd8u", dd8u);
//         waitKey();

//         imwrite("srcDmatNew.png", srcDmatNew);
//         imwrite("dstDmat.png", dstDmat);

        //���ƺ���: twoDmatsMSE
        Mat validMsk = (srcDmatNew != 0 & dstDmat != 0); //src, dst ����Ϊ�������
        int validCnt = countNonZero(validMsk); //��� gGdCnt, ֻ�ǲ�����ѭ�����ۼ�
        if(0 == validCnt)
            cout<<"ERROR: validCnt==0, �����ͼ�޽���"<<endl;
        Mat m12diff = abs(srcDmatNew - dstDmat); //src, dst ��Ч���صľ���ֵ��
        m12diff.setTo(0, validMsk == 0); //ȥ����Ч����
        float err = twoDmatsMSE(srcDmatNew, dstDmat);
        cout<<"err=="<<err<<endl
            <<"validCnt:="<<validCnt<<endl;

        float meanDiff = mean(m12diff, validMsk)[0];
        cout<<"meanDiff:= "<<meanDiff<<endl;
        double dmin, dmax;
        minMaxLoc(m12diff, &dmin, &dmax, 0, 0, validMsk);
        cout<<"dmin, dmax:= "<<dmin<<", "<<dmax<<endl;

        float dRtArr[SZ_6DOF] = {0}; //�洢sum(6dof����)
        //��ѭ����� align2dmapsKernel
//         ushort *pSrcDat = srcDmatNew.data,
//                *pDstDat = dstDmat.data; //��ֱ��, ���ﲻ��Ч��
        float sumM_oldM = 0; //һ�����Ա���
        for(int i = 0; i < imHeight; i++){
            //cout<<"i= "<<i<<endl;
            for(int j = 0; j < imWidth; j++){
                bool isValid = (validMsk.at<uchar>(i, j) != 0);
                if(!isValid)
                    continue;
//                 if(i==114)
//                     cout<<"j:= "<<j<<endl;
                int z_c = srcDmatNew.at<ushort>(i, j), //���׼�
                    huv = dstDmat.at<ushort>(i, j);
                int depthDiff = z_c - huv;
                if(abs(depthDiff) > 100){ //��ֵ��⣺��������Ȳ����ĳ��ֵ, ������. �ݶ�10cm
                    //printf("abs(depthDiff) > 100, == %d; (u,v)=(%d, %d)\n", depthDiff, j, i);
                    continue;
                }

                float su = dstGradu.at<float>(i, j),
                      sv = dstGradv.at<float>(i, j);

                float A = su * fu / z_c,
                      B = sv * fv / z_c,
                      K = (j - cu) * fu_inv,
                      L = (i - cv) * fv_inv,
                      //M = 1 + A * K + B * L; //�������Ϊ oldM
                      M = 1 + (su * (j - cu) + sv * (i - cv)) / z_c;

                float oldM = 1 + A * K + B * L;
                //cout <<"M-oldM:="<<M-oldM<<endl; //��С, ������ sum ����? �� //2016-6-26 20:35:34
                sumM_oldM += abs(M-oldM); //�ۼӽ��Ҳ�㹻С, 5w����, sum=1e-3 ����; ˵��oldM��ʽ��������Ҫ�����Դ!

                //����(u,v)��Ӧȫ������
                PtType ptSrc_g = srcCloud->at(j, i); //ע���� (j,i)˳��
                //ȫ�������תΪ���׳߶ȣ�
                //ptSrc_g *= 1000; 
                const int m2mm = 1000;
                ptSrc_g.x *= m2mm;
                ptSrc_g.y *= m2mm;
                ptSrc_g.z *= m2mm;

                float dAlpha = M * ptSrc_g.y - B *ptSrc_g.z,
                      dBeta = -M * ptSrc_g.x - A * ptSrc_g.z,
                      dGamma = A * ptSrc_g.y - B * ptSrc_g.x,
                      dTx = -A,
                      dTy = -B,
                      dTz = M;
                //��Ҫ������Ȳ�:
                dAlpha *= depthDiff;
                dBeta *= depthDiff;
                dGamma *= depthDiff;
                dTx *= depthDiff;
                dTy *= depthDiff;
                dTz *= depthDiff;

                dRtArr[0] += dAlpha;
                dRtArr[1] += dBeta;
                dRtArr[2] += dGamma;
                dRtArr[3] += dTx;
                dRtArr[4] += dTy;
                dRtArr[5] += dTz;
            }//for-j
        }//for-i
        cout<<"sumM_oldM:="<<sumM_oldM<<endl;

        //float gdStepSz = 1e-6; //��������: learn-rate, ����, ѧϰ��
        float gdStepSz[2] =
        {1e-8, 1e-4};
        for(size_t i=0; i<SZ_6DOF; i++){
            cout<<"dRtArr["<<i<<"]:"<<dRtArr[i];//<<endl;
            dRtArr[i] /= validCnt;
            //dRtArr[i] *= gdStepSz;
            dRtArr[i] *= gdStepSz[i / 3];
            cout<<",\t"<<dRtArr[i]<<endl;
        }

        //��ΪҪ�ݶ��½�������Ҫȡ����
        for(size_t i=0; i<SZ_6DOF; i++)
            dRtArr[i] = -dRtArr[i];

        Matrix3f Rinc = (Matrix3f)AngleAxisf(dRtArr[2], Vector3f::UnitZ()) * AngleAxisf(dRtArr[1], Vector3f::UnitY()) * AngleAxisf(dRtArr[0], Vector3f::UnitX());
        //Matrix3f Rinc = Matrix3f::Identity(); //����������Ҫ�����Դ�� a, b, gamma, ����ָ��I, ��ͼ������Ӱ��, ���� t �����Ч��
        Vector3f tinc(dRtArr[3], dRtArr[4], dRtArr[5]);

        tvec_g2c = Rinc * tvec_g2c + tinc;
        R_g2c = Rinc * R_g2c;
    }//for-iter

}//align2dmapsCPU

float twoDmatsMSE(Mat m1, Mat m2){
    //imwrite("m1.png", m1); //���� src ͶӰ���ͼ��
    //������, �������Թ۲���
    Mat m1_8u;
    m1.convertTo(m1_8u, CV_8UC1, 255/1e4);
    Mat m2_8u;
    m2.convertTo(m2_8u, CV_8UC1, 255/1e4);

    imshow("m1", m1);
    imshow("m1_8u", m1_8u);
    imshow("m2", m2);
    imshow("m2_8u", m2_8u);
    waitKey();

    Mat validMsk = (m1 != 0 & m2 != 0); //m1, m2 ����Ϊ�������
    int validCnt = countNonZero(validMsk);
    Mat m12diff;
    cv::absdiff(m1, m2, m12diff);
    m12diff.setTo(0, validMsk == 0); //ȥ����Ч����
    Mat m12sqr = m12diff.mul(m12diff);
    float sumOfSqr = cv::sum(m12sqr)[0];
    return sumOfSqr / validCnt;
}//twoDmatsMSE

//@brief testRmseTwoDmats �ĵ�Ԫ����
void testTwoDmatsMSE(){
    //python���Դ���:
    //a=ones((5,6))*3
    //b=ones((5,6))*5
    //a[0,:]=0
    //b[:,0]=2
    //msk=(a!=0)&(b!=0)
    //c=(a-b)[msk]
    //sum(square(c))/len(c)==3.5 #True

    int rr = 5,
        cc = 6;
    Mat m1(rr, cc, CV_16UC1, 3);
    Mat m2(rr, cc, CV_16UC1, 5);

    m1.row(0) = 0;
    m2.col(0) = 2;

    CV_Assert(twoDmatsMSE(m1, m2) == 3.5); //�Ѳ��� @OpencvCeshi.cpp //2016-6-20 22:11:48
    cout<<"---------------testTwoDmatsMSE OK---------------"<<endl;
}//testTwoDmatsMSE

float twoDmatsMSE(const DepthMap &m1_device, const DepthMap &m2_device){
    cv::Mat m1(m1_device.rows(), m1_device.cols(), CV_16UC1);
    m1_device.download(m1.data, m1_device.colsBytes());
    
    cv::Mat m2(m2_device.rows(), m2_device.cols(), CV_16UC1);
    m2_device.download(m2.data, m2_device.colsBytes());

    return twoDmatsMSE(m1, m2);
}//twoDmatsMSE

void foo_in_syn_cpp(){}

}//namespace zc
