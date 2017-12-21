#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <boost/format.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#if CV_MAJOR_VERSION == 3
#include <opencv2/rgbd.hpp>
namespace cv {
using namespace cv::rgbd;
}
#else
#include <opencv2/objdetect/objdetect.hpp>
#endif

#include <object_recognition_renderer/utils.h>
#include <object_recognition_renderer/renderer3d.h>

#include <opencv2/highgui/highgui.hpp>


#ifdef WIN32
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif
#include <stdint.h>
#include <string>
#define MAX_PATH_LEN 256

#ifdef WIN32
#define ACCESS(fileName,accessMode) _access(fileName,accessMode)
#define MKDIR(path) _mkdir(path)
#else
#define ACCESS(fileName,accessMode) access(fileName,accessMode)
#define MKDIR(path) mkdir(path,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif

#include <model.h>
#include "png++/png.hpp"

// 从左到右依次判断文件夹是否存在,不存在就创建
// example: /home/root/mkdir/1/2/3/4/
// 注意:最后一个如果是文件夹的话,需要加上 '\' 或者 '/'
int32_t createDirectory(const std::string &directoryPath)
{
    uint32_t dirPathLen = directoryPath.length();
    if (dirPathLen > MAX_PATH_LEN)
    {
        return -1;
    }
    char tmpDirPath[MAX_PATH_LEN] = { 0 };
    for (uint32_t i = 0; i < dirPathLen; ++i)
    {
        tmpDirPath[i] = directoryPath[i];
        if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/')
        {
            if (ACCESS(tmpDirPath, 0) != 0)
            {
                int32_t ret = MKDIR(tmpDirPath);
                if (ret != 0)
                {
                    return ret;
                }
            }
        }
    }
    return 0;
}

/**
 * @brief Reads a point cloud from a file that includes vertex positions (3 values), normals (3 values), and color information (3 values).
 *
 * The first line in the file is ignored.
 * Color information is ignored.
 * Only each 20th line of the file is read to decrease point cloud size.
 * The point cloud will be scaled by the given scale factor.
 * The point cloud will be translated such that its bounding box is centered at zero.
 *
 * @param path Point cloud file name including path.
 * @param scale Factor by which to scale the point cloud (e.g. for conversion of meters to millimeters)
 * @param points Output parameter. Point cloud.
 * @param normals Output parameter. Point cloud normals.
 * @return void
 */
//zyk info (centerx,centery,centerz,maxx,maxy,maxz,minx,miny,minz)
void getCloudInfo(std::string path, std::vector<double>& info)
{
    double maxX, maxY, minX, minY, maxZ, minZ;
    std::ifstream myfile(path.c_str());

    double x, y, z, nx, ny, nz, r, g, b;
    std::string firstLine;

    if (myfile.is_open())
    {
    // remove first line
    std::getline(myfile, firstLine);

    bool first = true;
    int i = 0;

    while(myfile >> x >> y >> z >> nx >> ny >> nz >> r >> g >> b)
    {
        if(i%10==0)
        {
        //To rotations by 90 degrees; (difference in coordinate system definition)
//        std::swap(x, z);
//        x *= -1;

//        std::swap(x, y);
//        x *= -1;

//        std::swap(nx, nz);
//        nx *= -1;

//        std::swap(nx, ny);
//        nx *= -1;

        if(first)
        {
            maxX = x; minX = x; maxY = y; minY = y; maxZ = z; minZ = z;
        }

        first = false;

        maxX = std::max(maxX, x); minX = std::min(minX, x);
        maxY = std::max(maxY, y); minY = std::min(minY, y);
        maxZ = std::max(maxZ, z); minZ = std::min(minZ, z);

//        points.push_back(cv::Point3d(scale*x,scale*y, scale*z));
//        normals.push_back(cv::Point3d(nx, ny, nz));
        }
        i++;
    }

    myfile.close();
    }
    else std::cout << "Unable to open point cloud file!";

//    cv::Point3d center(scale * (minX + maxX) / 2.0, scale * (minY + maxY) / 2.0, scale * (minZ + maxZ) / 2.0);

//    for(int i = 0; i < points.size(); i++)
//    {
//    points[i] = points[i] - center;
//    }
    info.clear();
    info.push_back(maxX);
    info.push_back(maxY);
    info.push_back(maxZ);
    info.push_back(minX);
    info.push_back(minY);
    info.push_back(minZ);
}
//accv_path : accv root path
//save_pat: save root path
//idx: index of rgbd
//cx,cy,cz: center of original .xyz point cloud
//ex,ey,ez: extent precalculated according to this paper
//return transformation 4x4
cv::Mat createInfoSingleStep(std::string accv_path,std::string save_path, std::string name, int idx, double cx,double cy, double cz, double ex,double ey,double ez){
    //get dataset name
    std::string rot_file=accv_path+"data/rot";
    std::string tra_file=accv_path+"data/tra";
    std::stringstream s;
    s<<idx<<".rot";
    rot_file+=s.str();
    s.str("");
    s<<idx<<".tra";
    tra_file+=s.str();
    std::ifstream rotf(rot_file.c_str());
    std::ifstream traf(tra_file.c_str());
    s.str("");
    s<<"info/info_"<<std::setfill('0')<<std::setw(5)<<idx<<".txt";
    //std::cout<<s.str()<<std::endl;
    std::ofstream infof((save_path+s.str()).c_str());
    cv::Mat rot = cv::Mat_<float>(3,3);
    cv::Mat tra = cv::Mat_<float>(3,1);
    if(rotf.is_open()){
        int row,col;
        rotf>>row>>col>>rot.at<float>(0,0)>>rot.at<float>(0,1)>>rot.at<float>(0,2)>>rot.at<float>(1,0)>>rot.at<float>(1,1)>>rot.at<float>(1,2)>>rot.at<float>(2,0)>>rot.at<float>(2,1)>>rot.at<float>(2,2);
        rotf.close();
        //std::cout<<"rot norm: "<<cv::determinant(rot)<<std::endl;
    }
    if(traf.is_open()){
        int row,col;
        traf>>row>>col>>tra.at<float>(0,0)>>tra.at<float>(1,0)>>tra.at<float>(2,0);
        traf.close();
        //std::cout<<tra<<std::endl;
    }
    cv::Mat test=cv::Mat_<float>(3,3);
    rot.col(1).copyTo(test.col(0));
    rot.col(2).copyTo(test.col(1));
    rot.col(0).copyTo(test.col(2));
    test.at<float>(0,0)=-test.at<float>(0,0);
    test.at<float>(0,1)=-test.at<float>(0,1);
    test.at<float>(1,2)=-test.at<float>(1,2);
    test.at<float>(2,2)=-test.at<float>(2,2);
    //std::cout<<test<<std::endl;


    //std::cout<<ex<<" "<<ey<<" "<<ez<<" "<<std::endl;

    cv::Mat pc=cv::Mat_<float>(3,1);
    pc.at<float>(0,0)=cx;
    pc.at<float>(1,0)=cy;
    pc.at<float>(2,0)=cz;


    cv::Mat c2=rot*pc+tra;
    c2.at<float>(0,0)*=0.01;
    c2.at<float>(1,0) *=-0.01;
    c2.at<float>(2,0) *=-0.01;
    //std::cout<<c2<<std::endl;

    std::swap(ex,ez);
    std::swap(ex,ey);
    ex*=0.00105;
    ey*=0.00105;
    ez*=0.00105;

    if(infof.is_open())
    {
        infof<<"image size"<<std::endl;
        infof<<"640 480"<<std::endl;
        infof<<name<<"ACCV"<<std::endl;
        infof<<"rotation:"<<std::endl;
        infof<<test.at<float>(0,0)<<" "<<test.at<float>(0,1)<<" "<<test.at<float>(0,2)<<std::endl<<test.at<float>(1,0)<<" "<<test.at<float>(1,1)<<" "<<test.at<float>(1,2)<<std::endl<<test.at<float>(2,0)<<" "<<test.at<float>(2,1)<<" "<<test.at<float>(2,2)<<std::endl;
        infof<<"center:"<<std::endl;
        infof<<c2.at<float>(0,0)<<" "<<c2.at<float>(1,0)<<" "<<c2.at<float>(2,0)<<std::endl;
        infof<<"extent:"<<std::endl;
        infof<<ex<<" "<<ey<<" "<<ez<<std::endl;
        infof.close();
    }
    cv::Mat a=cv::Mat::eye(4,4,CV_32F);

    test.copyTo(a(cv::Rect(0,0,3,3)));
    c2*=1000;
    c2.copyTo(a(cv::Rect(3,0,1,3)));
    //std::cout<<a<<std::endl;
    return a;
}


void get_bounding_box_for_node (const aiScene* scene,const  aiNode* nd,
    aiVector3D* min,
    aiVector3D* max,
    aiMatrix4x4* trafo
){
    aiMatrix4x4 prev;
    unsigned int n = 0, t;

    prev = *trafo;
    aiMultiplyMatrix4(trafo,&nd->mTransformation);

    for (; n < nd->mNumMeshes; ++n) {
        const  aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
        for (t = 0; t < mesh->mNumVertices; ++t) {

             aiVector3D tmp = mesh->mVertices[t];
            aiTransformVecByMatrix4(&tmp,trafo);

            min->x = aisgl_min(min->x,tmp.x);
            min->y = aisgl_min(min->y,tmp.y);
            min->z = aisgl_min(min->z,tmp.z);

            max->x = aisgl_max(max->x,tmp.x);
            max->y = aisgl_max(max->y,tmp.y);
            max->z = aisgl_max(max->z,tmp.z);
        }
    }

    for (n = 0; n < nd->mNumChildren; ++n) {
        get_bounding_box_for_node(scene,nd->mChildren[n],min,max,trafo);
    }
    *trafo = prev;
}

/* ---------------------------------------------------------------------------- */
void get_bounding_box (const aiScene*scene, aiVector3D* min,  aiVector3D* max)
{
    aiMatrix4x4 trafo;
    aiIdentityMatrix4(&trafo);

    min->x = min->y = min->z =  1e10f;
    max->x = max->y = max->z = -1e10f;
    get_bounding_box_for_node(scene,scene->mRootNode,min,max,&trafo);
}

void prepare_Renderer(int width, int height, float fx,float fy, float px, float py, Renderer3d & renderer, aiVector3D& center, aiVector3D& extent){

    //renderer = Renderer3d(mesh_path);
    renderer.set_parameters(width, height, fx, fy, px, py,  0.1, 2000);
    // get original model
    const struct aiScene* ps=renderer.getModel()->getScene();
    aiVector3D min_,max_;
    get_bounding_box(ps,&min_,&max_);
    center = min_+max_;
    extent=max_-min_;
    center/=2.0;

    //change the coordinate
    for (unsigned int i=0; i<ps->mNumMeshes; i++){
        for (unsigned int j=0; j<ps->mMeshes[i]->mNumVertices; j++){
            double x,y,z,nx,ny,nz;
            x=ps->mMeshes[i]->mVertices[j].x;
            y=ps->mMeshes[i]->mVertices[j].y;
            z=ps->mMeshes[i]->mVertices[j].z;
            nx=ps->mMeshes[i]->mNormals[j].x;
            ny=ps->mMeshes[i]->mNormals[j].y;
            nz=ps->mMeshes[i]->mNormals[j].z;

            std::swap(x, z);
            x *= -1;

            std::swap(x, y);
            x *= -1;

            std::swap(nx, nz);
            nx *= -1;

            std::swap(nx, ny);
            nx *= -1;

            ps->mMeshes[i]->mVertices[j].x=x;
            ps->mMeshes[i]->mVertices[j].y=y;
            ps->mMeshes[i]->mVertices[j].z=z;

            ps->mMeshes[i]->mNormals[j].x=nx;
            ps->mMeshes[i]->mNormals[j].y=ny;
            ps->mMeshes[i]->mNormals[j].z=nz;

        }
    }

    //calculate new center
    double tmp1=center[0],tmp2=center[1],tmp3=center[2];
    std::swap(tmp1, tmp3);
    tmp1 *= -1;

    std::swap(tmp1, tmp2);
    tmp1 *= -1;

    aiVector3D newCenter(tmp1,tmp2,tmp3);
    for (unsigned int i=0; i<ps->mNumMeshes; i++){
        for (unsigned int j=0; j<ps->mMeshes[i]->mNumVertices; j++){
            ps->mMeshes[i]->mVertices[j]-=newCenter;
        }
    }
}


IplImage * loadHintereDepth(std::string a_name)
{
    std::ifstream l_file(a_name.c_str(), std::ofstream::in | std::ofstream::binary);

    if (l_file.fail() == true)
    {
        printf("cv_load_depth: could not open file for writing!\n");
        return NULL;
    }
    int l_row;
    int l_col;

    l_file.read((char*)&l_row, sizeof(l_row));
    l_file.read((char*)&l_col, sizeof(l_col));

    IplImage * lp_image = cvCreateImage(cvSize(l_col, l_row), IPL_DEPTH_16U, 1);

    for (int l_r = 0; l_r < l_row; ++l_r)
    {
        for (int l_c = 0; l_c < l_col; ++l_c)
        {
            l_file.read((char*)&CV_IMAGE_ELEM(lp_image, unsigned short, l_r, l_c), sizeof(unsigned short));
        }
    }
    l_file.close();

    return lp_image;
}

void createObjSegColorDepthSingleStep(int width, int height,std::string accv_data_path, std::string save_data_folder, std::string name, int idx, Renderer3d& renderer,cv::Mat& Rt, cv::Mat& K){
    cv::Mat image, depth, mask;
    cv::Rect rect;
    rect.width=width;
    rect.height=height;

    renderer.setModelRt(Rt.t());

    renderer.renderDepthSimple(depth, mask, rect);
    cv::flip(mask,mask,0);
    cv::flip(depth,depth,0);
    //cv::imshow("b",mask);
    //cv::waitKey(0);

    //load hintere depth
    std::stringstream s;
    s<<idx;
    IplImage* ipdpt = loadHintereDepth(accv_data_path+"data/depth"+s.str()+".dpt");
    cv::Mat_<ushort> dpt=cv::cvarrToMat(ipdpt);
    cv::Mat_<cv::Vec3b> color = cv::imread(accv_data_path+"data/color"+s.str()+".jpg");
    //create obj
    //cv::Mat obj_coord = cv::Mat::zeros(width,height,CV_16UC3);
    png::image<png::basic_rgb_pixel<unsigned short> > imgPng(width,height);
    png::image<png::basic_rgb_pixel<unsigned char> > mask_out(width,height);
    png::image<png::basic_rgb_pixel<unsigned char> > color_out(width,height);
    png::image<unsigned short> depth_out(width,height);
    for(int i=0;i<height;++i)
        for(int j=0;j<width;++j)
        {
            if(mask.at<uchar>(i,j)!=0){

                float d=depth.at<ushort>(i,j);
                cv::Vec3f pt(j,height-i,1);
                cv::Mat v = K.inv()*cv::Mat(pt);
                v*=d;
                v.at<float>(2,0)=-v.at<float>(2,0);
                //std::cout<<v<<std::endl;
                cv::Mat coord = v-Rt(cv::Range(0,3),cv::Range(3,4));
                coord=Rt(cv::Range(0,3),cv::Range(0,3)).t()*coord;
                //std::cout<<coord<<std::endl;
                imgPng.set_pixel(j,i, png::basic_rgb_pixel<unsigned short>(coord.at<float>(0,0),coord.at<float>(1,0),coord.at<float>(2,0)));
                mask_out.set_pixel(j,i, png::basic_rgb_pixel<uchar>(255,255,255));
            }
            color_out.set_pixel(j,i,png::basic_rgb_pixel<uchar>(color(i,j)[2],color(i,j)[1],color(i,j)[0]));
            depth_out.set_pixel(j,i,dpt(i,j));

        }
    s.str("");
    s<<save_data_folder<<"obj/obj_"<<std::setfill('0')<<std::setw(5)<<idx<<".png";
    imgPng.write(s.str());
    s.str("");
    s<<save_data_folder<<"seg/seg_"<<std::setfill('0')<<std::setw(5)<<idx<<".png";
    mask_out.write(s.str());
    s.str("");
    s<<save_data_folder<<"rgb_noseg/color_"<<std::setfill('0')<<std::setw(5)<<idx<<".png";
    color_out.write(s.str());
    s.str("");
    s<<save_data_folder<<"depth_noseg/depth_"<<std::setfill('0')<<std::setw(5)<<idx<<".png";
    depth_out.write(s.str());
}



int main(int argc, char **argv) {
    // Define the display
    size_t width = 640, height = 480;
    //render parameters
    //float render_near=0.1, render_far=2000.0;
    float renderer_focal_length_x=572.41140;
    float renderer_focal_length_y=573.57043;
    float px=325.26110,py=242.04899;
    cv::Mat K=cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0)=renderer_focal_length_x;
    K.at<float>(1,1)=renderer_focal_length_y;
    K.at<float>(0,2)=px;
    //@todo ATTENTION,THIS HAS TO BE TRANSFORMED!!!!!! STILL TO BE QUESTIONED!
    K.at<float>(1,2)=height-py;

    std::string obj_name="duck";
    // the model name can be specified on the command line.
    std::string accv_data_folder="/home/zyk/git/rgbd/ACCV3D_dataset_working_dir/ACCV3D/"+obj_name+"/";
    std::string test_data_folder="/home/zyk/git/rgbd/ACCV3D_dataset_working_dir/learning_dataset/test/Accv_" + obj_name+"/";
    std::string train_data_folder="/home/zyk/git/rgbd/ACCV3D_dataset_working_dir/learning_dataset/training/Accv_"+obj_name + "/";
    createDirectory(test_data_folder+"info/");
    createDirectory(test_data_folder+"obj/");
    createDirectory(test_data_folder+"seg/");
    createDirectory(test_data_folder+"rgb_noseg/");
    createDirectory(test_data_folder+"depth_noseg/");

    createDirectory(train_data_folder+"info/");
    createDirectory(train_data_folder+"obj/");
    createDirectory(train_data_folder+"seg/");
    createDirectory(train_data_folder+"rgb_noseg/");
    createDirectory(train_data_folder+"depth_noseg/");


    //prepare renderer
    Renderer3d renderer(accv_data_folder+"mesh.ply");
    aiVector3D center,extent;
    prepare_Renderer(width,height,renderer_focal_length_x, renderer_focal_length_y, px, py, renderer,center,extent);
    for(int cnt=0;cnt<1179;++cnt){
        if(cnt%3){
            cv::Mat Rt=createInfoSingleStep(accv_data_folder,test_data_folder,obj_name,cnt,0.1*center[0],0.1*center[1],0.1*center[2],extent[0],extent[1],extent[2]);
            createObjSegColorDepthSingleStep(width,height,accv_data_folder,test_data_folder,obj_name,cnt,renderer,Rt,K);
        }
        else
        {
            cv::Mat Rt=createInfoSingleStep(accv_data_folder,train_data_folder,obj_name,cnt,0.1*center[0],0.1*center[1],0.1*center[2],extent[0],extent[1],extent[2]);
            createObjSegColorDepthSingleStep(width,height,accv_data_folder,train_data_folder,obj_name,cnt,renderer,Rt,K);
        }

    }


    return 0;
}
