#include <cstdio>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <list>
#include <queue>
#include <string>
#include <thread>
#include <mutex>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/outofcore/outofcore.h>
#include <pcl/outofcore/outofcore_impl.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <std_srvs/Trigger.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Header.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>

// #include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../../include/common.h"

// 定义全局变量
std::list<sensor_msgs::PointCloud2::ConstPtr> pointCloudBuf;
std::list<sensor_msgs::PointCloud2::ConstPtr>::iterator pointCloudIter;
std::list<sensor_msgs::Image::ConstPtr> imageBuf;
std::list<sensor_msgs::Image::ConstPtr>::iterator imageIter;
std::list<sensor_msgs::Imu::ConstPtr> imuBuf;
std::list<sensor_msgs::Imu::ConstPtr>::iterator imuIter;

std::mutex mBuf;
cv_bridge::CvImagePtr cv_cur_ptr; // 声明一个CvImage指针的实例
cv_bridge::CvImagePtr cv_last_ptr;
cv_bridge::CvImagePtr cv_next_ptr;
int image_use_count = 0;
int pointCloudCount = 0;

int threshold_lidar; // number of cloud point on the photo
string intrinsic_path, extrinsic_path;
float max_depth = 130;
float min_depth = 0.5;

cv::Mat cameraMatrix;
cv::Mat distCoeffs;
cv::Mat transformMatrix;
cv::Mat optimalNewCameraMatrix;

ros::Publisher pubRgb;
ros::Publisher pubDepth;
ros::Publisher pubProjectImage;

// 激光雷达点云回调函数
void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &pointCloudIn)
{
    // cout << "lidar callback" << endl;
    mBuf.lock();
    // std::cout << "pointCloudIn "<< pointCloudIn->header.stamp << std::endl;
    pointCloudBuf.push_back(pointCloudIn);
    mBuf.unlock();
}

// 相机图片回调函数
void imageCallback(const sensor_msgs::Image::ConstPtr &imageIn)
{
    // cout << "image callback" << endl;
    image_use_count++;
    if (image_use_count >= 1)
    {
        image_use_count = 0;
        // std::cout << "iamgeIn "<< imageIn->header.stamp.toSec() << std::endl;
        mBuf.lock();
        imageBuf.push_back(imageIn);
        mBuf.unlock();
    }
}

// IMU回调函数
void imuCallback(const sensor_msgs::Imu::ConstPtr &imuIn)
{
    // cout << "imu callback" << endl;
    mBuf.lock();
    //std::cout << "imuIn " << imuIn->header.stamp.toSec() << std::endl;
    imuBuf.push_back(imuIn);
    mBuf.unlock();
}

void getParameter()
{
    if (!ros::param::get("intrinsic_path", intrinsic_path))
    {
        cout << "Can not get the value of intrinsic_path" << endl;
        exit(1);
    }
    if (!ros::param::get("extrinsic_path", extrinsic_path))
    {
        cout << "Can not get the value of extrinsic_path" << endl;
        exit(1);
    }
    if (!ros::param::get("threshold_lidar", threshold_lidar))
    {
        cout << "Can not get the value of threshold_lidar" << endl;
        exit(1);
    }

    vector<float> intrinsic;
    getIntrinsic(intrinsic_path, intrinsic);
    vector<float> distortion;
    getDistortion(intrinsic_path, distortion);
    vector<float> extrinsic;
    getExtrinsic(extrinsic_path, extrinsic);

    // set intrinsic parameters of the camera
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = intrinsic[0]; // fx
    cameraMatrix.at<double>(0, 2) = intrinsic[2]; // cx
    cameraMatrix.at<double>(1, 1) = intrinsic[4]; // fy
    cameraMatrix.at<double>(1, 2) = intrinsic[5]; // cy

    // set radial distortion and tangential distortion
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = distortion[0];
    distCoeffs.at<double>(1, 0) = distortion[1];
    distCoeffs.at<double>(2, 0) = distortion[2];
    distCoeffs.at<double>(3, 0) = distortion[3];
    distCoeffs.at<double>(4, 0) = distortion[4];

    // set intrinsic parameters of the lidar and camera
    transformMatrix = cv::Mat::eye(4, 4, CV_64F);
    transformMatrix.at<double>(0, 0) = extrinsic[0];
    transformMatrix.at<double>(0, 1) = extrinsic[1];
    transformMatrix.at<double>(0, 2) = extrinsic[2];
    transformMatrix.at<double>(0, 3) = extrinsic[3];
    transformMatrix.at<double>(1, 0) = extrinsic[4];
    transformMatrix.at<double>(1, 1) = extrinsic[5];
    transformMatrix.at<double>(1, 2) = extrinsic[6];
    transformMatrix.at<double>(1, 3) = extrinsic[7];
    transformMatrix.at<double>(2, 0) = extrinsic[8];
    transformMatrix.at<double>(2, 1) = extrinsic[9];
    transformMatrix.at<double>(2, 2) = extrinsic[10];
    transformMatrix.at<double>(2, 3) = extrinsic[11];
}

void getTheoreticalUV(float *theoryUV, const vector<float> &intrinsic, const vector<float> &extrinsic, double x, double y, double z)
{
    // set the intrinsic and extrinsic matrix
    double matrix1[3][3] = {{intrinsic[0], intrinsic[1], intrinsic[2]}, {intrinsic[3], intrinsic[4], intrinsic[5]}, {intrinsic[6], intrinsic[7], intrinsic[8]}};
    double matrix2[3][4] = {{extrinsic[0], extrinsic[1], extrinsic[2], extrinsic[3]}, {extrinsic[4], extrinsic[5], extrinsic[6], extrinsic[7]}, {extrinsic[8], extrinsic[9], extrinsic[10], extrinsic[11]}};
    double matrix3[4][1] = {x, y, z, 1};

    // transform into the opencv matrix
    cv::Mat matrixIn(3, 3, CV_64F, matrix1);
    cv::Mat matrixOut(3, 4, CV_64F, matrix2);
    cv::Mat coordinate(4, 1, CV_64F, matrix3);

    // calculate the result of u and v
    cv::Mat result = matrixIn * matrixOut * coordinate;
    float u = result.at<double>(0, 0);
    float v = result.at<double>(1, 0);
    float depth = result.at<double>(2, 0);

    theoryUV[0] = u / depth;
    theoryUV[1] = v / depth;
}

// set the color by distance to the cloud
void getColor(int &result_r, int &result_g, int &result_b, float cur_depth)
{
    float scale = (max_depth - min_depth) / 10;
    if (cur_depth < min_depth)
    {
        result_r = 0;
        result_g = 0;
        result_b = 0xff;
    }
    else if (cur_depth < min_depth + scale)
    {
        result_r = 0;
        result_g = int((cur_depth - min_depth) / scale * 255) & 0xff;
        result_b = 0xff;
    }
    else if (cur_depth < min_depth + scale * 2)
    {
        result_r = 0;
        result_g = 0xff;
        result_b = (0xff - int((cur_depth - min_depth - scale) / scale * 255)) & 0xff;
    }
    else if (cur_depth < min_depth + scale * 4)
    {
        result_r = int((cur_depth - min_depth - scale * 2) / scale * 255) & 0xff;
        result_g = 0xff;
        result_b = 0;
    }
    else if (cur_depth < min_depth + scale * 7)
    {
        result_r = 0xff;
        result_g = (0xff - int((cur_depth - min_depth - scale * 4) / scale * 255)) & 0xff;
        result_b = 0;
    }
    else if (cur_depth < min_depth + scale * 10)
    {
        result_r = 0xff;
        result_g = 0;
        result_b = int((cur_depth - min_depth - scale * 7) / scale * 255) & 0xff;
    }
    else
    {
        result_r = 0xff;
        result_g = 0;
        result_b = 0xff;
    }
}

void projectPointCloud(const sensor_msgs::PointCloud2::ConstPtr &pointCloudCur, cv::Mat &src_img, const ros::Time time)
{
    pcl::PointCloud<pcl::PointXYZI> CloudIn;
    pcl::fromROSMsg(*pointCloudCur, CloudIn);

    //! 去畸变
    cv::Mat view, rview, map1, map2;
    cv::Size imageSize = src_img.size();
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
    cv::remap(src_img, src_img, map1, map2, cv::INTER_LINEAR); // correct the distortion

    //! 剪切图像上下边缘
    cv::Rect rect(0, 109, 1280, 450);
    src_img = src_img(rect);
    cv::Size newSize = src_img.size();
    // cout << "width: " << newSize.width << endl << "height: " << newSize.height << endl;

    //! 去畸变之后的理想内参
    optimalNewCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
    vector<float> optimalNewIntrinsic = {0, 0, 0, 0, 0, 0, 0, 0, 1};
    optimalNewIntrinsic[0] = optimalNewCameraMatrix.at<double>(0, 0);       // fx
    optimalNewIntrinsic[2] = optimalNewCameraMatrix.at<double>(0, 2) - 0;   // cx
    optimalNewIntrinsic[4] = optimalNewCameraMatrix.at<double>(1, 1);       // fy
    optimalNewIntrinsic[5] = optimalNewCameraMatrix.at<double>(1, 2) - 109; // cy
    // for (auto it = optimalNewIntrinsic.cbegin(); it != optimalNewIntrinsic.cend(); ++it)
    // {
    //     cout << *it << " " << endl;
    // }

    vector<float> extrinsic = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    extrinsic[0] = transformMatrix.at<double>(0, 0);
    extrinsic[1] = transformMatrix.at<double>(0, 1);
    extrinsic[2] = transformMatrix.at<double>(0, 2);
    extrinsic[3] = transformMatrix.at<double>(0, 3);
    extrinsic[4] = transformMatrix.at<double>(1, 0);
    extrinsic[5] = transformMatrix.at<double>(1, 1);
    extrinsic[6] = transformMatrix.at<double>(1, 2);
    extrinsic[7] = transformMatrix.at<double>(1, 3);
    extrinsic[8] = transformMatrix.at<double>(2, 0);
    extrinsic[9] = transformMatrix.at<double>(2, 1);
    extrinsic[10] = transformMatrix.at<double>(2, 2);
    extrinsic[11] = transformMatrix.at<double>(2, 3);

    // ROS_INFO("Start to project the lidar cloud");
    cv::Mat projectImage = src_img.clone();
    float x, y, z;
    float theoryUV[2] = {0, 0};
    int myCount = 0;

    for (unsigned int i = 0; i < CloudIn.points.size(); ++i)
    {
        x = CloudIn.points[i].x;
        y = CloudIn.points[i].y;
        z = CloudIn.points[i].z;

        getTheoreticalUV(theoryUV, optimalNewIntrinsic, extrinsic, x, y, z);
        int u = floor(theoryUV[0] + 0.5);
        int v = floor(theoryUV[1] + 0.5);
        int r, g, b;
        getColor(r, g, b, x);

        cv::Point p(u, v);
        cv::circle(projectImage, p, 1, cv::Scalar(b, g, r), -1);
        ++myCount;
        if (myCount > threshold_lidar)
        {
            break;
        }
    }

    // 最后发布的消息格式
    sensor_msgs::Image::Ptr project_image;
    project_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8" ,projectImage).toImageMsg();
    project_image->header.stamp = time;
    pubProjectImage.publish(*project_image);
    
    cv::namedWindow("project_image", cv::WINDOW_KEEPRATIO);
    cv::imshow("project_image", projectImage);
    cv::waitKey(1);
}

//! ROS与OpenCV的相互转换
// cv_bridge::CvImage
// cv_bridge::CvImagePtr(解引用即可变成cv_bridge::CvImage类型)
// cv_bridge::CvImageConstPtr
// sensor_msgs::Image(publish的时候, 带不带ptr均可, 但接收的是sensor_msgs::Image::ConstPtr, 尽量带ptr)
// sensor_msgs::Image::Ptr(解引用即可变成sensor_msgs::Image类型)
// sensor_msgs::Image::ConstPtr(这是一个指向const变量的指针)
// cv::Mat
//! ROS转OpenCV
// cv_bridge::toCvCopy把sensor_msgs::Image::Ptr转成cv_bridge::CvImagePtr
// cv_bridge::CvImagePtr 类型的cv_ptr, cv::Mat类型的img, img = cv_ptr->image
//! OpenCV转ROS
// cv::Mat初始化cv_bridge::CvImage类型的cv或cv::Mat类型的img, cv_bridge::CvImagePtr类型的cv_ptr, cv_ptr->image = img
// cv.toImageMsg()或cv_ptr->toImageMsg()转成sensor_msgs::Image::Ptr
// 为与ROS转OpenCV的过程保持一致(总是ptr转ptr), 尽量使用cv_ptr->toImageMsg()

// 激光雷达点云转深度图
void lidarToDepth(const sensor_msgs::PointCloud2::ConstPtr &pointCloudCur)
{
    // pointCloudCount++;
    
    pcl::PointCloud<pcl::PointXYZI> CloudIn;
    pcl::fromROSMsg(*pointCloudCur, CloudIn);

    int width = 1280, height = 720 - 270;
    float fx = optimalNewCameraMatrix.at<double>(0, 0);
    float fy = optimalNewCameraMatrix.at<double>(1, 1);
    float cx = optimalNewCameraMatrix.at<double>(0, 2) - 0;
    float cy = optimalNewCameraMatrix.at<double>(1, 2) - 109;

    Eigen::Vector3f v = Eigen::Vector3f::Zero();
    v(0) = transformMatrix.at<double>(0, 3);
    v(1) = transformMatrix.at<double>(1, 3);
    v(2) = transformMatrix.at<double>(2, 3);

    Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
    R(0, 0) = transformMatrix.at<double>(0, 0);
    R(0, 1) = transformMatrix.at<double>(0, 1);
    R(0, 2) = transformMatrix.at<double>(0, 2);
    R(1, 0) = transformMatrix.at<double>(1, 0);
    R(1, 1) = transformMatrix.at<double>(1, 1);
    R(1, 2) = transformMatrix.at<double>(1, 2);
    R(2, 0) = transformMatrix.at<double>(2, 0);
    R(2, 1) = transformMatrix.at<double>(2, 1);
    R(2, 2) = transformMatrix.at<double>(2, 2);

    // https://github.com/ISEE-Technology/CamVox/issues/25
    // 这里的sensorPose的旋转部分是(0, -1, 0; 0, 0, -1; 1, 0, 0), 是点在lidar坐标系下坐标到点在camera坐标系下坐标的坐标变换矩阵, coordinate_frame是CAMERA_FRAME
    Eigen::Translation3f t(v);                             // Lidar->Camera extrinsic translation
    Eigen::Quaternionf q = Eigen::Quaternionf(R);          // Lidar->Camera extrinsic rotation （Expressed by quaternion ）
    Eigen::Affine3f sensorPose = t * q.toRotationMatrix(); // Lidar coordinate system affine transformation to the initial pose of the camera coordinate system .
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;

    // 这里的sensorPose的旋转部分是单位阵, 是lidar坐标系到camera坐标系的基变换矩阵, coordinate_frame是LASER_FRAME
    // Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(-0.082984, -0.0221759, 0.0727962);
    // Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0, 0, 0);
    // pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

    // printf("Affine matrix :\n");
    // printf("    |%6.6f %6.6f %6.6f %6.6f| \n", sensorPose(0, 0), sensorPose(0, 1), sensorPose(0, 2), sensorPose(0, 3));
    // printf("A = |%6.6f %6.6f %6.6f %6.6f| \n", sensorPose(1, 0), sensorPose(1, 1), sensorPose(1, 2), sensorPose(1, 3));
    // printf("    |%6.6f %6.6f %6.6f %6.6f| \n", sensorPose(2, 0), sensorPose(2, 1), sensorPose(2, 2), sensorPose(2, 3));
    // printf("    |%6.6f %6.6f %6.6f %6.6f| \n", sensorPose(3, 0), sensorPose(3, 1), sensorPose(3, 2), sensorPose(3, 3));

    // createFromPointCloudWithFixedSize()函数要的放射变换矩阵是从激光坐标系到虚拟的深度相机坐标系的基变换矩阵
    // 方法1: 设置深度相机坐标系coordinate_frame为CAMERA_FRAME, 此时从激光坐标系到虚拟的深度相机坐标系的基变换矩阵是(0, 0, 1; -1, 0, 0; 0, -1, 0)
    // 方法2: 设置深度相机坐标系coordinate_frame为LASER_FRAME, 此时从激光坐标系到虚拟的深度相机坐标系的基变换矩阵是单位阵
    pcl::RangeImagePlanar::Ptr rangeImage(new pcl::RangeImagePlanar);
    rangeImage->createFromPointCloudWithFixedSize(CloudIn, width, height, cx, cy, fx, fy, sensorPose.inverse(), coordinate_frame);

    // 给range_image设置header
    rangeImage->header.seq = CloudIn.header.seq;
    rangeImage->header.frame_id = CloudIn.header.frame_id;
    rangeImage->header.stamp = CloudIn.header.stamp;

    int cols = rangeImage->width;
    int rows = rangeImage->height;

    // 以[0.5m, 130m]作为有效的深度区间
    float factor = 1.0f / (130.0f - 0.5f);
    float offset = -0.5;
    // float factor = 1.0f / (47.0f - 0.5f);
    // float offset = -0.5;

    cv::Mat _rangeImage;                                                      // rangeimage转成图片才能以msg发送出去
    _rangeImage = cv::Mat::zeros(rows, cols, cv_bridge::getCvType("mono16")); // 最后的OpenCV格式的图像

    // 遍历每一个点，生成OpenCV格式的图像
    //! range的数值大小:
    // 无穷远点 ---> 0
    // 小于0.5m ---> 0
    // 大于0.5m, 小于130m ---> 减掉0.5之后, 再除以129.5, 得到0至1之间的小数
    // 大于130m ---> 1
    //! _rangeImage.at<ushort>(j, i)的数值大小:
    // std::numeric_limits<ushort>::max()为65535
    // numeric_limits是指C++预设的数值极限, unsigned short类型数值极限的最大值为65535
    // _rangeImage.at<ushort>(j, i) = 取整(range * 65535)
    // _rangeImage.at<ushort>(j, i)约为实际物理尺度的65535 / 130 = 504倍

    // cout << "pointCloudCount: " << pointCloudCount << endl;
    // if ((pointCloudCount >= 1) && (pointCloudCount <= 1500))
    // {
    //     cout << "0.5 --- 60" << endl;
    //     for (int i = 0; i < cols; ++i)
    //     {
    //         for (int j = 0; j < rows; ++j)
    //         {
    //             float r = rangeImage->getPoint(i, j).range;
    //             if (r > 35)
    //             {
    //                 r = 0;
    //             }
    //             float range = (!std::isinf(r)) ? std::max(0.0f, std::min(1.0f, factor * (r + offset))) : 0.0;
    //             _rangeImage.at<ushort>(j, i) = static_cast<ushort>((range)*std::numeric_limits<ushort>::max());
    //         }
    //     }
    // }
    // else
    // {
    //     cout << "0.5 --- 130" << endl;
    //     for (int i = 0; i < cols; ++i)
    //     {
    //         for (int j = 0; j < rows; ++j)
    //         {
    //             float r = rangeImage->getPoint(i, j).range;
    //             float range = (!std::isinf(r)) ? std::max(0.0f, std::min(1.0f, factor * (r + offset))) : 0.0;
    //             _rangeImage.at<ushort>(j, i) = static_cast<ushort>((range)*std::numeric_limits<ushort>::max());
    //         }
    //     }
    // }

    for (int i = 0; i < cols; ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            float r = rangeImage->getPoint(i, j).range;
            float range = (!std::isinf(r)) ? std::max(0.0f, std::min(1.0f, factor * (r + offset))) : 0.0;
            _rangeImage.at<ushort>(j, i) = static_cast<ushort>((range)*std::numeric_limits<ushort>::max());
        }
    }
    
    // cv::Rect rect(600, 0, 680, 450);
    // _rangeImage = _rangeImage(rect);
    
    // 最后发布的消息格式
    sensor_msgs::Image::Ptr depth;
    depth = cv_bridge::CvImage(std_msgs::Header(), "mono16", _rangeImage).toImageMsg();
    pcl_conversions::fromPCL(rangeImage->header, depth->header); // header的转变
    depth->header.stamp = pointCloudCur->header.stamp;
    pubDepth.publish(*depth);
    // std::cout << "depth stamp: " << depth->header.stamp << std::endl;

    cv::namedWindow("depth_image", cv::WINDOW_KEEPRATIO);
    cv::imshow("depth_image", _rangeImage);
    cv::waitKey(1);
}

//! 关于两种回调函数处理模式的分析
// 模式一: ros::spinOnce()模式
// 此时回调函数和主功能模块在同一个线程, 写在同一个大循环里面
// 每次开始循环, 执行一次ros::spinOnce(), 触发回调函数接收一次数据, 数据不断向容器装填
// 如果想利用容器中的数据, 就需要设计一个不断随循环累加的计数器
// ros::spinOnce()执行, 数据装填, 对应计数器累加, 保持可以一直利用最新数据的状态, 即计数器与新进数据保持对应关系
// 以计数器为索引, 在容器中获取索引位置的数据进行使用
// 模式二: ros::spin();
// 回调函数写在主线程里, 主功能模块写在子线程里
// 主线程部分执行ros::spin(), 相当于一个不断触发回调函数接收数据的死循环
// 子线程部分执行主功能模块, 一般也写在一个死循环里
// 主线程与子线程是分时复用的, 并且在不同线程使用同一容器时, 需要使用线程锁/互斥量保证数据不会乱掉
// 因为是分时复用的, 所以主线程回调函数执行一会, 程序就会切到主功能模块线程执行一会, 然后再回到主线程回调函数执行一会, 周而复始
// 两个线程就通过同一容器进行数据上的流通/交互
// 在回调函数执行时, 容器中会攒下来一小段消息数据
// 在主功能模块执行时, 使用.front()这样的函数读取容器中的数据, 再使用.pop()这样的函数将容器中刚被读取的数据释放掉
// 容器中积攒下来的小段数据就这样被消耗掉了, 然后再到回调函数执行部分, 重新在容器中积攒一小段消息数据
// 再到主功能模块, 通过.front()和.pop()这样的方式消耗数据, 周而复始
// 保证主功能模块处理的总是最新接收的数据
// process线程
//TODO 写一个节点, 实现lidar和rgb的软同步(两种选择), 然后做lidar生成depth的转换
//TODO 这里的关键在于, 需要保证rgb和depth的时间戳与频率相同, 而rgb与lidar的频率不同
//TODO 要么把rgb降低到与lidar频率相同, 要么通过重复合并lidar, 把lidar提高到和rgb频率相同
//TODO 这之后, 再做lidar到depth的转换, 所以lidar和depth是天然同步的
//TODO 最终实现lidar-rgb-depth三者的同步

void process()
{
    while (1)
    {
        if ((!imageBuf.empty()) && (!pointCloudBuf.empty()) && (pointCloudBuf.size() > 10) && (imageBuf.size() > 10))
        {
            mBuf.lock();
            sensor_msgs::PointCloud2::ConstPtr pointCloudCur = pointCloudBuf.front(); // 读取点云链表头节点的图片地址
            pointCloudBuf.pop_front();                                                // 删除点云链表头结点
            double pointCloudTimeCur = pointCloudCur->header.stamp.toSec();           // 读取当前帧点云时间戳
            sensor_msgs::Image::Ptr imageCur;

            double imageTimeCur = 0;
            double imageTimeLast = 0;
            double imageTimeNext = 0;

            bool sign = false;
            // int count = 0;
            for (imageIter = imageBuf.begin(); (imageIter != imageBuf.end()) && ((pointCloudTimeCur) > (*imageIter)->header.stamp.toSec()); imageIter++)
            {
                // count++;
                sign = true;
            }
            if (sign == true)
            {
                // std::cout << "count: " << count << std::endl;
                imageIter--;
            }

            imageTimeLast = (*imageIter)->header.stamp.toSec(); // 迭代器使用"运算符重载的*"返回容器中的数据
            imageIter++;
            imageTimeNext = (*imageIter)->header.stamp.toSec();

            double timeFlag = (pointCloudTimeCur - imageTimeLast) - (imageTimeNext - pointCloudTimeCur);
            // std::cout << "timeFlag: " << timeFlag << std::endl;
            if (timeFlag <= 0)
            {
                // std::cout << "time: " << (pointCloudTimeCur - imageTimeLast) << std::endl;
                imageIter--;
            }

            imageTimeCur = (*imageIter)->header.stamp.toSec();
            try
            {
                cv_cur_ptr = cv_bridge::toCvCopy(*imageIter, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception &e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            while (imageBuf.front()->header.stamp.toSec() <= imageTimeCur)
            {
                imageBuf.pop_front();
            }

            mBuf.unlock();

            cv::Mat src_img;
            src_img = cv_cur_ptr->image;

            projectPointCloud(pointCloudCur, src_img, pointCloudCur->header.stamp);
            lidarToDepth(pointCloudCur);

            // cv::Rect rect(600, 0, 680, 450);
            // src_img = src_img(rect);
            
            cv_cur_ptr->image = src_img;
            imageCur = cv_cur_ptr->toImageMsg();
            imageCur->header.stamp = pointCloudCur->header.stamp;
            pubRgb.publish(*imageCur);
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_to_depth");
    ros::NodeHandle nh;

    // ros::Subscriber subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/livox_undistort", 1000, pointCloudCallback); // 点云
    // ros::Subscriber subImage = nh.subscribe<sensor_msgs::Image>("/isee_rgb", 1000, imageCallback);                        // 图像
    ros::Subscriber subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1000, pointCloudCallback);
    ros::Subscriber subImage = nh.subscribe<sensor_msgs::Image>("/camera/image_raw", 1000, imageCallback);
    ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu>("/imu", 1000, imuCallback); // IMU

    pubRgb = nh.advertise<sensor_msgs::Image>("/rgb", 1000);
    pubDepth = nh.advertise<sensor_msgs::Image>("/depth", 1000);
    pubProjectImage = nh.advertise<sensor_msgs::Image>("/project_image", 1000);

    getParameter();

    std::thread lidarToDepthProcess{process}; // process线程
    ros::spin();

    return 0;
}
