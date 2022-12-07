#include <stdio.h>
#include <cmath>
#include <sstream>
#include <iostream>
#include <vector>
#include <chrono>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>

#include <opencv2/opencv.hpp>

#include "camera.h"
#include "livox_ros_driver/CustomMsg.h"
#include "data_preprocess/GPS.h"

using namespace std;
using namespace cv;
using namespace MVS;

typedef pcl::PointXYZI PointTypeXYZI;

vector<sensor_msgs::Image> camera_datas; // 未使用
vector<sensor_msgs::PointCloud2ConstPtr> lidar_datas;
vector<nav_msgs::Odometry> imu_datas;
vector<data_preprocess::GPS> rtk_datas;

double lidar_delta_time = 0.1;                                                                              // 10Hz的lidar数据，你可以根据你所使用的传感器修改这项参数，未使用
pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudFullResColor_pcd(new pcl::PointCloud<pcl::PointXYZRGB>()); // 未使用

ros::Publisher pub_isee_depth;
ros::Publisher pub_apx_rpy;

// 函数声明
void livox_lidar2depth(uint32_t num_lidar);

// 统一时间格式，使用协调世界时间UTC
inline double to_time(sensor_msgs::Image camera_data)
{
    return (camera_data.header.stamp.sec) * 1.0 + (camera_data.header.stamp.nsec / 1000000000.0);
}

inline double to_time(sensor_msgs::PointCloud2ConstPtr lidar_data)
{
    return (lidar_data->header.stamp.sec) * 1.0 + (lidar_data->header.stamp.nsec / 1000000000.0);
}

inline double to_time(nav_msgs::Odometry imu_data)
{
    return (imu_data.header.stamp.sec) * 1.0 + (imu_data.header.stamp.nsec / 1000000000.0);
}

inline double to_time(data_preprocess::GPS rtk_data) //sensor_msgs::NavSatFix
{
    return (rtk_data.header.stamp.sec) * 1.0 + (rtk_data.header.stamp.nsec / 1000000000.0);
}

// data_preprocess::GPS::ConstPtr sensor_msgs::NavSatFix::ConstPtr
// GNSS回调函数
void rtkCbk(const data_preprocess::GPS::ConstPtr &msg)
{
    rtk_datas.push_back(*msg); //! gps_save
}

// transform sensor_msgs::Imu to nav_msgs::Odometry
// 订阅IMU数据，转换成odometry数据，再发布出去，然后odometry回调函数再订阅odometry数据
//TODO apximuCbk()函数和imuCbk()函数合并，取消自发自收环节，是不是因为发布的是nav_msgs::Odometry，而接受时是nav_msgs::Odometry::ConstPtr
// IMU回调函数
void apximuCbk(const sensor_msgs::Imu::ConstPtr &msg)
{
    nav_msgs::Odometry tempOdo;
    tempOdo.header.stamp = msg->header.stamp;
    tempOdo.pose.pose.orientation.x = msg->orientation.x;
    tempOdo.pose.pose.orientation.y = msg->orientation.y;
    tempOdo.pose.pose.orientation.z = msg->orientation.z;
    tempOdo.pose.pose.orientation.w = msg->orientation.w;
    tempOdo.header.frame_id = "/camera_init";

    pub_apx_rpy.publish(tempOdo);
}

// lidar回调函数
void lidarCbk(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    if (lidar_datas.size() > 0)
    {
        if (to_time(lidar_datas[lidar_datas.size() - 1]) > to_time(msg))
        {
            ROS_INFO("lidar time error");
            return;
        }
    }

    lidar_datas.push_back(msg); //! lidar_save
}

// odometry回调函数
void imuCbk(const nav_msgs::Odometry::ConstPtr &msg)
{
    imu_datas.push_back(*msg); //! imu_save
}

// 主函数
int main(int argc, char **argv)
{
    ros::init(argc, argv, "livox_horizon_to_depth");
    ros::NodeHandle n;

    cv::Mat src;
    MVS::Camera MVS_cap(n);
    image_transport::ImageTransport cam_image(n);
    image_transport::CameraPublisher image_pub = cam_image.advertiseCamera("/isee_rgb", 1); //! 发布rgb图像
    boost::shared_ptr<camera_info_manager::CameraInfoManager> cam_info;
    sensor_msgs::Image image_msg;
    cam_info.reset(new camera_info_manager::CameraInfoManager(n, "main_camera", ""));

    if (!cam_info->isCalibrated())
    {
        cam_info->setCameraName("/dev/video0");
        sensor_msgs::CameraInfo cam_info_;
        cam_info_.header.frame_id = image_msg.header.frame_id;
        cam_info_.width = 1520;
        cam_info_.height = 568;
        cam_info->setCameraInfo(cam_info_);
    }

    sensor_msgs::CameraInfoPtr camera_info;
    cv_bridge::CvImagePtr cv_ptr = boost::make_shared<cv_bridge::CvImage>();
    cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;

    ros::Subscriber sub_rtk = n.subscribe("/gps", 1000, rtkCbk);                 // inertial_gps 10hz //! 订阅GNSS
    ros::Subscriber sub_apx_rpy = n.subscribe("/imu", 10000, apximuCbk);         // inertial_imu 100hz //! 订阅IMU
    pub_apx_rpy = n.advertise<nav_msgs::Odometry>("pub_apx_rpy", 1000);          //! 发布odometry
    ros::Subscriber sub_point = n.subscribe("/livox_undistort", 1000, lidarCbk); // livox_lidar 10hz //! 订阅去畸变/不失真的lidar
    ros::Subscriber sub_imu = n.subscribe("pub_apx_rpy", 1000, imuCbk);          // inertial_imu_need //! 订阅odometry

    pub_isee_depth = n.advertise<sensor_msgs::Image>("/isee_depth", 1);       //! 发布depth图像

    std::string map_file_path;
    ros::param::get("~map_file_path", map_file_path);

    // timestamp align
    // 时间戳对齐
    // 以雷达为基准, 先确定雷达在两帧IMU数据之间, 然后每一帧雷达数据, 对应着读一帧相应时刻的rgb数据
    // 并把雷达的时间戳赋值给rgb, 这样就使得雷达和rgb数据频率相同, 时间戳相同
    // 实际上rgb频率要比雷达高, 这里是相当于从大量的rgb数据中挑出与当前雷达帧时间相近的, 从而实现雷达与rgb的同步
    // 对于每一帧雷达, 生成相应的depth数据, 并把雷达的时间戳赋值给depth, 这样就使得雷达和depth数据频率相同, 时间戳相同
    // 以上, 间接实现rgb与depth的同步, 最终实现lidar-rgb-depth三者的同步
    //TODO lidar和depth是天然同步的, 这里的关键是写一个节点, 实现lidar和rgb的软同步, 然后做lidar生成depth的转换, 最终实现lidar-rgb-depth三者的同步
    uint32_t num_camera = 0;
    uint32_t num_lidar = 0;
    uint32_t num_imu = 1;
    uint32_t num_camera_last = 0;
    uint32_t num_lidar_last = 0;
    uint32_t num_imu_last = 1;

    bool init_flag = false; // 标志位
    while (n.ok())
    {
        ros::spinOnce();

        if (num_lidar < lidar_datas.size())
        {
            bool imu_flag = false;

            // imu data align
            if (num_imu < imu_datas.size())
            {
                if (to_time(imu_datas[num_imu - 1]) <= to_time(lidar_datas[num_lidar]))
                {
                    if (to_time(imu_datas[num_imu]) >= to_time(lidar_datas[num_lidar]))
                    {
                        imu_flag = true;
                    }
                    else
                    {
                        num_imu++;
                    }
                }
                else
                {
                    num_lidar++;
                    num_camera++;
                    continue;
                }
            }

            if (imu_flag)
            {
                if (init_flag)
                {
                    //! 发布rgb图像
                    MVS_cap >> src; // <<运算符重载成读图函数
                    if (src.empty())
                    {
                        continue;
                    }
                    cv_ptr->image = src;
                    image_msg = *(cv_ptr->toImageMsg());
                    image_msg.header.stamp = lidar_datas[num_lidar_last]->header.stamp;
                    image_msg.header.frame_id = "camera_init";
                    camera_info = sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo(cam_info->getCameraInfo()));
                    camera_info->header.frame_id = image_msg.header.frame_id;
                    camera_info->header.stamp = image_msg.header.stamp;
                    image_pub.publish(image_msg, *camera_info);

                    //! 发布depth图像
                    // 记录lidar点云转depth图所用时间
                    auto t0 = std::chrono::steady_clock::now();
                    livox_lidar2depth(num_lidar_last);
                    std::cout << "point cloud 2 depth 时间" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()
                              << '\n';

                    std::cout << setprecision(20);
                    std::cout << "camera stamp: " << image_msg.header.stamp << std::endl;
                    std::cout << "imu stamp: " << to_time(imu_datas[num_imu_last]) << std::endl;

                    std::cout << num_imu_last << std::endl;
                    std::cout << num_lidar_last << std::endl;
                    std::cout << num_camera_last << std::endl;

                    num_lidar_last = num_lidar;
                    num_camera_last = num_camera;
                    num_imu_last = num_imu;
                }
                else
                {
                    std::cout << " RTK service start ! " << std::endl;
                    init_flag = true;
                }
                //  std::cout<<setprecision(20);
                //  std::cout<<"lidar_datas : "<< to_time(lidar_datas[num_lidar]) << std::endl;
                //  std::cout<<"camera_datas : "<< to_time(camera_datas[num_camera]) << std::endl; // 未使用
                //  std::cout<<"imu_datas[num_imu] : "<< to_time(imu_datas[num_imu]) << std::endl;
                //  std::cout<<"imu_datas[num_imu+1] : "<< to_time(imu_datas[num_imu+1]) << std::endl;

                num_lidar++;
                num_camera++;
            }
        }
    }
    return 0;
}

// lidar点云转depth图
void livox_lidar2depth(uint32_t num_lidar)
{

    pcl::PointCloud<pcl::PointXYZI> CloudIn;
    pcl::fromROSMsg(*lidar_datas[num_lidar], CloudIn);

    int width = 1520, height = 568, size = 2, type = 0;
    float fx = 1732.78774, fy = 1724.88401, cx = 798.426021, cy = 312.570668;

    // https://github.com/ISEE-Technology/CamVox/issues/25
    // After round 2 best rpy:-0.0375,0.05,0.6
    // Rotation matrix :
    //     | 0.999945 -0.010472 0.000866 |
    // R = | 0.010472 0.999945 0.000664 |
    //     | -0.000873 -0.000654 0.999999 |
    // t = (-0.082984, -0.0221759, 0.0727962)
    // Eigen::Translation3f t； // Lidar->Camera extrinsic translation
    // Eigen::Quaternionf q； // Lidar->Camera extrinsic rotation （Expressed by quaternion ）
    // Eigen::Affine3f sensorPose = t*q.toRotationMatrix(); // Lidar coordinate system affine transformation to the initial pose of the camera coordinate system .
    // pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;

    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(-0.082984, -0.0221759, 0.0727962);
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
    float noiseLevel = 0.0; //设置噪声水平
    float minRange = 0.0f;  //成像时考虑该阈值外的点

    pcl::RangeImagePlanar::Ptr rangeImage(new pcl::RangeImagePlanar);
    rangeImage->createFromPointCloudWithFixedSize(CloudIn, width, height, cx, cy, fx, fy, sensorPose, coordinate_frame);

    // 给range_image设置header
    rangeImage->header.seq = CloudIn.header.seq;
    rangeImage->header.frame_id = CloudIn.header.frame_id;
    rangeImage->header.stamp = CloudIn.header.stamp;

    int cols = rangeImage->width;
    int rows = rangeImage->height;

    // 转换因子
    float factor = 1.0f / (130.0f - 0.5f);
    float offset = -0.5;

    cv::Mat _rangeImage;                                                      // rangeimage转成图片才能以msg发送出去
    _rangeImage = cv::Mat::zeros(rows, cols, cv_bridge::getCvType("mono16")); // 最后的OpenCV格式的图像
    // 遍历每一个点，生成OpenCV格式的图像
    for (int i = 0; i < cols; ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            float r = rangeImage->getPoint(i, j).range;
            float range = (!std::isinf(r)) ? std::max(0.0f, std::min(1.0f, factor * (r + offset))) : 0.0;
            _rangeImage.at<ushort>(j, i) = static_cast<ushort>((range)*std::numeric_limits<ushort>::max());
        }
    }
    // 最后发布的消息格式
    sensor_msgs::ImagePtr msg;
    msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", _rangeImage).toImageMsg();
    pcl_conversions::fromPCL(rangeImage->header, msg->header); // header的转变
    msg->header.stamp = lidar_datas[num_lidar]->header.stamp;
    pub_isee_depth.publish(msg);
    std::cout << "depth stamp: " << msg->header.stamp << std::endl;
}
