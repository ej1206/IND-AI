#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#define ROI_X 10
#define ROI_Y 10
#define ROI_Z 5
ros::Publisher roi_pub;

void pointcloud_callback(sensor_msgs::PointCloud2 msg);


int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcl_roi");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/os1_cloud_node/points", 1, pointcloud_callback);
    roi_pub = n.advertise<sensor_msgs::PointCloud2>("/roi_points", 1);
    ros::spin();
    return 0;
}

void pointcloud_callback(sensor_msgs::PointCloud2 msg)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::fromROSMsg(msg, cloud);
    pcl::PointCloud<pcl::PointXYZI> roi_cloud;
    for (size_t i = 0;i<cloud.size();i++)
    {
        if(abs(cloud.at(i).x) < ROI_X && (abs(cloud.at(i).y)< ROI_Y) && abs(cloud.at(i).z) < ROI_Z)
        {
            roi_cloud.push_back(cloud.at(i));
        }
    }
    //sensor header
    roi_cloud.header = cloud.header;
    roi_cloud.is_dense = cloud.is_dense;
    roi_cloud.height = 1;
    roi_cloud.width = roi_cloud.size();
    sensor_msgs::PointCloud2 pub_msg;
    pcl::toROSMsg(roi_cloud, pub_msg);
    roi_pub.publish(pub_msg);
}
