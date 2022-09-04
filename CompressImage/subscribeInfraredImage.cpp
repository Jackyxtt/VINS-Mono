#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/image_encodings.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
cv::Mat imgCallback;
static void ImageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
      cv_bridge::CvImagePtr cv_ptr;
            //和vins_mono代码一样，不然图片显示不出来
            sensor_msgs::Image img;
            img.header = msg->header;
            img.height = msg->height;
            img.width = msg->width;
            img.is_bigendian = msg->is_bigendian;
            img.step = msg->step;
            img.data = msg->data;
            img.encoding = "mono8";
            cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);

            cv_ptr = cv_bridge::cvtColor(cv_ptr, sensor_msgs::image_encodings::BGR8);

            imgCallback = cv_ptr->image;
      
      cv::imshow("imgCallback",imgCallback);
      cv::resizeWindow("imgCallback", 640, 480);
      cv::waitKey(1);
      cout<<"cv_ptr: "<<cv_ptr->image.cols<<" h: "<<cv_ptr->image.rows<<endl;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Could not convert from '%s' to 'MONO8'.", msg->encoding.c_str());
    }
}
int main(int argc, char **argv)
{
  ros::init(argc, argv, "Image");
  ros::NodeHandle nh;
  ros::Subscriber image_sub;
  std::string image_topic = "/thermal_image_raw";
  image_sub = nh.subscribe(image_topic,10,ImageCallback);
 
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    ROS_INFO("ROS OK!");
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}