// 成功实现了多层金字塔光流跟踪法
// 使用了手动特征缩放，上一层提取到的特征点points1,以及追踪到的特征点points2,在图像金字塔的下一层中扩大二倍
// 效果不如直接使用cv::calcOpticalFlowPyrLK追踪的效果好

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main()
{
    // 读取初始图像
    cv::Mat frame1 = cv::imread("/home/dev/test/MCVIO-main/VINS_Mono_ws/src/VINS-Mono/feature_tracker/src/example_photos/LK1.png");
    cv::Mat frame2 = cv::imread("/home/dev/test/MCVIO-main/VINS_Mono_ws/src/VINS-Mono/feature_tracker/src/example_photos/LK2.png");

    // 转换为灰度图像
    cv::Mat gray1, gray2;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    // 构建金字塔
    std::vector<cv::Mat> pyramid1, pyramid2;
    int numLevels = 3; // 金字塔层数
    cv::buildOpticalFlowPyramid(gray1, pyramid1, cv::Size(21, 21), numLevels);
    cv::buildOpticalFlowPyramid(gray2, pyramid2, cv::Size(21, 21), numLevels);

    // 设置光流参数
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    int flags = cv::OPTFLOW_LK_GET_MIN_EIGENVALS;


    std::vector<cv::Point2f> points1, points2;
    // 对每个金字塔层进行光流跟踪

    // 0,2,4,6层的channel数为1,才可以正常跟踪
    for (int level = 4; level >= 0; )
    {
        
        if (level == 4){
            // 提取特征点
            int maxCorners = 1000; // 最大特征点数
            double qualityLevel = 0.01; // 特征点质量水平阈值
            double minDistance = 8.0; // 特征点之间的最小距离
            cv::goodFeaturesToTrack(pyramid1[level], points1, maxCorners, qualityLevel, minDistance);
            points2 = points1;
        }

        if (level < 4)
        {
            // 将上一层的特征点位置乘以2，用于在当前金字塔层上进行光流跟踪
            for (cv::Point2f& point : points1)
            {
                point *= 2;
            }

            for (cv::Point2f& point : points2)
            {
                point *= 2;
            }
        }



        // 使用Lucas-Kanade光流算法进行跟踪
        cv::calcOpticalFlowPyrLK(pyramid1[level], pyramid2[level], points1, points2, status, err, cv::Size(21, 21), numLevels, criteria, flags);
        cout << "finish calcOpticalFlowPyrLK" << endl;

        cv::Mat tmp = pyramid1[level];
        // 绘制特征点和光流轨迹
        for (int i = 0; i < points1.size(); ++i)
        {
            if (status[i])
            {
                cv::Point2f pt1 = points1[i];
                cv::Point2f pt2 = points2[i];
                
                cv::line(tmp, pt1, pt2, cv::Scalar(0, 0, 255), 2);
                cv::circle(tmp, pt2, 3, cv::Scalar(0, 255, 0), -1);
            }
        }
        // 显示结果
        cv::imshow("Optical Flow", tmp);
        cv::waitKey(0);

        level = level - 2;
    }

    return 0;
}
