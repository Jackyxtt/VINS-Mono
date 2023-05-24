// 成功实现了多层金字塔光流跟踪法
// 直接使用cv::calcOpticalFlowPyrLK，输入为两个图像金字塔
// 效果不错
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

bool compareOpticalFlowPyramids(const std::vector<cv::Mat>& pyramid1, const std::vector<cv::Mat>& pyramid2) {
    if (pyramid1.size() != pyramid2.size()) {
        return false;  // 金字塔层数不同
    }

    for (size_t i = 0; i < pyramid1.size(); ++i) {
        if (pyramid1[i].size() != pyramid2[i].size() || pyramid1[i].type() != pyramid2[i].type()) {
            return false;  // 金字塔层尺寸或类型不同
        }

        std::vector<cv::Mat> channels1, channels2;
        cv::split(pyramid1[i], channels1);
        cv::split(pyramid2[i], channels2);

        for (int c = 0; c < channels1.size(); ++c) {
            cv::Mat channel_diff;
            cv::absdiff(channels1[c], channels2[c], channel_diff);

            if (cv::countNonZero(channel_diff) > 0) {
                return false;  // 金字塔层像素值不同
            }
        }
    }

    return true;  // 金字塔相同
}

int main()
{
    // 读取初始图像
    cv::Mat frame1 = cv::imread("/home/dev/test/MCVIO-main/VINS_Mono_ws/src/VINS-Mono/feature_tracker/src/example_photos/LK1.png");
    cv::Mat frame2 = cv::imread("/home/dev/test/MCVIO-main/VINS_Mono_ws/src/VINS-Mono/feature_tracker/src/example_photos/LK2.png");
    cout << "finish imread" << endl;

    // 转换为灰度图像
    cv::Mat gray1, gray2;
    gray1 = frame1;
    gray2 = frame2;

    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    // 提取特征点
    int maxCorners = 100; // 要提取的最大特征点数
    double qualityLevel = 0.3; // 特征点的质量水平阈值
    double minDistance = 7; // 特征点之间的最小距离
    int blockSize = 7; // 角点检测中的邻域大小
    bool useHarrisDetector = false; // 是否使用Harris角点检测器
    double k = 0.04; // Harris角点检测器的自由参数
    std::vector<cv::Point2f> points1, points2;

    cv::goodFeaturesToTrack(gray1, points1, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
    points2 = points1;


    // 构建金字塔
    std::vector<cv::Mat> pyramid1, pyramid2;

    cout << "input image has " << gray1.channels() << " channels" << endl;
    cv::buildOpticalFlowPyramid(gray1, pyramid1, cv::Size(21, 21), 3);
    cv::buildOpticalFlowPyramid(gray2, pyramid2, cv::Size(21, 21), 3);
    cout << pyramid1.size() << endl;
    cout << "finish buildOpticalFlowPyramid" << endl;
    // 设置光流参数
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    int flags = cv::OPTFLOW_USE_INITIAL_FLOW;


    if (compareOpticalFlowPyramids(pyramid1, pyramid2)){
        cout << "pyramid1 and pyramid2 is same " << endl;
    }else{
        cout << "pyramid1 and pyramid2 not same " << endl;
    }

    cv::calcOpticalFlowPyrLK(pyramid1, pyramid2, points1, points2, status, err, cv::Size(21, 21), 3, criteria, flags);
    cout << "finish calcOpticalFlowPyrLK" << endl;

    // 绘制特征点和光流轨迹
    for (int i = 0; i < points1.size(); ++i)
    {
        if (status[i])
        {
            cv::Point2f pt1 = points1[i];
            cv::Point2f pt2 = points2[i];
            cv::line(frame2, pt1, pt2, cv::Scalar(0, 0, 255), 2);
            cv::circle(frame2, pt2, 3, cv::Scalar(0, 255, 0), -1);
        }
    }

    // 显示结果
    cv::imshow("Optical Flow", frame2);
    cv::waitKey(0);

    return 0;
}