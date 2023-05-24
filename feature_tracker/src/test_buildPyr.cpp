// 建立图像金字塔，并可视化每张图像
// 0,2,4,6层的channel数为1,才可以正常显示
// 其他奇数层的channel数不为1,不知道是什么原因
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

int main()
{
    // 读取输入图像
    cv::Mat image = cv::imread("/home/dev/test/MCVIO-main/VINS_Mono_ws/src/VINS-Mono/feature_tracker/src/example_photos/LK1.png",cv::IMREAD_GRAYSCALE);;

    // 构建金字塔
    std::vector<cv::Mat> pyramid;
    int numLevels = 3; // 金字塔层数
    cv::buildOpticalFlowPyramid(image, pyramid, cv::Size(21, 21), numLevels);
    cout << pyramid.size() << endl;
    // 可视化金字塔图像

    // 0,2,4,6层的channel数为1,才可以正常显示
    for (int level = 6; level < pyramid.size(); ++level)
    {
        cout << pyramid[level].size() << endl;
        cout << pyramid[level].channels() << endl;
        // 创建一个窗口
        cv::namedWindow("Pyramid Level", cv::WINDOW_NORMAL);

        // 调整窗口大小以适应图像
        cv::resizeWindow("Pyramid Level", pyramid[level].cols, pyramid[level].rows);

        // 在窗口中显示当前金字塔层的图像
        cv::imshow("Pyramid Level", pyramid[level]);

        // 等待按键
        cv::waitKey(0);

        // 销毁窗口
        cv::destroyWindow("Pyramid Level");
    }

    return 0;
}