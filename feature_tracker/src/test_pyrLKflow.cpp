// 成功实现了多层金字塔光流跟踪法
// 直接使用cv::calcOpticalFlowPyrLK，输入为两个图像金字塔
// 效果不错
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <vector>

using namespace cv;
using namespace std;

int patch_size = 15;
int pyramid_levels = 3; 
int max_iteration = 30;
int track_precision = 0.01;
int COL = 640;
int ROW = 512;
cv::Mat mask;
std::vector<cv::Mat> cur_img_pyramid_;
std::vector<cv::Mat> prev_img_pyramid_;
std::vector<cv::Mat> forw_img_pyramid_;
std::vector<cv::Mat> img_pyramid_;
vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
vector<cv::Point2f> n_pts;
vector<int> track_cnt;
vector<int> ids;
int MIN_DIST = 30;
int MAX_CNT = 100;

void addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void setMask()
{
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    // prefer to keep features that are tracked for long time
    // 倾向于留下被追踪时间很长的特征点
    // 构造(cnt，pts，id)序列，（追踪次数，当前特征点坐标，id）
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    // cout << "finish constructing cnt_pts_id" << endl;

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });
    // cout << "finish sorting" << endl;

    //清空cnt，pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)// 这个特征点对应的mask值为255，表明点是黑的，还没占有
        {
            //则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // cout << "finish reconstructing forw_pts, ids, track_cnt" << endl;
             //在mask中将当前特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
    // cout << "finish setting mask" << endl;
}

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

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
        else
            cout << "reducing " << i << endl;
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
        else
            cout << "reducing " << i << endl;
    v.resize(j);
}

void img_callback(const sensor_msgs::CompressedImageConstPtr &img_msg){
    // 将图像编码8UC1转换为mono8,单色8bit
    cv_bridge::CvImageConstPtr ptr;

        // std::cout << "toCvCopy 8UC3 to BGR8" << std::endl;
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat show_img = ptr->image;

    cv::Mat img;
    if (true)
    {
        //自适应直方图均衡
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(show_img, img);
    }
    
    img_pyramid_.clear();
    cv::buildOpticalFlowPyramid(
      img, img_pyramid_,
      cv::Size(patch_size, patch_size),
      pyramid_levels, true, cv::BORDER_REFLECT_101,
      cv::BORDER_CONSTANT, false);
    cout << "finish buildOpticalFlowPyramid" << endl;

    // 2. 判断当前帧图像forw_img是否为空
    if (forw_img_pyramid_.empty())
    {
        //如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
        cout << "first image " << endl;
        prev_img_pyramid_ = cur_img_pyramid_ = forw_img_pyramid_ = img_pyramid_;
    }
    else
    {
        //否则，说明之前就已经有图像读入，只需要更新当前帧forw_img的数据
        cout << "new incoming image " << endl;
        forw_img_pyramid_ = img_pyramid_;
    }
    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();
    forw_pts = cur_pts;

    if (cur_pts.size() > 0)// 前一帧有特征点
    {
        vector<uchar> status;
        vector<float> err;
        // 3. 调用cv::calcOpticalFlowPyrLK()对前一帧的特征点cur_pts进行LK金字塔光流跟踪，得到forw_pts
        //status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0
        if (compareOpticalFlowPyramids(forw_img_pyramid_, cur_img_pyramid_)){
            cout << "cur_img_pyramid_ and forw_img_pyramid_ is same " << endl;
        }else{
            cout << "cur_img_pyramid_ and forw_img_pyramid_ not same " << endl;
        }
        cv::calcOpticalFlowPyrLK(
        cur_img_pyramid_, forw_img_pyramid_, 
        cur_pts, forw_pts, 
        status, err, 
        cv::Size(patch_size, patch_size),
        pyramid_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
            max_iteration,
            track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);
        cout << "finish calcOpticalFlowPyrLK" << endl;

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))// 将当前帧跟踪的位于图像边界外的点标记为0
                status[i] = 0;
        // 4. 根据status,把跟踪失败的点剔除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        cout << "finish reduceVector" << endl;
    }
    
    for (auto &n : track_cnt)
        n++;
    cout << "finish adding track_cnt" << endl;

    //PUB_THIS_FRAME=1 需要发布特征点
    if (1)
    {
        // // 6. rejectWithF()通过基本矩阵剔除outliers
        // rejectWithF();

        // 7. setMask()保证相邻的特征点之间要相隔30个像素,设置mask
        setMask();
        cout << "finish setMask" << endl;

        // 8. 寻找新的特征点 goodFeaturesToTrack()
        //计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            n_pts.clear();
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;

            /** 
             *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值
             *)   
             */

            cv::goodFeaturesToTrack(img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
            cout << "finish goodFeaturesToTrack " << endl;
        }
        else
            n_pts.clear();

        // 9. addPoints()向forw_pts添加新的追踪点
        //添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,track_cnt初始化为1.
        addPoints();
        cout << "finish addPoints " << endl;

        // 绘制特征点和光流轨迹
        cout << "---cur frame has " << forw_pts.size() << " features " << endl;
        for (int i = 0; i < forw_pts.size(); ++i)
        {
            cv::Point2f pt1 = forw_pts[i];
            cv::Point2f pt2 = forw_pts[i];
            cv::line(img, pt1, pt2, cv::Scalar(0, 0, 255), 2);
            cv::circle(img, pt2, 3, cv::Scalar(0, 255, 0), -1);
        }

        // 显示结果
        cv::imshow("Optical Flow", img);
        cv::waitKey(5);

    }

    // 10. 更新帧、特征点
    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据

    prev_img_pyramid_ = cur_img_pyramid_;

    prev_pts = cur_pts;

    //把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img_pyramid_ = forw_img_pyramid_;

    cur_pts = forw_pts;

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_pyrLK");
    ros::NodeHandle nh("~");

    ros::Subscriber sub_img = nh.subscribe("/usb_cam/image_raw/compressed", 100, img_callback);

    ros::spin();
    return 0;
}