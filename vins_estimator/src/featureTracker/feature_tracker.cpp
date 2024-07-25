/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker.h"

// 常量和定义：
#define LET_WIDTH 256 // 512 256
#define LET_HEIGHT 192 // 384 192

struct greaterThanPtr
{
    bool operator () (const float * a, const float * b) const
    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
    #ifdef LET_NET
        printf("Using LET-Net feature tracker\n");
        let_init(); // 初始化 LET-Net 模型。
    #else
        printf("Using Oringal feature tracker\n");
    #endif
}

void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 对图片进行一系列操作，返回特征点featureFrame。
// 其中还包含了：图像处理、区域mask、检测特征点、计算像素速度等
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r; // 定义一个时间，为了计时
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    if(!_img1.empty() && stereo_cam)
    {
        rightImg = _img1;
    }
    
    #ifdef LET_NET
        // if (EQUALIZE)
        // {
        //     // 使用 CLAHE 对图像进行直方图均衡化。
        //     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        //     TicToc t_c; // 用于记录 CLAHE 操作的时间
        //     clahe->apply(cur_img, cur_img);

        //     if(!_img1.empty() && stereo_cam)
        //     {
        //         clahe->apply(rightImg, rightImg);
        //     }
        //     ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
        // }

        if(!_img1.empty() && stereo_cam)
        {
            desc = let_net(cur_img); // 调用 LET-Net 进行特征点检测和描述。 左目
            rightDesc = let_net_right(rightImg);
            cv::cvtColor(cur_img, gray, cv::COLOR_BGR2GRAY); // 将图像转换为灰度图像。
            cv::cvtColor(rightImg, grayRight, cv::COLOR_BGR2GRAY);
        }
        else
        {
            desc = let_net(cur_img);
            cv::cvtColor(cur_img, gray, cv::COLOR_BGR2GRAY);
        }
    #endif

    cur_pts.clear();

    // 1. let-net 的声明变量和计算缩放比例：
    #ifdef LET_NET
        std::vector<cv::Point2f> corners1, corners2;
        int w0 = cur_img.cols; int h0 = cur_img.rows; // 获取图像的宽和高
        float k_w = float(w0) / float(LET_WIDTH); // 计算宽度缩放比例
        float k_h = float(h0) / float(LET_HEIGHT); // 计算高度缩放比例
    #endif

    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        if(hasPrediction)
        {
            cur_pts = predict_pts; //当前的点等于上一次的点

            #ifdef LET_NET
                // printf("LET_NET has prediction\n");
                corners1.resize(cur_pts.size()); 
                for (int i = 0; i < int(prev_pts.size()); i++)
                {
                    corners1[i].x = prev_pts[i].x / k_w; // 将 上一帧 点的x坐标缩放到网络输入大小
                    corners1[i].y = prev_pts[i].y / k_h; // 将 上一帧 点的y坐标缩放到网络输入大小
                }
                // 3. 计算光流
                // 使用金字塔光流法 cv::calcOpticalFlowPyrLK 计算从上一帧到当前帧的光流。
                // 这里输入的是缩放后的 上一帧 特征点位置 corners1，输出的是在当前帧中的位置 corners2，并记录每个点的状态 status 和误差 err。
                cv::calcOpticalFlowPyrLK(last_desc, desc, corners1, corners2, status, err, cv::Size(21, 21), 3);
                /*
                参数解释：
                    last_desc：前一帧的特征描述子。
                    desc：当前帧的特征描述子。
                    corners1：前一帧的特征点坐标。
                    corners2：输出的在当前帧中找到的特征点坐标。
                    status：输出的每个点的状态，如果为 1，则表示该点被找到，否则表示该点未被找到。（是否成功跟踪）。
                    err：输出误差向量，每个特征点的跟踪误差。
                    cv::Size(21, 21)：搜索窗口的大小（21x21像素）。
                    5：金字塔层数。
                作用：
                    这是在使用特征描述子进行光流计算，通过描述子来匹配前后两帧的特征点。适用于使用特征描述子的场景，如深度学习特征点提取后的跟踪。
                */

                // 4. 调整光流计算后的点的位置
                // resize corners2 to cur_pts
                // 将计算后的 当前帧 的特征点位置从网络大小缩放回原图像大小，以便后续处理。
                cur_pts.resize(corners2.size());
                for (int i = 0; i < int(corners2.size()); i++)
                {
                    cur_pts[i].x = corners2[i].x * k_w; // 将计算后的点的x坐标缩放回原图大小
                    cur_pts[i].y = corners2[i].y * k_h; 
                }

                // subpixel refinement
                // 5. 光流计算后的点的位置进行亚像素精确化
                // 使用 cv::cornerSubPix 对光流计算后的点进行亚像素精确化。这一步会进一步提高特征点的位置精度，使得特征点的定位更加精确。
                cv::cornerSubPix(gray,
                                cur_pts,
                                cv::Size(3, 3),
                                cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                5, 0.01));


            #else
                // printf("vins has prediction\n");
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1,
                                        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),  //迭代算法的终止条件
                                        cv::OPTFLOW_USE_INITIAL_FLOW);
            #endif

            int succ_num = 0; //成功和上一帧匹配的数目
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10) // 小于10时，好像会扩大搜索，输入的基于最大金字塔层次数为3
            {
                #ifdef LET_NET
                    cv::calcOpticalFlowPyrLK(last_desc, desc, corners1, corners2, status, err, cv::Size(21, 21), 5);

                    cur_pts.resize(corners2.size());
                    for (int i = 0; i < int(corners2.size()); i++)
                    {
                        cur_pts[i].x = corners2[i].x * k_w; // 将计算后的点的x坐标缩放回原图大小
                        cur_pts[i].y = corners2[i].y * k_h; 
                    }

                    cv::cornerSubPix(gray,
                                    cur_pts,
                                    cv::Size(3, 3),
                                    cv::Size(-1, -1),
                                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                    5, 0.01));
                #else
                    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
                #endif
            }

        }
        else
        {
            #ifdef LET_NET
                // printf("LET_NET no prediction\n");

                // 2. 调整 上一帧 点的位置：
                // resize prev_pts to corners1
                // 为了将 上一帧 特征点的位置调整到与网络处理大小一致，使得在后续的光流计算中，点的位置与网络的特征图匹配。
                corners1.resize(prev_pts.size()); 
                for (int i = 0; i < int(prev_pts.size()); i++)
                {
                    corners1[i].x = prev_pts[i].x / k_w; // 将 上一帧 点的x坐标缩放到网络输入大小
                    corners1[i].y = prev_pts[i].y / k_h; // 将 上一帧 点的y坐标缩放到网络输入大小
                }

                cv::calcOpticalFlowPyrLK(last_desc, desc, corners1, corners2, status, err, cv::Size(21, 21), 5);

                cur_pts.resize(corners2.size());
                for (int i = 0; i < int(corners2.size()); i++)
                {
                    cur_pts[i].x = corners2[i].x * k_w; // 将计算后的点的x坐标缩放回原图大小
                    cur_pts[i].y = corners2[i].y * k_h; 
                }

                cv::cornerSubPix(gray,
                                cur_pts,
                                cv::Size(3, 3),
                                cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                5, 0.01));

            #else
                // printf("no prediction\n");
                // 如果没有进行预测的话，直接是基于最大金字塔层次数为3
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            #endif

        }
    
        // reverse check 方向检查
        if(FLOW_BACK)
        {
            vector<uchar> reverse_status;

            #ifdef LET_NET
                vector<cv::Point2f> reverse_pts = corners1;
                cv::calcOpticalFlowPyrLK(desc, last_desc, corners2, reverse_pts, reverse_status, err, cv::Size(21, 21), 1);
                                        // cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), 
                                        // cv::OPTFLOW_USE_INITIAL_FLOW);

                  for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(corners1[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }

            #else
                vector<cv::Point2f> reverse_pts = prev_pts;
                // 注意！这里输入的参数和上边的前后是相反的
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
                                        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), 
                                        cv::OPTFLOW_USE_INITIAL_FLOW);
                // cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 

                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            #endif

        }
        
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    if (1)
    {
        //rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            #ifdef LET_NET
                if(mask.empty())
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1)
                    cout << "mask type wrong " << endl;
                if (mask.size() != cur_img.size())
                    cout << "wrong size " << endl;

                cv::resize(mask, mask, cv::Size(LET_WIDTH, LET_HEIGHT));
                // 第一帧检测特征点
                letFeaturesToTrack(score, n_pts, MAX_CNT - cur_pts.size(), 0.0001, MIN_DIST, mask);
                int w0 = cur_img.cols; int h0 = cur_img.rows;
                float k_w = float(w0) / float(LET_WIDTH);
                float k_h = float(h0) / float(LET_HEIGHT);
                for (auto & n_pt : n_pts)
                {
                    n_pt.x *= k_w;
                    n_pt.y *= k_h;
                }
                // subpixel refinement
                cv::cornerSubPix(gray,
                                n_pts,
                                cv::Size(3, 3),
                                cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                5, 0.01));
            #else
                if(mask.empty())
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1)
                    cout << "mask type wrong " << endl;
                cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
            #endif
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        //printf("feature cnt after add %d\n", (int)ids.size());
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // ---------------如果是双目的
    if(!_img1.empty() && stereo_cam)
    // 把左目的点在右目上找到，然后计算右目上的像素速度。
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();

        // --------------------如果当前帧非空
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n");  //在右侧图像上追踪特征
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft; //左右目的状态
            vector<float> err;

            #ifdef LET_NET
                std::vector<cv::Point2f> cornersL, cornersR;
            
                cornersL.resize(cur_pts.size()); 
                for (int i = 0; i < int(cur_pts.size()); i++)
                {
                    cornersL[i].x = cur_pts[i].x / k_w; // 将 上一帧 点的x坐标缩放到网络输入大小
                    cornersL[i].y = cur_pts[i].y / k_h; // 将 上一帧 点的y坐标缩放到网络输入大小
                }

                cv::calcOpticalFlowPyrLK(desc, rightDesc, cornersL, cornersR, status, err, cv::Size(21, 21), 5);
                cur_right_pts.resize(cornersR.size());
                for (int i = 0; i < int(cornersR.size()); i++)
                {
                    cur_right_pts[i].x = cornersR[i].x * k_w; // 将计算后的点的x坐标缩放回原图大小
                    cur_right_pts[i].y = cornersR[i].y * k_h; 
                }
                // subpixel refinement
                cv::cornerSubPix(grayRight,
                                cur_right_pts,
                                cv::Size(3, 3),
                                cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                5, 0.01));

            #else
                // cur left ---- cur right
                cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
                // reverse check cur right ---- cur left
            #endif

            if(FLOW_BACK)
            {
                #ifdef LET_NET
                    cv::calcOpticalFlowPyrLK(rightDesc, desc, cornersR, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cornersL[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }

                #else
                    cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                #endif
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}

// 使用 LET-Net 模型和掩码检测特征。
void FeatureTracker::letFeaturesToTrack(cv::InputArray image,
                                             cv::OutputArray _corners,
                                             int maxCorners,
                                             double qualityLevel,
                                             double minDistance,
                                             cv::InputArray _mask, int blockSize)
{
    // 确保输入参数的有效性
    CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
    CV_Assert(_mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(image)));

    cv::Mat eig = image.getMat(), tmp;
    double maxVal = 0;

    // 计算图像中的最小值和最大值（用于后续的阈值处理）
    cv::minMaxLoc(eig, 0, &maxVal, 0, 0, _mask);
    // 阈值处理，将低于 maxVal * qualityLevel 的值置为零
    cv::threshold(eig, eig, maxVal * qualityLevel, 0, cv::THRESH_TOZERO);
    // 膨胀操作，方便后续的极大值检测
    cv::dilate(eig, tmp, cv::Mat());

    cv::Size imgsize = eig.size();
    std::vector<const float*> tmpCorners;

    // 将 mask 转换为 Mat 格式
    cv::Mat Mask = _mask.getMat();

    // 遍历图像的每一个像素
    for( int y = 1; y < imgsize.height - 1; y++ )
    {
        const float* eig_data = (const float*)eig.ptr(y);
        const float* tmp_data = (const float*)tmp.ptr(y);
        const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

        for( int x = 1; x < imgsize.width - 1; x++ )
        {
            float val = eig_data[x];
            // 如果该像素满足一定条件（非零值，等于膨胀后的值，且满足掩码条件），则将其作为候选特征点。
            if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
                tmpCorners.push_back(eig_data + x);
        }
    }

    std::vector<cv::Point2f> corners;
    std::vector<float> cornersQuality;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if (total == 0)
    {
        _corners.release();
        return;
    }

    // 将候选点按质量从大到小排序
    std::sort( tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

    // 如果 minDistance 大于等于 1，将图像划分为网格，确保特征点之间的最小距离。
    if (minDistance >= 1)
    {
        // Partition the image into larger grids
        // 将图像划分为更大的网格
        int w = eig.cols;
        int h = eig.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size - 1) / cell_size;
        const int grid_height = (h + cell_size - 1) / cell_size;

        std::vector<std::vector<cv::Point2f> > grid(grid_width*grid_height);

        minDistance *= minDistance;

        // 遍历候选特征点，检查其与网格中的其他特征点之间的距离，如果满足最小距离条件，则将其保留。
        for( i = 0; i < total; i++ )
        {
            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell - 1;
            int y1 = y_cell - 1;
            int x2 = x_cell + 1;
            int y2 = y_cell + 1;

            // boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            for( int yy = y1; yy <= y2; yy++ )
            {
                for( int xx = x1; xx <= x2; xx++ )
                {
                    std::vector <cv::Point2f> &m = grid[yy*grid_width + xx];

                    if( m.size() )
                    {
                        for(j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if( dx*dx + dy*dy < minDistance )
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if (good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                cornersQuality.push_back(*tmpCorners[i]);

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if( maxCorners > 0 && (int)ncorners == maxCorners )
                    break;
            }
        }
    }
    else
    {
        for( i = 0; i < total; i++ )
        {
            cornersQuality.push_back(*tmpCorners[i]);

            int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
            int y = (int)(ofs / eig.step);
            int x = (int)((ofs - y*eig.step)/sizeof(float));

            corners.push_back(cv::Point2f((float)x, (float)y));
            ++ncorners;

            if( maxCorners > 0 && (int)ncorners == maxCorners )
                break;
        }
    }
    // 将选定的特征点转换为输出格式
    cv::Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}

// 初始化用于特征检测和描述的 LET-Net 模型。
void FeatureTracker::let_init(){
    score = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_32FC1);
    desc = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    last_desc = cv::Mat(LET_HEIGHT, LET_WIDTH, CV_8UC3);

    // gray
    net.load_param("/workspace/vins_letnet_ws/src/VINS-Mono-LET-Net/model/letnet-gray-opt.param");
    net.load_model("/workspace/vins_letnet_ws/src/VINS-Mono-LET-Net/model/letnet-gray-opt.bin");

    // // BGR
    // net.load_param("/workspace/vins_letnet_ws/src/VINS-Mono-LET-Net/model/model.param");
    // net.load_model("/workspace/vins_letnet_ws/src/VINS-Mono-LET-Net/model/model.bin");
}

cv::Mat FeatureTracker::let_net(const cv::Mat& image_bgr) {
    last_desc = desc.clone();

    // 将 BGR 图像转换为灰度图像
    cv::Mat img_gray;
    cv::cvtColor(image_bgr, img_gray, cv::COLOR_BGR2GRAY);

    // 调整灰度图像大小
    cv::Mat img_resized;
    cv::resize(img_gray, img_resized, cv::Size(LET_WIDTH, LET_HEIGHT));

    // 创建 NCNN 提取器
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

     // 将灰度图像转换为 NCNN Mat 格式
    in = ncnn::Mat::from_pixels(img_resized.data, ncnn::Mat::PIXEL_GRAY, img_resized.cols, img_resized.rows);
    in.substract_mean_normalize(mean_vals_gray, norm_vals_gray);

    // 提取特征
    ex.input("input", in);
    ex.extract("score", out1);
    ex.extract("descriptor", out2);

    // 逆归一化处理
    out1.substract_mean_normalize(mean_vals_inv_gray, norm_vals_inv_gray);
    out2.substract_mean_normalize(mean_vals_inv_gray, norm_vals_inv_gray);

    // 将 NCNN Mat 转换为 OpenCV Mat
    memcpy((uchar*)score.data, out1.data, LET_HEIGHT*LET_WIDTH*sizeof(float));
    cv::Mat desc_tmp(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    out2.to_pixels(desc_tmp.data, ncnn::Mat::PIXEL_BGR);
    desc = desc_tmp.clone();
    
    // // 保存描述子图像
    // cv::imwrite("desc.png", desc);
    return desc;
}

cv::Mat FeatureTracker::let_net_right(const cv::Mat& image_bgr) {

    // 将 BGR 图像转换为灰度图像
    cv::Mat img_gray;
    cv::cvtColor(image_bgr, img_gray, cv::COLOR_BGR2GRAY);

    // 调整灰度图像大小
    cv::Mat img_resized;
    cv::resize(img_gray, img_resized, cv::Size(LET_WIDTH, LET_HEIGHT));

    // 创建 NCNN 提取器
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

     // 将灰度图像转换为 NCNN Mat 格式
    in = ncnn::Mat::from_pixels(img_resized.data, ncnn::Mat::PIXEL_GRAY, img_resized.cols, img_resized.rows);
    in.substract_mean_normalize(mean_vals_gray, norm_vals_gray);

    // 提取特征
    ex.input("input", in);
    ex.extract("score", out1);
    ex.extract("descriptor", out2);

    // 逆归一化处理
    out1.substract_mean_normalize(mean_vals_inv_gray, norm_vals_inv_gray);
    out2.substract_mean_normalize(mean_vals_inv_gray, norm_vals_inv_gray);

    // 将 NCNN Mat 转换为 OpenCV Mat
    memcpy((uchar*)score.data, out1.data, LET_HEIGHT*LET_WIDTH*sizeof(float));
    cv::Mat desc_tmp(LET_HEIGHT, LET_WIDTH, CV_8UC3);
    out2.to_pixels(desc_tmp.data, ncnn::Mat::PIXEL_BGR);
    cv::Mat descRight = desc_tmp.clone();
    
    return descRight;
}