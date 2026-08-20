#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;
// 裁剪黑边
Rect cropBlackBorder(const Mat& img) {
    if(img.empty()) return Rect();
    int min_x = img.cols, max_x = 0;
    int min_y = img.rows, max_y = 0;
    bool found = false;
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            Vec3b p = img.at<Vec3b>(y, x);
            if(p[0] > 5 || p[1] > 5 || p[2] > 5) {
                min_x = min(min_x, x);
                max_x = max(max_x, x);
                min_y = min(min_y, y);
                max_y = max(max_y, y);
                found = true;
            }
        }
    }
    if(!found) return Rect(0, 0, img.cols, img.rows);
    int padding = 5;
    min_x = max(0, min_x - padding);
    min_y = max(0, min_y - padding);
    max_x = min(img.cols - 1, max_x + padding);
    max_y = min(img.rows - 1, max_y + padding);
    return Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
}
// 计算两张图的匹配点数
int countMatches(const Mat& img1, const Mat& img2) {
    Ptr<ORB> orb = ORB::create(250);
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    // 特征提取，论文中SIFT和ORB的相同点是都提取关键点和描述子
    //不同点是SIFT有尺度不变性，ORB更快
    orb->detectAndCompute(img1, Mat(), kp1, desc1);
    orb->detectAndCompute(img2, Mat(), kp2, desc2);
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    // 特征匹配，论文中k-d tree和knnMatch的相同点都是用Lowe's比率
    //不同点是k-d tree 速度更快，我的BFMatcher暴力匹配，虽然速度慢但是更准，我的特征点是250,所以也能接受
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);
    float ratio = 0.75f;
    int good = 0;
    for(auto& m : knn_matches) {
        if(m[0].distance < ratio * m[1].distance) {
            good++;
        }
    }
    return good;
}
// 自动排序
vector<Mat> autoSortImages(vector<Mat>& images) {
    int n = images.size();
    if(n <= 1) return images;
    // 计算每对图片的匹配点数
    vector<vector<int>> match_matrix(n, vector<int>(n, 0));
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            int m = countMatches(images[i], images[j]);
            match_matrix[i][j] = m;
            match_matrix[j][i] = m;
        }
    }
    // 找到匹配点数最多的一对作为起点
    int start_i = 0, start_j = 1, max_match = 0;
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            if(match_matrix[i][j] > max_match) {
                max_match = match_matrix[i][j];
                start_i = i;
                start_j = j;
            }
        }
    }
    // 确定哪张是“第一张”
    //与第三张匹配点数更多的那个是第二张，另一个是第一张
    int first = start_i, second = start_j;
    for(int k = 0; k < n; k++) {
        if(k == start_i || k == start_j) continue;
        if(match_matrix[start_j][k] > match_matrix[start_i][k]) {
            first = start_i;
            second = start_j;
        }else {
            first = start_j;
            second = start_i;
        }
        break;
    }
    // 按顺序排列
    vector<Mat> sorted;
    vector<bool> used(n, false);
    sorted.push_back(images[first]);
    used[first] = true;
    sorted.push_back(images[second]);
    used[second] = true;
    // 依次找下一张
    while((int)sorted.size() < n) {
        int last_idx = -1;
        int best_match = -1;
        for(int i = 0; i < n; i++) {
            if(used[i]) continue;
            int m = countMatches(sorted.back(), images[i]);
            if(m > best_match) {
                best_match = m;
                last_idx = i;
            }
        }
        if(last_idx == -1) break;
        sorted.push_back(images[last_idx]);
        used[last_idx] = true;
        cout << " autoSort: " << last_idx+1 << " page,count " << best_match << endl;
    }
    cout << " autoSort OK: ";
    for(int i = 0; i < n; i++) {
        cout << (i+1) << " ";
    }
    cout << endl;
    return sorted;
}
// 渐变融合
//论文是分频率融合，我是线性融合，都是为了消除拼接缝
//论文的对高频细节的保护和对颜色过渡的平滑度更好，但是实现更复杂，我的项目只是三个图拼接所以线性融合也够用
Mat gradientBlend(const Mat& img1, const Mat& img2, const Mat& mask1, const Mat& mask2) {
    Mat result = Mat::zeros(img1.size(), img1.type());
    int overlap_left = img1.cols, overlap_right = 0;
    for(int y = 0; y < img1.rows; y++) {
        for(int x = 0; x < img1.cols; x++) {
            if(mask1.at<uchar>(y, x) > 0 && mask2.at<uchar>(y, x) > 0) {
                overlap_left = min(overlap_left, x);
                overlap_right = max(overlap_right, x);
            }
        }
    }
    if(overlap_right <= overlap_left) {
        img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));
        img2.copyTo(result(Rect(img1.cols, 0, img2.cols, img2.rows)));
        return result;
    }
    int overlap_width = overlap_right - overlap_left + 1;
    for(int y = 0; y < img1.rows; y++) {
        for(int x = 0; x < img1.cols; x++) {
            Vec3b p1 = img1.at<Vec3b>(y, x);
            Vec3b p2 = img2.at<Vec3b>(y, x);
            bool in1 = (mask1.at<uchar>(y, x) > 0);
            bool in2 = (mask2.at<uchar>(y, x) > 0);
            if(in1 && in2) {
                float alpha = float(x - overlap_left) / overlap_width;
                float w1 = 1.0f - alpha;
                float w2 = alpha;
                result.at<Vec3b>(y, x) = p1 * w1 + p2 * w2;
            }else if(in1) {
                result.at<Vec3b>(y, x) = p1;
            }else if(in2) {
                result.at<Vec3b>(y, x) = p2;
            }
        }
    }
    return result;
}

//  拼接两张图
Mat stitchTwoImages(const Mat& img1, const Mat& img2) {
    Ptr<ORB> orb = ORB::create(250);
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    // 数学本质：将图像从像素空间映射到高维特征空间（描述子）
    // ORB 描述子 = 局部图像块的二进制指纹，用于后续相似度度量
    orb->detectAndCompute(img1, Mat(), kp1, desc1);
    orb->detectAndCompute(img2, Mat(), kp2, desc2);
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    vector<vector<DMatch>> knn_matches;
    // 最近邻搜索：在描述子空间中寻找距离最近的点对
    // 几何意义：在高维特征空间中找到“最相似”的局部特征
    matcher->knnMatch(desc1, desc2, knn_matches, 2);
    float ratio = 0.75f;
    vector<DMatch> good_matches;
    for(auto& m : knn_matches) {
        // Lowe's 比率测试：最近邻与次近邻的距离之比 < 0.75
        // 统计学依据：正确匹配的最近邻距离通常显著小于次近邻
        if(m[0].distance < ratio * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }
    if(good_matches.size() < 10) return Mat();
    vector<Point2f> pts1, pts2;
    for(auto& m : good_matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
    //和论文一致，都用了 RANSAC 剔除误匹配都求解 3×3 单应性矩阵 H
    //不同是论文用 SIFT 特征 + RANSAC 配准，我用 ORB 特征 + RANSAC 配准
    //论文还做了全局 Bundle Adjustment 进一步优化
    Mat H = findHomography(pts2, pts1, RANSAC, 3.0);
    // H是一个3x3的矩阵，findHomography可以找到一个矩阵H，把img2的点变换到img1的坐标里
    // 特征向量表示变换中方向不变的方向，特征值表示该方向的缩放倍数
    // SVD分解的几何意义：
    // H = U * Σ * V^T，V 旋转原始图像，Σ 做缩放，U 做最后的旋转
    // 即：任何变换 = 旋转 → 缩放 → 再旋转
    if(H.empty()) return Mat();
    int h = max(img1.rows, img2.rows);
    int w = img1.cols + img2.cols;
    Mat warped;
    // 透视变换：将 img2 所有像素按 H 矩阵映射到 img1 坐标系
    // 数学形式：p_img1 = H * p_img2（齐次坐标）
    warpPerspective(img2, warped, H, Size(w, h));
    Mat mask1 = Mat::zeros(h, w, CV_8UC1);
    Mat mask2 = Mat::zeros(h, w, CV_8UC1);
    Mat canvas1 = Mat::zeros(h, w, img1.type());
    img1.copyTo(canvas1(Rect(0, 0, img1.cols, img1.rows)));
    mask1(Rect(0, 0, img1.cols, img1.rows)).setTo(255);
    for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
            Vec3b p = warped.at<Vec3b>(y, x);
            if(p[0] > 5 || p[1] > 5 || p[2] > 5) {
                mask2.at<uchar>(y, x) = 255;
            }
        }
    }
    Mat result = gradientBlend(canvas1, warped, mask1, mask2);
    Rect roi = cropBlackBorder(result);
    if(roi.width > 0 && roi.height > 0) {
        result = result(roi);
    }
    return result;
}
//主函数
int main(int argc, char** argv) {
    vector<string> paths;
    if(argc > 1) {
        for (int i = 1; i < argc; i++) paths.push_back(argv[i]);
    }else {
        cout << " img1.jpg, img2.jpg, img3.jpg " << endl;
        paths = {
            "/home/sean/PanoramaProject/img1.jpg",
            "/home/sean/PanoramaProject/img2.jpg",
            "/home/sean/PanoramaProject/img3.jpg"
        };
    }
    vector<Mat> images;
    for(auto& p : paths) {
        Mat img = imread(p);
        if(img.empty()) {
            cerr << " false: " << p << endl;
            return -1;
        }
        images.push_back(img);
    }
    cout << " firstSort: " << endl;
    for(int i = 0; i < (int)images.size(); i++) {
        cout << "  [" << i << "] " << paths[i] << endl;
    }
    cout << " autoSort.. " << endl;
    vector<Mat> sorted = autoSortImages(images);
    Mat result = sorted[0];
    for(int i = 1; i < (int)sorted.size(); i++) {
        cout << "stitch " << i+1 << " page" << endl;
        Mat next = stitchTwoImages(result, sorted[i]);
        if(next.empty()) {
            cerr << " false " << endl;
            return -1;
        }
        result = next;
    }
    imwrite(" panorama_sorted.jpg ", result);
    cout << " OK " << endl;
    cout << " final_size: " << result.cols << " x " << result.rows << endl;
    cout << " save: panorama_sorted.jpg " << endl;

    return 0;
}