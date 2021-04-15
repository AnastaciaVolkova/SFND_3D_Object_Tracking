
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/*
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Indices of keypoints and distances to corresponding previous ones.
    std::vector<std::tuple<int, double>> ix_dist;

    // Remember keypoints which are inside roi.
    for (int m = 0; m < kptMatches.size(); m++){
        if (boundingBox.roi.contains(kptsCurr[kptMatches[m].trainIdx].pt)){
            ix_dist.push_back(
                std::make_tuple(m,
                sqrt(
                pow(kptsCurr[kptMatches[m].trainIdx].pt.x-kptsPrev[kptMatches[m].queryIdx].pt.x, 2.0) +
                pow(kptsCurr[kptMatches[m].trainIdx].pt.y-kptsPrev[kptMatches[m].queryIdx].pt.y, 2.0)
                ))
            );
        }
    }

    // Sort distances.
    std::sort(ix_dist.begin(), ix_dist.end(), [](auto a, auto b){return std::get<1>(a) < std::get<1>(b);});

    // Save only matches, which keypoints distances are in [Q1;Q3]
    for (auto it = ix_dist.begin() + ix_dist.size()/4; it < (ix_dist.end() - ix_dist.size()/4); it++)
        boundingBox.kptMatches.push_back(kptMatches[std::get<0>(*it)]);
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // 1. Compute velocity.
    // 1.1. Compute distance changes between corresponding lidar points of current and previous frames.
    std::vector<double> d_dist;
    std::transform(
        lidarPointsCurr.begin(),
        lidarPointsCurr.end(),
        lidarPointsPrev.begin(),
        std::back_inserter(d_dist),
        [](auto c, auto p){return sqrt(pow(c.x-p.x, 2)+pow(c.y-p.y, 2)+pow(c.z-p.z, 2)); });

    // 1.2 Average over distance changes which are between Q1 and Q3.
    std::sort(d_dist.begin(), d_dist.end());
    double avg_dist = std::accumulate(d_dist.begin() + d_dist.size()/4, d_dist.end() - d_dist.size()/4, 0.0);
    avg_dist /= (d_dist.size()/2);

    double d_t = 1/frameRate; // Get delta from frame rate.

    double v = (avg_dist)/d_t; // Compute velocity.

    // Take into account distances which are between Q1 and Q3.
    // 2.1. Compute distances to objects on current frame.
    d_dist.clear();
    std::transform(
        lidarPointsCurr.begin(),
        lidarPointsCurr.end(),
        std::back_inserter(d_dist),
        [](auto a){return sqrt(pow(a.x, 2) + pow(a.y, 2) + pow(a.z, 2));}
    );
    // 2.2. Average over distances which are between Q1 and Q3.
    std::sort(d_dist.begin(), d_dist.end());
    avg_dist = std::accumulate(d_dist.begin() + d_dist.size()/4, d_dist.end() - d_dist.size()/4, 0.0);
    avg_dist /= (d_dist.size()/2);
    TTC = avg_dist/v;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::map<int, std::map<int, std::vector<cv::DMatch>>> freq;

    for (auto m: matches){
        for (auto bb_prev: prevFrame.boundingBoxes){
            // Check if train point of match is inside current bounding box of previous frame.
            if (bb_prev.roi.contains(prevFrame.keypoints[m.queryIdx].pt)){
                for (auto bb_curr: currFrame.boundingBoxes){
                    // Check if query point is inside bounding box of current frame.
                    if (bb_curr.roi.contains(currFrame.keypoints[m.trainIdx].pt)){
                        freq[bb_prev.boxID][bb_curr.boxID].push_back(m);
                    }
                }
            }
        }
    }

    for (auto pr = freq.begin(); pr != freq.end(); pr++){
        auto it = std::max_element(std::begin(pr->second), std::end(pr->second), [](auto a, auto b){return a.second.size() < b.second.size();});
        bbBestMatches[pr->first] = it->first;
        currFrame.boundingBoxes[it->first].kptMatches = it->second;
        for (auto m: it->second){
            currFrame.boundingBoxes[it->first].keypoints.push_back(currFrame.keypoints[m.trainIdx]);
        }
    }
}
