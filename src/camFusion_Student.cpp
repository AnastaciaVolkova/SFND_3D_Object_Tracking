
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(vector<BoundingBox> &boundingBoxes, vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
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
void show3DObjects(vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, const string& save_dir, bool bWait)
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
#if defined(SAVE)
{
        static int frame_num = 0;
        ostringstream oss;
        oss << save_dir << "/3d_objects_" << setfill('0') << setw(2) << frame_num << ".png";
        cv::imwrite(oss.str(), topviewImg);
        frame_num++;
}
#else
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
#endif
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, vector<cv::DMatch> &kptMatches)
{
    // Indices of keypoints and distances to corresponding previous ones.
    using IxDist = vector<tuple<int, double>>;
    IxDist ix_dist;

    // Remember keypoints which are inside roi.
    for (int m = 0; m < kptMatches.size(); m++){
        if (boundingBox.roi.contains(kptsCurr[kptMatches[m].trainIdx].pt)){
            ix_dist.push_back(
                make_tuple(m,
                sqrt(
                pow(kptsCurr[kptMatches[m].trainIdx].pt.x-kptsPrev[kptMatches[m].queryIdx].pt.x, 2.0) +
                pow(kptsCurr[kptMatches[m].trainIdx].pt.y-kptsPrev[kptMatches[m].queryIdx].pt.y, 2.0)
                ))
            );
        }
    }

    // Sort distances.
    sort(ix_dist.begin(), ix_dist.end(), [](auto a, auto b){return get<1>(a) < get<1>(b);});

    // Find most frequent distance range with help of histogram.
    double d = (get<1>(*(ix_dist.end()-1)) - get<1>(*ix_dist.begin()))/10; // Historgram step.
    double s = d;
    auto prev = ix_dist.begin();
    vector<tuple<IxDist::iterator, IxDist::iterator>> hist;
    for (int i = 0; i < 10; i++){
        auto x = upper_bound(ix_dist.begin(), ix_dist.end(), s, [](auto a, auto b){return a < get<1>(b); });
        hist.push_back(make_tuple(prev, x));
        s += d;
        prev = x;
    }

    // Get bin with maximum number of elements.
    auto it = max_element(hist.begin(), hist.end(),
    [](auto a, auto b){return distance(get<0>(a),get<1>(a)) < distance(get<0>(b),get<1>(b));});

    auto it_dec = it;
    auto it_inc = it;

    int total_el = distance(get<0>(*it),get<1>(*it));

    // Regards left-right bins.
    // Search until sum of bins reaches half of elements number.
    while (total_el < ix_dist.size()/2){
        auto a=it, b=it;
        if (*it_dec > *it_inc){
            a = it_dec;
            if (it_dec > hist.begin())
                it_dec--;
        }
        else if (*it_dec < *it_inc){
            b = it_inc;
            if (it < hist.end()-1)
                it_inc++;
        }
        else {
            a = it_dec;
            b = it_dec;
            if (it_dec > hist.begin())
                it_dec--;
            if (it < hist.end()-1)
                it_inc++;
        };
        total_el = accumulate(it_dec, it_inc+1, 0.0, [](auto a, auto v){return a + distance(get<0>(v),get<1>(v));});
    }

    // Take into account most popular values.
    auto lo = get<0>(*it_dec);
    auto hi = get<1>(*it_inc);

    // Save only matches, which keypoints distances are in [Q1;Q3]
    for (auto it = lo; it < hi; it++)
        boundingBox.kptMatches.push_back(kptMatches[get<0>(*it)]);
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr,
                      vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // 1. Compute ratios of distance between every point of current and previous frames.
    vector<float> ratios; // Ratios of distances between points of previous and current frames.
    double minDist = 100.0;
    for (int i = 0; i < kptMatches.size(); i++){
        cv::Point2f kp_prev_i = kptsPrev[kptMatches[i].queryIdx].pt;
        cv::Point2f kp_curr_i = kptsCurr[kptMatches[i].trainIdx].pt;
        for (int j = i+1; j < kptMatches.size(); j++){
            cv::Point2f kp_prev_j = kptsPrev[kptMatches[j].queryIdx].pt;
            cv::Point2f kp_curr_j = kptsCurr[kptMatches[j].trainIdx].pt;
            float d_prev = cv::norm(kp_prev_i-kp_prev_j);
            float d_curr = cv::norm(kp_curr_i-kp_curr_j);
            if (d_curr > minDist)
                ratios.push_back(d_prev/d_curr);
        }
    }

    //2. Get rid of outliers.
    sort(ratios.begin(), ratios.end());

    vector<float>::iterator lo, hi;
    double q1, q3, irq;

    // Decide wheather object is approaching or moving away.
    auto i = upper_bound(ratios.begin(), ratios.end(), 1.0);

    if (distance(ratios.begin(), i) >= distance(i, ratios.end()))
        ratios.erase(i, ratios.end());
    else
        ratios.erase(ratios.begin(), i);

    if (ratios.size()<10){
        q1 = ratios.front();
        q3 = ratios.back();
    } else {
        q3 = *(ratios.end() - ratios.size()/4);
        q1 = *(ratios.begin() + ratios.size()/4);
    }

    irq = q3-q1;

#if defined(SAVE)
    cout << "Camera: irq=" << irq << endl;
    cout << "Camera: pts_num=" << kptMatches.size() << endl;
#endif
    double avg_ratio;
    if (irq == 0)
        avg_ratio = q1;
    else{
        // Average over ratios  which are compliant to Interquartile Rule.
        lo = upper_bound(ratios.begin(), ratios.end(), q1 - 1.5*irq);
        hi = upper_bound(ratios.begin(), ratios.end(), q3 + 1.5*irq)-1;
        avg_ratio = accumulate(lo, hi, 0.0)/distance(lo, hi);
    }

    // r = s1/s2 - ratio between distance of 2 points of previous frame and distance of 2 points of current frame.
    // r = d1/d2 - d1 - distance to object on previous frame, d2 - distance to object on current frame.
    // TTC = d2/v, v=(d2-d1)/d_t.
    // TTC = d2*dt/(d2-d1) = dt/(1-d1/d2) = dt / (1-r)
    double d_t = 1/frameRate; // Get time delta from frame rate.

    if (avg_ratio == 1)
        TTC=NAN;
    else{
        TTC = d_t/(1-avg_ratio);
        if (TTC<=0) TTC = -TTC; // Avoid negative time if object is moving away.
    }
}


void computeTTCLidar(vector<LidarPoint> &lidarPointsPrev,
                     vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // 1. Compute velocity.
    // 1.1. Compute distance changes between corresponding lidar points of current and previous frames.
    vector<double> d_dist;
    for (int i = 0; i < min(lidarPointsCurr.size(), lidarPointsPrev.size()); i++)
        d_dist.push_back(sqrt(
            pow(lidarPointsCurr[i].x-lidarPointsPrev[i].x, 2)+
            pow(lidarPointsCurr[i].y-lidarPointsPrev[i].y, 2)));

    // 1.2 Average over distance changes which are compliant to Interquartile Rule.
    sort(d_dist.begin(), d_dist.end());

    double q3 = *(d_dist.end() - d_dist.size()/4);
    double q1 = *(d_dist.begin() + d_dist.size()/4);
    double irq = q3-q1;
#if defined(SAVE)
    cout << "Lidar: irq=" << irq << endl;
#endif

    auto hi = upper_bound(d_dist.begin(), d_dist.end(), 1.5*irq + q3)-1;
    auto lo = upper_bound(d_dist.begin(), d_dist.end(), q1 - 1.5*irq);
    double avg_dist = accumulate(lo, hi, 0.0)/distance(lo, hi);

    double d_t = 1/frameRate; // Get delta from frame rate.

    double v = (avg_dist)/d_t; // Compute velocity.

    // Take into account distances which are compliant to Interquartile Rule.
    // 2.1. Compute distances to objects on current frame.
    d_dist.clear();
    transform(
        lidarPointsCurr.begin(),
        lidarPointsCurr.end(),
        back_inserter(d_dist),
        [](auto a){return sqrt(pow(a.x, 2) + pow(a.y, 2));}
    );
    // 2.2. Average over distances which are compliant to Interquartile Rule.
    sort(d_dist.begin(), d_dist.end());
    q3 = *(d_dist.end() - d_dist.size()/4);
    q1 = *(d_dist.begin() + d_dist.size()/4);
    irq = q3-q1;
    hi = upper_bound(d_dist.begin(), d_dist.end(), 1.5*irq + q3)-1;
    lo = upper_bound(d_dist.begin(), d_dist.end(), q1 - 1.5*irq);
    avg_dist = accumulate(lo, hi, 0.0)/distance(lo, hi);
    TTC = avg_dist/v;
}

void matchBoundingBoxes(vector<cv::DMatch> &matches, map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    map<int, map<int, vector<cv::DMatch>>> freq;
    vector<cv::DMatch> matches_un(matches);

    // Get rid of keypoints which are in the same bounding boxes.
    vector<cv::DMatch>::iterator it = matches_un.begin();

    while(it != matches_un.end()) {
        int n = count_if(
            prevFrame.boundingBoxes.begin(),
            prevFrame.boundingBoxes.end(),
            [&](auto a){return a.roi.contains(prevFrame.keypoints[it->queryIdx].pt);});

        int m = count_if(
            currFrame.boundingBoxes.begin(),
            currFrame.boundingBoxes.end(),
            [&](auto a){return a.roi.contains(currFrame.keypoints[it->trainIdx].pt);});

        if ((n>1)||(m>1))
            it = matches_un.erase(it);
        else
            it++;
    }

    for (auto m: matches_un){
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

    vector<tuple<int, int, vector<cv::DMatch>>> freq_v;
    for (auto i: freq)
        for (auto j: i.second)
            freq_v.push_back(make_tuple(i.first, j.first, j.second));

    sort(freq_v.begin(), freq_v.end(), [](auto a, auto b){return get<2>(a).size() > get<2>(b).size();});

    auto freq_v_it = freq_v.begin();
    while (freq_v_it != freq_v.end()){
        int i = get<0>(*freq_v_it);
        int j = get<1>(*freq_v_it);

        while(true) {
            auto f = find_if(freq_v_it+1, freq_v.end(),
            [&i,&j](auto a){return (get<0>(a) == i) || (get<1>(a) == j);});
            if (f == freq_v.end()) {
                freq_v_it++;
                break;
            };
            freq_v.erase(f);
        };
    }

    for (auto f: freq_v){
        bbBestMatches[get<0>(f)] = get<1>(f);
        //currFrame.boundingBoxes[get<1>(f)].kptMatches = get<2>(f);
        //for (auto m: get<2>(f))
        //    currFrame.boundingBoxes[get<1>(f)].keypoints.push_back(currFrame.keypoints[m.trainIdx]);
    }
}
