#include <numeric>
#include "matching2D.hpp"

using namespace std;

//#define WRITE_IMAGE

// Compute distribution if keypoints neighborhood MP.7
void GetDistribution(const vector<cv::KeyPoint>& keypoints, float& mean, float& rms){
    float sum_neigh_size = accumulate(keypoints.begin(), keypoints.end(), 0.0f, [](float a, const cv::KeyPoint& kp){return a + kp.size;});
    mean = sum_neigh_size / keypoints.size();
    float sum_sq_diff = accumulate(keypoints.begin(), keypoints.end(), 0.0f,
    [&mean](float a, const cv::KeyPoint& kp){return a + powf(kp.size-mean, 2.0f);});
    rms = sqrtf(sum_sq_diff)/keypoints.size();
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0? cv::NORM_HAMMING: cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matcher uses ";
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "MAT FLANN matcher uses ";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "selector NN with n=" << matches.size() << " matches in t="<< 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        std::vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        double min_desc_ratio = 0.8;
        for (auto i: knn_matches){
          if (i[0].distance < i[1].distance*min_desc_ratio){
            matches.push_back(i[0]);
          }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "selector KNN with n=" << matches.size() << " matches in t=" << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        cout << "BRISK descriptor extractor: ";
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0) /*ORB, FREAK, AKAZE, SIFT*/
    {
        cout << "BRIEF descriptor extractor: ";
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType.compare("ORB") == 0){
        cout << "ORB descriptor extractor: ";
        extractor = cv::ORB::create();
    } else if (descriptorType.compare("FREAK") == 0){
        cout << "FREAK descriptor extractor: ";
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType.compare("AKAZE") == 0){
        cout << "AKAZE descriptor extractor: ";
        extractor = cv::AKAZE::create();
    } else if (descriptorType.compare("SIFT") == 0){
        cout << "SIFT descriptor extractor: ";
        extractor = cv::SIFT::create();
    } else {
        throw "Invalid descriptpr type";
    }
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "n=" << descriptors.rows << " in t=" << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detector: n=" << keypoints.size() << " keypoints in t=" << 1000 * t / 1.0 << " ms";

    // Compute distribution if keypoints neighborhood MP.7
    float mean, rms;
    GetDistribution(keypoints, mean, rms);
    cout << " m1=" << mean << " rms=" << rms << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
#if  defined(WRITE_IMAGE)
        static int img_num = 0;
        cv::imwrite(string("Shi_Tomasi_") + to_string(img_num++) + string(".jpg"), visImage);
#endif
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    int block_size = 2;
    int aperture_size = 3;
    double k = 0.04;
    int threshold = 100;
    double overlap = 0;

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, block_size, aperture_size, k, cv::BORDER_DEFAULT);
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (int i = 0; i < dst_norm_scaled.rows; i++){
        for (int j = 0; j < dst_norm_scaled.cols; j++){
            cv::KeyPoint cur_key_point(j, i, 2*aperture_size, -1, dst_norm_scaled.at<char>(i, j));
            if (cur_key_point.response >= threshold){
                bool to_add = true;
                for (auto kp: keypoints) {
                    if (cv::KeyPoint::overlap(cur_key_point, kp) > overlap){
                        if (cur_key_point.response>kp.response)
                            kp = cur_key_point;
                        to_add = false;
                        break;
                    }
                }
                if (to_add)
                    keypoints.push_back(cur_key_point);
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detector: n=" << keypoints.size() << " keypoints in t=" << 1000 * t / 1.0 << " ms";

    // Compute distribution if keypoints neighborhood MP.7
    float mean, rms;
    GetDistribution(keypoints, mean, rms);
    cout << " m1=" << mean << " rms=" << rms << endl;

    if (bVis){
        std::string window_name = "Harris corner detector";
        cv::namedWindow(window_name, 6);
        cv::Mat vis_image = img.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, vis_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(window_name, vis_image);
        cv::waitKey(0);
#if  defined(WRITE_IMAGE)
        static int img_num = 0;
        cv::imwrite(string("HARRIS_") + to_string(img_num++) + string(".jpg"), vis_image);
#endif
    }
};

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0){
        cout << "FAST detector: ";
        detector = cv::FastFeatureDetector::create();
    }
    else if (detectorType.compare("BRISK") == 0){
        cout << "BRISK detector: ";
        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("ORB") == 0){
        cout << "ORB detector: ";
        detector = cv::ORB::create();
    }
    else if (detectorType.compare("AKAZE") == 0){
        cout << "AKAZE detector: ";
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0){
        cout << "SIFT detector: ";
        detector = cv::SIFT::create();
    }
    else
        throw "Invalid detector type";

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "n=" << keypoints.size() << " keypoints in t=" << 1000 * t / 1.0 << " ms";

    // Compute distribution if keypoints neighborhood MP.7
    float mean, rms;
    GetDistribution(keypoints, mean, rms);
    cout << " m1=" << mean << " rms=" << rms << endl;

    if (bVis){
        std::string window_name = detectorType + " feature detector";
        cv::namedWindow(window_name, 6);
        cv::Mat vis_image = img.clone();
        cv::drawKeypoints(img, keypoints, vis_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(window_name, vis_image);
        cv::waitKey(0);
#if  defined(WRITE_IMAGE)
        static int img_num = 0;
        cv::imwrite(string(detectorType + "_") + to_string(img_num++) + string(".jpg"), vis_image);
#endif
    }
};
