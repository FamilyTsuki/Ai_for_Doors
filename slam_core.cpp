#include <opencv2/opencv.hpp>
#include <vector>

extern "C" {
    cv::Mat old_gray;
    std::vector<cv::Point2f> p0;

    float process_frame(int width, int height, unsigned char* data, int* point_count) {
        cv::Mat frame(height, width, CV_8UC3, data);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        float total_movement = 0.0f;

        if (old_gray.empty() || p0.empty()) {
            cv::goodFeaturesToTrack(gray, p0, 100, 0.005, 10);
            gray.copyTo(old_gray);
            *point_count = (int)p0.size();
            return 0.0f;
        }

        std::vector<cv::Point2f> p1;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(old_gray, gray, p0, p1, status, err);

        std::vector<cv::Point2f> good_new;
        int tracked_count = 0;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && p1[i].x >= 0 && p1[i].x < width && p1[i].y >= 0 && p1[i].y < height) {
                good_new.push_back(p1[i]);
                total_movement += cv::norm(p1[i] - p0[i]);
                tracked_count++;
                cv::circle(frame, p1[i], 3, cv::Scalar(0, 255, 0), -1);
            }
        }

        if (good_new.size() < 80) {
            std::vector<cv::Point2f> fresh;
            cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1);
            for (const auto& pt : good_new) cv::circle(mask, pt, 15, 0, -1);
            cv::goodFeaturesToTrack(gray, fresh, 100 - (int)good_new.size(), 0.005, 10, mask);
            good_new.insert(good_new.end(), fresh.begin(), fresh.end());
        }

        p0 = good_new;
        *point_count = (int)p0.size();
        gray.copyTo(old_gray);
        return (tracked_count > 0) ? (total_movement / tracked_count) : 0.0f;
    }
}