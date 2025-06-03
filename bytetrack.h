// bytetrack.h (enhanced with Kalman filtering + velocity smoothing)
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

struct Detection {
    cv::Rect bbox;
    int class_id;
    float confidence;
};

struct Track {
    int track_id;
    cv::Rect bbox;
    int class_id;
    int age;
    cv::KalmanFilter kf;
    cv::Point velocity;           // Instantaneous velocity
    cv::Point smoothed_velocity; // Smoothed velocity using EMA
};

class ByteTrack {
public:
    ByteTrack(int max_age = 30, float iou_thresh = 0.3);
    std::vector<Track> update(const std::vector<Detection>& detections);

private:
    std::vector<Track> tracks;
    int max_age;
    float iou_threshold;
    float smoothing_factor = 0.5f; // EMA factor (0.0–1.0)

    cv::KalmanFilter createKalmanFilter(const cv::Rect& bbox);
    cv::Point predictCenter(cv::KalmanFilter& kf);
};
