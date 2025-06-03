// bytetrack.h
#pragma once
#include <opencv2/core.hpp>
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
};

class ByteTrack {
public:
    ByteTrack(int max_age = 30, float iou_thresh = 0.3);
    std::vector<Track> update(const std::vector<Detection>& detections);

private:
    std::vector<Track> tracks;
    int max_age;
    float iou_threshold;
};
