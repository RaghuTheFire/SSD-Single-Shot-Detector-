// bytetrack.cpp (enhanced with Kalman filtering and velocity estimation)
#include "bytetrack.h"
#include <opencv2/video/tracking.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

int global_track_id = 0;

float computeIoU(const Rect& a, const Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);
    int interArea = max(0, x2 - x1) * max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;
    return interArea > 0 ? static_cast<float>(interArea) / unionArea : 0.0f;
}

KalmanFilter ByteTrack::createKalmanFilter(const Rect& bbox) {
    KalmanFilter kf(4, 2, 0); // [x, y, dx, dy] → [x, y]
    kf.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);
    kf.measurementMatrix = Mat::eye(2, 4, CV_32F);
    setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, Scalar::all(1));
    kf.statePost.at<float>(0) = bbox.x + bbox.width / 2;
    kf.statePost.at<float>(1) = bbox.y + bbox.height / 2;
    kf.statePost.at<float>(2) = 0;
    kf.statePost.at<float>(3) = 0;
    return kf;
}

Point ByteTrack::predictCenter(KalmanFilter& kf) {
    Mat prediction = kf.predict();
    return Point(prediction.at<float>(0), prediction.at<float>(1));
}

ByteTrack::ByteTrack(int max_age, float iou_thresh)
    : max_age(max_age), iou_threshold(iou_thresh) {}

vector<Track> ByteTrack::update(const vector<Detection>& detections) {
    vector<Track> updated_tracks;
    vector<bool> matched(detections.size(), false);

    for (auto& track : tracks) {
        Point predicted = predictCenter(track.kf);
        float best_iou = 0;
        int best_idx = -1;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (matched[i]) continue;
            Point center(detections[i].bbox.x + detections[i].bbox.width / 2,
                         detections[i].bbox.y + detections[i].bbox.height / 2);
            Rect predicted_box(center.x - track.bbox.width / 2,
                               center.y - track.bbox.height / 2,
                               track.bbox.width, track.bbox.height);
            float iou = computeIoU(predicted_box, detections[i].bbox);
            if (iou > best_iou && iou > iou_threshold) {
                best_iou = iou;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx != -1) {
            const Detection& det = detections[best_idx];
            Point meas(det.bbox.x + det.bbox.width / 2, det.bbox.y + det.bbox.height / 2);
            Mat measurement = (Mat_<float>(2,1) << meas.x, meas.y);
            track.kf.correct(measurement);
            Point new_center = predictCenter(track.kf);
            track.velocity = new_center - Point(track.bbox.x + track.bbox.width/2, track.bbox.y + track.bbox.height/2);
            track.bbox = Rect(new_center.x - det.bbox.width / 2, new_center.y - det.bbox.height / 2,
                              det.bbox.width, det.bbox.height);
            track.class_id = det.class_id;
            track.age = 0;
            updated_tracks.push_back(track);
            matched[best_idx] = true;
        } else {
            track.age++;
            if (track.age < max_age)
                updated_tracks.push_back(track);
        }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matched[i]) {
            Track new_track;
            new_track.track_id = global_track_id++;
            new_track.bbox = detections[i].bbox;
            new_track.class_id = detections[i].class_id;
            new_track.age = 0;
            new_track.kf = createKalmanFilter(detections[i].bbox);
            new_track.velocity = Point(0, 0);
            updated_tracks.push_back(new_track);
        }
    }

    tracks = updated_tracks;
    return tracks;
}
