// mobilenet_ssd_tracker.cpp (updated with RTSP, snapshots, consistent colors)
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include "bytetrack.h"

using namespace cv;
using namespace std;

bool use_gpu = true;
bool live_video = true;
string video_source = "rtsp://your_rtsp_stream";
float confidence_threshold = 0.5;
float iou_threshold = 0.3;

vector<string> CLASSES = {"background", "aeroplane", "bicycle", "bird", "boat",
                          "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                          "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                          "sofa", "train", "tvmonitor"};

vector<Vec3b> COLORS(CLASSES.size());
void generateConsistentColors() {
    for (size_t i = 0; i < COLORS.size(); ++i) {
        int r = (i * 123) % 255;
        int g = (i * 321) % 255;
        int b = (i * 231) % 255;
        COLORS[i] = Vec3b(b, g, r);
    }
}

atomic<bool> keepRunning(true);
tbb::concurrent_bounded_queue<Mat> frameQueue;

void captureFrames(VideoCapture& cap) 
{
    Mat frame;
    while (keepRunning) 
    {
        if (!cap.read(frame)) break;
        frameQueue.push(frame.clone());
    }
}

void processFrames(Net& net) 
{
    ByteTrack tracker(30, iou_threshold);
    Mat frame;
    auto start = chrono::steady_clock::now();
    int frameCount = 0;
    float fps = 0.0f;
    int snap_id = 0;

    while (keepRunning) 
    {
        if (!frameQueue.try_pop(frame)) continue;

        Mat blob;
        blobFromImage(frame, blob, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
        net.setInput(blob);
        Mat detections = net.forward();

        vector<Detection> validDetections;
        int h = frame.rows, w = frame.cols;

        for (int i = 0; i < detections.size[2]; ++i) 
        {
            float confidence = detections.at<float>(0, 0, i, 2);
            if (confidence > confidence_threshold) 
            {
                int class_id = static_cast<int>(detections.at<float>(0, 0, i, 1));
                int xLeftBottom = static_cast<int>(detections.at<float>(0, 0, i, 3) * w);
                int yLeftBottom = static_cast<int>(detections.at<float>(0, 0, i, 4) * h);
                int xRightTop = static_cast<int>(detections.at<float>(0, 0, i, 5) * w);
                int yRightTop = static_cast<int>(detections.at<float>(0, 0, i, 6) * h);
                Rect box(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

                validDetections.push_back({box, class_id, confidence});
            }
        }

        vector<Track> tracks = tracker.update(validDetections);

        for (const auto& t : tracks) 
        {
            rectangle(frame, t.bbox, COLORS[t.class_id], 2);
            string label = format("%s #%d", CLASSES[t.class_id].c_str(), t.track_id);
            putText(frame, label, Point(t.bbox.x, t.bbox.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, COLORS[t.class_id], 2);
        }

        frameCount++;
        if (frameCount % 10 == 0) 
        {
            auto end = chrono::steady_clock::now();
            float seconds = chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0;
            fps = frameCount / seconds;
            frameCount = 0;
            start = chrono::steady_clock::now();
        }

        putText(frame, format("FPS: %.2f", fps), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        imshow("SSD + Tracker", frame);
        int key = waitKey(1);
        if (key == 27) keepRunning = false;
        if (key == 's' || key == 'S') 
        {
            string filename = format("snapshot_%d.jpg", snap_id++);
            imwrite(filename, frame);
            cout << "[INFO] Snapshot saved: " << filename << endl;
        }
    }
}

int main() 
{
    generateConsistentColors();

    Net net = dnn::readNetFromCaffe("ssd_files/MobileNetSSD_deploy.prototxt",
                                    "ssd_files/MobileNetSSD_deploy.caffemodel");
    if (net.empty()) {
        cerr << "Failed to load SSD model." << endl;
        return -1;
    }
    if (use_gpu) 
    {
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
    }

    VideoCapture cap(live_video ? video_source : 0);
    if (!cap.isOpened()) 
    {
        cerr << "Error: Cannot open video source." << endl;
        return -1;
    }

    frameQueue.set_capacity(10);

    thread producer(captureFrames, ref(cap));
    thread consumer(processFrames, ref(net));

    producer.join();
    consumer.join();

    return 0;
}
