#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <vector>
#include <random>

using namespace cv;
using namespace std;

// Configuration parameters
bool use_gpu = true;
bool live_video = false; // Set to true for live video feed, false for video file
float confidence_level = 0.5; // Minimum confidence level for object detection

// Define the classes for which the model can detect objects
const vector<string> CLASSES = {"background", "aeroplane", "bicycle", "bird", "boat",
                                 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                 "sofa", "train", "tvmonitor"};

// Generate random COLORS for each class
vector<Vec3b> COLORS(CLASSES.size());
void generateColors() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 255);
    for (size_t i = 0; i < CLASSES.size(); ++i) {
        COLORS[i] = Vec3b(dis(gen), dis(gen), dis(gen));
    }
}

// Function to process frames
void processFrames(tbb::concurrent_bounded_queue<Mat>& frameQueue, Net& net) {
    while (true) {
        Mat frame;
        if (frameQueue.try_pop(frame)) {
            // Prepare the frame for object detection
            Mat blob;
            cv::dnn::blobFromImage(frame, blob, 0.007843, Size(300, 300), Scalar(127.5));
            net.setInput(blob);
            Mat detections = net.forward();

            int h = frame.rows;
            int w = frame.cols;

            // Loop over the detections
            for (int i = 0; i < detections.size[2]; i++) {
                float confidence = detections.at<float>(0, 0, i, 2);

                // Check if the confidence level is above the threshold
                if (confidence > confidence_level) {
                    int idx = static_cast<int>(detections.at<float>(0, 0, i, 1));
                    Rect box = Rect(detections.at<float>(0, 0, i, 3) * w,
                                    detections.at<float>(0, 0, i, 4) * h,
                                    (detections.at<float>(0, 0, i, 5) - detections.at<float>(0, 0, i, 3)) * w,
                                    (detections.at<float>(0, 0, i, 6) - detections.at<float>(0, 0, i, 4)) * h);

                    // Draw a bounding box and label around the detected object
                    string label = format("%s: %.2f%%", CLASSES[idx].c_str(), confidence * 100);
                    rectangle(frame, box, COLORS[idx], 2);
                    int y = box.y - 15 < 15 ? box.y + 15 : box.y - 15;
                    putText(frame, label, Point(box.x, y), FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1);
                }
            }

            // Display the frame
            imshow("Live detection", frame);
            if (waitKey(1) == 27) break; // Break the loop if the 'Esc' key is pressed
        }
    }
}

int main() {
    // Load pre-trained MobileNet SSD model
    Net net = cv::dnn::readNetFromCaffe("ssd_files/MobileNetSSD_deploy.prototxt",
                                         "ssd_files/MobileNetSSD_deploy.caffemodel");

    // Set backend and target to CUDA if GPU is enabled
    if (use_gpu) {
        cout << "[INFO] setting preferable backend and target to CUDA..." << endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    // Open a video stream (live or from a file)
    cout << "[INFO] accessing video stream..." << endl;
    VideoCapture vs(live_video ? 0 : "test-2.mp4");
    if (!vs.isOpened()) {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }

    // Generate random colors
    generateColors();

    // Create a queue for frame exchange
    tbb::concurrent_bounded_queue<Mat> frameQueue;

    // Start a thread for processing frames
    thread processingThread(processFrames, ref(frameQueue), ref(net));

    // Main video processing loop
    Mat frame;
    while (true) {
        if (!vs.read(frame)) break; // Read a frame from the video source
        frameQueue.push(frame); // Push the frame to the queue
    }

    // Wait for the processing thread to finish
    processingThread.join();

    cout << "[INFO] Video processing finished." << endl;
    return 0;
}

