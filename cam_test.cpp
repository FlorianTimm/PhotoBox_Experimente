#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main()
{
    // Initialize libcamera
    libcamera::CameraManager manager;
    manager.start();
    auto camera = manager.get(0);
    camera->acquire();

    // Capture an image
    libcamera::Stream stream(camera->streams()[0]);
    stream->allocate();
    camera->queueRequest(stream->capture());

    libcamera::FrameBuffer buffer(stream->width(), stream->height(),
                                  libcamera::formats::YUV420);
    stream->attachBuffer(&buffer);

    camera->start();
    camera->waitForIdle();

    // Convert the captured image to OpenCV format
    cv::Mat image(stream->height(), stream->width(), CV_8UC2,
                  buffer.planes()[0].data());

    // Detect ArUco markers
    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds);

    // Draw detected markers on the image
    cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

    // Display the image
    cv::imshow("ArUco Marker Detection", image);
    cv::waitKey(0);

    // Release resources
    camera->stop();
    camera->release();

    return 0;
}
