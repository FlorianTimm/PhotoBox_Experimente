#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <image>\n", argv[0]);
        return 1;
    }
    // Read the image file
    cv::Mat image = cv::imread(argv[1]);
    // Detect ArUco markers in the image
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    // Create your own custom dictionary
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::generateCustomDictionary(32, 3);
    // Detect ArUco markers in the image using the custom dictionary
    cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds);

    printf("Detected %lu markers\n", markerIds.size());
    for (size_t i = 0; i < markerIds.size(); i++)
    {
        printf("Marker ID: %d\n", markerIds[i]);
    }

    cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
    cv::imwrite("output.jpg", image);

    return 0;
}
