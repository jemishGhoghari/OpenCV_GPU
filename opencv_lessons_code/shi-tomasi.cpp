#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>

int main() {
    cv::Mat img = cv::imread("lena.png");
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Point2f> corners;

    std::cout << "Image type: " << cv::typeToString(gray.type()) << "\n";

    cv::goodFeaturesToTrack(gray, corners, 500, 0.01, 25);

    // std::cout << "Image type Rows: " << corners.rows << " Cols: " << corners.cols << "\n";

    // std::cout << "Type: " << corners.type() << "\n";

    // int pix = (int)corners.at<uchar>(0);
    // int pix1 = (int)corners.at<uchar>(1);
    // std::cout << "Type: " << pix << " Pix2: " << pix1 << "\n";

    // for (int i = 0; i < corners.cols; ++i) {
    //     for (int j = 0; j < corners.rows; ++j)
    //     {
            
    //     }
    // }

    for (const auto &point : corners) {
        // std::cout << "Point X: " << point.x << " Y:" << point.y << "\n";
        cv::circle(img, cv::Point(point.x, point.y), 4, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("Conrners", img);
    cv::waitKey(0);
    return 0;
}