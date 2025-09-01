#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv)
{
    cv::Mat img;
    // argc: number of arguments
    if (argc < 2)
    {
        std::cout << "[WARNING]: path to image not provided, switching to default image\n";
        img = cv::imread(cv::samples::findFile("lena.png"), cv::IMREAD_COLOR);
    }
    else
    {
        // arg1 is executable name itself
        // arg2 is first argument provided to executale
        std::cout << "Arg 1: " << argv[0] << ", Arg 2: " << argv[1] << "\n";
        const std::string path = argv[1];
        // imread -> reading image to Mat var
        img = cv::imread(path, cv::IMREAD_COLOR);
    }

    // resize: resize image to new WxH
    // cv::resize(img, img, cv::Size(1920, 1200));

    if (img.empty())
    {
        std::cerr << "Image not found\n";
        return 2;
    }

    std::cout << "Image: " << img.cols << "x" << img.rows << " channels: " << img.channels() << " type: " << img.type() << "\n";

    // nameWindow creates a window with specified name. Helpful for creating multiple windows and distinguish them with unique names.
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    // setWindowProperty -> set different properties of the window (eg: fullscreen, autosize etc.)
    cv::setWindowProperty("Original Image", cv::WND_PROP_AUTOSIZE, cv::WINDOW_AUTOSIZE);
    // imshow -> display image to the created window
    cv::imshow("Original Image", img);
    // clone -> copy image to another Mat.
    cv::Mat img_copy = img.clone();
    // imwrite -> write image to disk
    cv::imwrite("output_copy.png", img_copy);
    // waitKey -> wait DELAY ms for the key pressed event.
    cv::waitKey();

    return 0;
}