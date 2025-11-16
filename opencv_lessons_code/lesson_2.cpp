/**
 * Pixel addressing and ROI operations
 */

#include <opencv2/opencv.hpp>
#include <iostream>

static std::string typeToString(int type)
{
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        /* code */
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
    }

    r += "C" + std::to_string(chans);
    return r;
}

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

    cv::Rect roi(img.cols / 2, img.rows / 4, img.cols / 2, img.rows / 2);
    cv::Mat mid = img(roi).clone();

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Assignment 1: Manual Gray scale calculation
    cv::Mat manual_gray(img.cols, img.rows, CV_8UC1);
    for (int x = 0; x < img.cols; ++x)
    {
        for (int y = 0; y < img.rows; ++y)
        {
            cv::Vec3b &pix_val = img.at<cv::Vec3b>(x, y);
            manual_gray.at<cv::Vec<uchar, 1>>(x, y)[0] = pix_val[0] * 0.114 + pix_val[1] * 0.587 + pix_val[2] * 0.299;
        }
    }

    std::cout << "Gray Image Size: " << gray.cols << "x" << gray.rows << "\n";
    // Assignment 2: Compute a 256-bin grayscale histogram from scratch (no calcHist); render it as an image.
    

    // int cx = gray.cols / 2;
    // int cy = gray.rows / 2;
    // int hw = std::min(50, std::min(cx, cy));

    // for (int y = cy - hw; y < cy + hw; ++y)
    // {
    //     for (int x = cx - hw; x < cx + hw; ++x)
    //     {
    //         uchar &p = gray.at<uchar>(y, x);
    //         p = 255 - p;
    //     }
    // }

    std::cout << "Mid Type: " << typeToString(mid.type()) << " Type ID: " << typeid(mid).name() << "\n";

    cv::imshow("Test Image Manual Gray", manual_gray);
    cv::imshow("Test Image cvColor mathod", gray);
    // std::cout << "Image type: " << typeToString(img.type()) << " From CV function: " << img.type() << "\n";
    cv::waitKey(0);
    // Destroy All Windows
    cv::destroyAllWindows();
    return 0;
}