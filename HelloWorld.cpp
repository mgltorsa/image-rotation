#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main()
{
    Mat image = imread("hubble_image.png", IMREAD_UNCHANGED);
    return 0;
}
