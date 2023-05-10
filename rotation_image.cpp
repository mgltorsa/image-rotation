#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

using namespace cv;

Size recalculates_size(Size original_size, double angle)
{

    int new_width = original_size.width, new_height = original_size.height;
    double radians = angle * CV_PI / 180.0;

    double sin_angle = abs(sin(radians));
    double cos_angle = abs(cos(radians));
    new_width = int(original_size.width * cos_angle + original_size.height * sin_angle);
    new_height = int(original_size.width * sin_angle + original_size.height * cos_angle);

    Size size(new_width, new_height);

    return size;
}

int main(int argc, char **argv)
{

    if (argc != 4)
    {
        printf("usage: %s <input_image> <output_image> <angle>\n", argv[0]);
        return -1;
    }

    time_t startTime = time(nullptr);

    // Load the image
    Mat image = imread(argv[1], IMREAD_COLOR);

    // Rotate the image
    Point2f center(image.cols / 2.0F, image.rows / 2.0F);

    // ARGV[3] =ANgle in degrees
    double angle = atof(argv[3]);
    Mat rotation_matrix = getRotationMatrix2D(center, angle, 1.0);

    Mat rotated_image;
    Size new_size = recalculates_size(image.size(), angle);

    rotation_matrix.at<double>(0, 2) += (new_size.width - image.cols) / 2.0;
    rotation_matrix.at<double>(1, 2) += (new_size.height - image.rows) / 2.0;

    // Size size(image.size().width * 2, image.size().height * 2);
    warpAffine(image, rotated_image, rotation_matrix, new_size);

    // Save the rotated image
    imwrite(argv[2], rotated_image);

    time_t endTime = time(nullptr);
    double elapsedTime = difftime(endTime, startTime);

    printf("serial;%s;%f,%f\n", argv[1], angle, elapsedTime);
    return 0;
}