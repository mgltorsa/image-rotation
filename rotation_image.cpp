#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv)
{

    if (argc != 4)
    {
        printf("usage: %s <input_image> <output_image> <angle>\n", argv[0]);
        return -1;
    }

    // Load the image
    Mat image = imread(argv[1], IMREAD_COLOR);

    // Rotate the image
    Point2f center(image.cols / 2.0, image.rows / 2.0);

    // ARGV[3] =ANgle in degrees
    Mat rotation_matrix = getRotationMatrix2D(center, atoi(argv[3]), 1.0);
    Mat rotated_image;
    warpAffine(image, rotated_image, rotation_matrix, image.size());

    // Save the rotated image
    imwrite(argv[2], rotated_image);

    return 0;
}