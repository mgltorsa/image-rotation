#include <stdio.h>
// #include <mpi.h>

#include <opencv2/opencv.hpp>

// mpirun -n 4 ./upscale_image input.jpg output.jpg 0 0 640 480

using namespace cv;

int main(int argc, char **argv)
{

    // TODO: Display image example
    // if (argc != 2)
    // {
    //     printf("usage: DisplayImage.out <Image_Path>\n");
    //     return -1;
    // }
    // Mat image;
    // image = imread(argv[1], 1);
    // if (!image.data)
    // {
    //     printf("No image data \n");
    //     return -1;
    // }
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    // imshow("Display Image", image);
    // waitKey(0);
    // return 0;
    // End example

    if (argc != 7)
    {
        printf("usage: %s <input_image> <output_image> <x> <y> <width> <height>\n", argv[0]);
        return -1;
    }

    Mat input_image = imread(argv[1], IMREAD_COLOR);

    if (input_image.empty())
    {
        printf("error: could not load input image\n");
        return -1;
    }

    int x, y, width, height;

    x = atoi(argv[3]);
    y = atoi(argv[4]);
    width = atoi(argv[5]);
    height = atoi(argv[6]);

    Rect roi(x, y, width, height);

    Mat roi_image = input_image(roi);

    // Upscale the chunk using Lanczos resampling
    Mat upscaled_roi_image = Mat(input_image.size(), input_image.type());
    
    resize(roi_image, upscaled_roi_image, Size(width * 2, height * 2), 0, 0, INTER_LANCZOS4);

    imwrite(argv[2], upscaled_roi_image);
    return 0;
}