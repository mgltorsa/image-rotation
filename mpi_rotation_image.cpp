#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>

#define TAG_OFFSET 0
#define TAG_ROWS 1
#define TAG_COLS 2
#define TAG_TYPE 3
#define TAG_IMAGE_STEP 4
#define TAG_IMAGE 5

using namespace std;
using namespace cv;

void rotate(Mat &input, Mat &output, double angle)
{
    Point2f center((input.cols - 1) / 2.0, (input.rows - 1) / 2.0);
    Mat rotation = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(input, output, rotation, input.size());
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4)
    {
        if (rank == 0)
        {
            cerr << "usage: <input_image> <output_image> <angle> " << argv[1] << endl;
        }
        MPI_Finalize();
        return 1;
    }

    printf("Total processes %d\n", size);

    double angle = atof(argv[3]);

    if (size < 2)
    {
        cerr << "There should be at least 2 processes" << endl;
        return 1;
    }

    if (rank == 0)
    {
        Mat image = imread(argv[1], IMREAD_COLOR);

        if (image.empty())
        {
            cerr << "Error: could not open input image " << argv[1] << endl;
            MPI_Finalize();
            return 2;
        }

        int rows_per_process = ceil(image.rows / (double)size);
        printf("Process#%d - image.rows: %d / size: %f  Row_per_process=%d \n", rank, image.rows, (double)size, rows_per_process);
        for (int i = 1; i < size; i++)
        {
            int offset = i * rows_per_process;
            int rows_to_send = min(rows_per_process, image.rows - offset);
            int cols = image.cols;
            int image_step = image.step;
            int type = image.type();

            MPI_Send(&offset, 1, MPI_INT, i, TAG_OFFSET, MPI_COMM_WORLD);
            MPI_Send(&rows_to_send, 1, MPI_INT, i, TAG_ROWS, MPI_COMM_WORLD);
            MPI_Send(&cols, 1, MPI_INT, i, TAG_COLS, MPI_COMM_WORLD);

            MPI_Send(&type, 1, MPI_INT, i, TAG_TYPE, MPI_COMM_WORLD);
            MPI_Send(&image_step, 1, MPI_INT, i, TAG_IMAGE_STEP, MPI_COMM_WORLD);

            printf("Process#0 sending: Off: %d, rows: %d and step: %d\n", offset, rows_to_send, image_step);

            MPI_Send(image.ptr(offset), rows_to_send * image_step, MPI_UNSIGNED_CHAR, i, TAG_IMAGE, MPI_COMM_WORLD);
        }

        int offset = 0;
        int rows_to_process = min(rows_per_process, image.rows - offset);

        Mat output(rows_to_process, image.cols, image.type());
        printf("Process#0 preparing to rotate my part");

        rotate(image.rowRange(offset, offset + rows_to_process), output, angle);

        // for (int i = 1; i < size; i++)
        // {
        //     MPI_Recv(&offset, 1, MPI_INT, i, TAG_OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     MPI_Recv(&rows_to_process, 1, MPI_INT, i, TAG_OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     MPI_Recv(output.ptr(offset), rows_to_process * image.step, MPI_UNSIGNED_CHAR, i, TAG_IMAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // }

        imwrite(argv[2], output);

        cout << "Image rotated successfully" << endl;
    }
    else
    {
        int offset, rows_to_process, cols, img_step, type;
        MPI_Recv(&offset, 1, MPI_INT, 0, TAG_OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows_to_process, 1, MPI_INT, 0, TAG_ROWS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&cols, 1, MPI_INT, 0, TAG_COLS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&type, 1, MPI_INT, 0, TAG_TYPE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&img_step, 1, MPI_INT, 0, TAG_IMAGE_STEP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process#%d with off: %d and rows: %d and img_step: %d\n", rank, offset, rows_to_process, img_step);

        Mat input(rows_to_process, cols, type);

        printf("Process#%d RECV Off: %d, rows: %d and step: %lu\n", rank, offset, rows_to_process, rows_to_process * input.step);

        MPI_Recv(input.ptr(), rows_to_process * img_step, MPI_UNSIGNED_CHAR, 0, TAG_IMAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Mat output(rows_to_process, input.cols, input.type());

        printf("Process#%d received image", rank);

        rotate(input, output, angle);

        // MPI_Send(&offset, 1, MPI_INT, 0, TAG_OFFSET, MPI_COMM_WORLD);
        // MPI_Send(&rows_to_process, 1, MPI_INT, 0, TAG_OFFSET, MPI_COMM_WORLD);
        // MPI_Send(output.ptr(), rows_to_process * output.step, MPI_UNSIGNED_CHAR, 0, TAG_IMAGE, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
