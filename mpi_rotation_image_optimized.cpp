#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>
#include <omp.h>

#define TAG_IMAGE 1
#define TAG_CHUNK 2
#define TAG_OFFSET 3
#define TAG_ROWS 4
#define TAG_IMAGE_STEP 5

using namespace std;
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

void rotate(Mat &input, Mat &output, double angle, Size original_size)
{
    Point2f center((original_size.width) / 2.0, (original_size.height) / 2.0);
    Mat rotation_matrix = getRotationMatrix2D(center, -angle, 1.0);

    Size new_size = recalculates_size(original_size, angle);

    rotation_matrix.at<double>(0, 2) += (new_size.width - original_size.width) / 2.0;
    rotation_matrix.at<double>(1, 2) += (new_size.height - original_size.height) / 2.0;

    warpAffine(input, output, rotation_matrix, new_size);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();

    if (argc != 4)
    {
        if (rank == 0)
        {
            cerr << "usage: <input_image> <output_image_folder> <angle> " << argv[1] << endl;
        }
        MPI_Finalize();
        return 1;
    }

    double angle = atof(argv[3]);

    if (size < 2)
    {
        cerr << "There should be at least 2 processes" << endl;
        return 1;
    }

    int offset;
    int image_step;
    int type;
    int cols;
    int rows;
    Size image_size;
    Mat image;

    image = imread(argv[1], IMREAD_UNCHANGED);

    if (image.empty())
    {
        cerr << "Error: could not open input image " << argv[1] << endl;
        MPI_Finalize();
        return 2;
    }

    rows = image.rows;
    cols = image.cols;
    image_step = image.step;
    type = image.type();
    image_size = image.size();

    int rows_per_process = ceil(rows / (double)size);
    offset = (rank)*rows_per_process;
    int rows_to_process = min(rows_per_process, rows - offset);

    Mat chunk(rows_to_process, image_size.width, type, image_step);
    Mat chunk_output;

    rotate(chunk, chunk_output, angle, chunk.size());

    if (rank == 0)
    {

        Size new_size = recalculates_size(image_size, angle);
        Mat output(new_size, type, image_step);

        Mat chunks[size];
        Mat masks[size];

        Point2d new_centers[size];
        chunks[0] = chunk_output;
        cv::Mat mask;
        cv::compare(chunk_output, cv::Scalar(0, 0, 0), mask, cv::CMP_NE);
        masks[0] = mask;
        int i;

#pragma omp parallel private(i)
        for (i = 1; i < size; i++)
        {
            offset = (rank)*rows_per_process;

            rows_to_process = min(rows_per_process, rows - offset);

            Size chunk_size(rows_to_process, image.size().width);
            int chunk_step = image.step;

            MPI_Recv(&chunk_size, 2, MPI_INT, i, TAG_CHUNK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&chunk_step, 1, MPI_INT, i, TAG_IMAGE_STEP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Mat rotated_chunk(chunk_size, image.type(), image_step);
            MPI_Recv(rotated_chunk.ptr(), chunk_size.height * chunk_step, MPI_UNSIGNED_CHAR, i, TAG_IMAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // CHUNKS PROCESSING
            chunks[i] = rotated_chunk;
            mask;
            compare(rotated_chunk, cv::Scalar(0, 0, 0), mask, cv::CMP_NE);
            masks[i] = mask;
        }

        for (i = 0; i < size; i++)
        {
            if (i == 0)
            {
                new_centers[i] = Point2f(abs(output.size().width - chunks[i].size().width), 0);
            }

            else if ((i) == (size - 1))
            {
                new_centers[i] = Point2f(0, new_centers[0].x);
            }
            else
            {
                new_centers[i] = Point2f(new_centers[0].x / i, new_centers[0].x / i);
            }

            if (angle == 45)
            {
                chunks[i]
                    .copyTo(output(cv::Rect(new_centers[i].x, new_centers[i].y, chunks[i].cols, chunks[i].rows)), masks[i]);
            }
        }

        imwrite(string(argv[2]) + "/" + string(argv[1]), output);
    }
    else
    {
        int cols_to_process = chunk_output.cols;
        int chunk_type = chunk_output.type();
        int chunk_step = chunk_output.step;
        Size chunk_size = chunk_output.size();

        MPI_Send(&chunk_size, 2, MPI_INT, 0, TAG_CHUNK, MPI_COMM_WORLD);
        MPI_Send(&chunk_step, 1, MPI_INT, 0, TAG_IMAGE_STEP, MPI_COMM_WORLD);

        MPI_Send(chunk_output.ptr(), chunk_size.height * chunk_output.step, MPI_UNSIGNED_CHAR, 0, TAG_IMAGE, MPI_COMM_WORLD);
    }

    // Perform some computation

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;

    // Print the elapsed time
    if (rank == 0)
    {
        // printf("mpi-%d;%s;%2.f;%f\n", size, argv[1], angle, elapsedTime);
        printf("mpi-%d-optimized;%s;%2.f;%f\n", size, argv[1], angle, elapsedTime);
    }

    MPI_Finalize();
    return 0;
}
