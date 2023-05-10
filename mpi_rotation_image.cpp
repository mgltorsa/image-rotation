#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>

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

int *calculate_out_coordinates(int cols, int rows, int chunk_cols, int chunk_rows, double angle)
{
    // assume the original image is of size (width, height)
    int width = cols;
    int height = rows;

    // assume each chunk is of size (chunk_width, chunk_height)
    int chunk_width = chunk_cols;
    int chunk_height = chunk_rows;

    // assume the rotation angle is stored in a variable called 'angle'
    double radians = angle * M_PI / 180.0;
    double cos_theta = cos(radians);
    double sin_theta = sin(radians);

    // compute the center of the original image
    int cx = width / 2;
    int cy = height / 2;

    // compute the center of the chunk
    int chunk_cx = chunk_width / 2;
    int chunk_cy = chunk_height / 2;

    // compute the coordinates of the center of the rotated chunk in the output image
    int out_cx = (int)(cx + (chunk_cx - cx) * cos_theta - (chunk_cy - cy) * sin_theta);
    int out_cy = (int)(cy + (chunk_cx - cx) * sin_theta + (chunk_cy - cy) * cos_theta);

    // compute the top-left corner of the rotated chunk in the output image
    int out_x = out_cx - chunk_cx;
    int out_y = out_cy - chunk_cy;

    int *coordinates;
    coordinates[0] = out_x;
    coordinates[1] = out_y;

    return coordinates;
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

    int image_step;
    int type;
    int cols;
    int rows;
    Size image_size;

    Mat image;

    if (rank == 0)
    {
        image = imread(argv[1], IMREAD_COLOR);

        if (image.empty())
        {
            cerr << "Error: could not open input image " << argv[1] << endl;
            MPI_Finalize();
            return 2;
        }

        image_step = image.step;
        type = image.type();
        image_size = image.size();
    }

    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image_step, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image_size, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {

        int rows = image_size.height;
        int rows_per_process = ceil(rows / (double)(size - 1));

        // printf("Process#%d - image.rows: %d, image.cols:%d size: %f  Row_per_process=%d \n", rank, rows, cols, (double)size, rows_per_process);

        for (int i = 1; i < size; i++)
        {
            int offset = (i - 1) * rows_per_process;
            int rows_to_send = min(rows_per_process, rows - offset);

            MPI_Send(&offset, 1, MPI_INT, i, TAG_OFFSET, MPI_COMM_WORLD);
            MPI_Send(&rows_to_send, 1, MPI_INT, i, TAG_ROWS, MPI_COMM_WORLD);

            // printf("Process#0 sent: Off: %d, rows: %d and step: %d\n", offset, rows_to_send, image_step);

            MPI_Send(image.ptr(offset), rows_to_send * image_step, MPI_UNSIGNED_CHAR, i, TAG_IMAGE, MPI_COMM_WORLD);
        }

        int offset = 0;
        int rows_to_process = min(rows_per_process, rows - offset);

        Size new_size = recalculates_size(image_size, angle);
        Mat output(new_size, type, image_step);

        Mat chunks[size];
        Mat masks[size];
        Point2f new_centers[size];
        Point2f originalCenter(output.cols / 2.0f, output.rows / 2.0f);

        for (int i = 1; i < size; i++)
        {
            Size chunk_size;
            int chunk_step;
            MPI_Recv(&chunk_size, 2, MPI_INT, i, TAG_CHUNK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&chunk_step, 1, MPI_INT, i, TAG_IMAGE_STEP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // printf("Recv chunk size - w:%d h:%d s:%d from process#%d\n", chunk_size.width, chunk_size.height, chunk_step, i);

            Mat rotated_chunk(chunk_size, image.type(), chunk_step);
            MPI_Recv(rotated_chunk.ptr(), chunk_size.height * chunk_step, MPI_UNSIGNED_CHAR, i, TAG_IMAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("recv chunk from process: %d. chunk_rows:%d chunk_cols:%d\n", i, rotated_chunk.rows, rotated_chunk.cols);

            imwrite(string(argv[2]) + "/process-" + to_string(i) + ".png", rotated_chunk);

            // int *coordinates = calculate_out_coordinates(output.cols, output.rows, rotated_chunk.cols, rotated_chunk.rows, angle);

            // int out_x = coordinates[0];
            // int out_y = coordinates[1];

            // printf("chunk: %d, outx: %d outy: %d \n", i, out_x, out_y);
            chunks[i - 1] = rotated_chunk;
            cv::Mat mask;
            cv::compare(rotated_chunk, cv::Scalar(0, 0, 0), mask, cv::CMP_NE);
            masks[i - 1] = mask;

            if (i - 1 == 0)
            {
                new_centers[i - 1] = Point2f(abs(output.size().width - rotated_chunk.size().width), 0);
            }

            // Root process does not rotate
            else if ((i - 1) == (size - 2))
            {
                new_centers[i - 1] = Point2f(0, new_centers[0].x);
            }
            else
            {
                new_centers[i - 1] = Point2f(new_centers[0].x / i, new_centers[0].x / i);
            }

            if (angle == 45)
            {
                chunks[i - 1]
                    .copyTo(output(cv::Rect(new_centers[i - 1].x, new_centers[i - 1].y, chunks[i - 1].cols, chunks[i - 1].rows)), masks[i - 1]);
            }
        }

        // printf("New center0 x:%f, y:%f\n", new_centers[0].x, new_centers[0].y);
        // printf("New center1 x:%f, y:%f\n", new_centers[1].x, new_centers[1].y);
        // printf("New center2 x:%f, y:%f\n", new_centers[2].x, new_centers[2].y);

        // chunks[0]
        //     .copyTo(output(cv::Rect(131.5, 0, chunks[0].cols, chunks[0].rows)), masks[0]);
        // chunks[1].copyTo(output(cv::Rect(131.5 / 2, 131.5 / 2, chunks[1].cols, chunks[1].rows)), masks[1]);
        // chunks[2].copyTo(output(cv::Rect(0, 131.5, chunks[2].cols, chunks[2].rows)), masks[2]);

        // printf("Process#0 recv image\n");

        imwrite(string(argv[2]) + "/" + string(argv[1]), output);

        // cout << "Image rotated successfully" << endl;
    }
    else
    {
        int offset, rows_to_process;
        MPI_Recv(&offset, 1, MPI_INT, 0, TAG_OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows_to_process, 1, MPI_INT, 0, TAG_ROWS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // printf("Process#%d received off:%d and rows_to_process: %d, w: %d\n", rank, offset, rows_to_process, image_size.width);

        Mat input(rows_to_process, image_size.width, type, image_step);

        MPI_Recv(input.ptr(), rows_to_process * image_step, MPI_UNSIGNED_CHAR, 0, TAG_IMAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // imwrite("partial-in-2-" + to_string(rank) + string(argv[2]), input);

        Mat output;

        rotate(input, output, angle, input.size());

        // imwrite("partial-2-" + to_string(rank) + string(argv[2]), output);

        rows_to_process = output.rows;
        int cols_to_process = output.cols;
        int chunk_type = output.type();
        int chunk_step = output.step;
        Size chunk_size = output.size();

        // printf("send w:%d h:%d s:%d, process#%d\n", chunk_size.width, chunk_size.height, chunk_step, rank);

        MPI_Send(&chunk_size, 2, MPI_INT, 0, TAG_CHUNK, MPI_COMM_WORLD);
        MPI_Send(&chunk_step, 1, MPI_INT, 0, TAG_IMAGE_STEP, MPI_COMM_WORLD);

        MPI_Send(output.ptr(), chunk_size.height * output.step, MPI_UNSIGNED_CHAR, 0, TAG_IMAGE, MPI_COMM_WORLD);
        // printf("Sent chunk from process#%d\n", rank);
    }

    // Perform some computation

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();
    double elapsedTime = endTime - startTime;

    // Print the elapsed time
    if (rank == 0)
    {
        printf("mpi-%d;%s;%f;%f\n", size, argv[1], angle, elapsedTime);
    }

    MPI_Finalize();
    return 0;
}
