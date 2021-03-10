#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <chrono>
#define WINDOW_HEIGHT 720
#define WINDOW_WIDTH 1280
#define INDEX_CAMERA 1



using namespace std;
using namespace cv;
void detectAndDisplay(Mat& frame);
CascadeClassifier face_cascade;

int main(int argc, const char** argv)
{

    if (!face_cascade.load("haarcascade_frontalface_alt.xml"))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open(INDEX_CAMERA);

    /*capture.set(CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH);
    capture.set(CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT);*/

    namedWindow("Camera", WINDOW_AUTOSIZE);

    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay(frame);
        imshow("Camera", frame);
        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 0;
}
void detectAndDisplay(Mat& frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    std::vector<Rect> faces;
    chrono::high_resolution_clock::time_point start_face = chrono::high_resolution_clock::now();
    face_cascade.detectMultiScale(frame_gray, faces, 2.1, 2, 0, Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
    }
    //-- Show what you got
    chrono::high_resolution_clock::time_point end_face = chrono::high_resolution_clock::now();
    auto face_ms = chrono::duration_cast<chrono::milliseconds>(end_face - start_face).count();

    cout << face_ms << " ms" << endl;
    //imshow("Camera", frame);
}