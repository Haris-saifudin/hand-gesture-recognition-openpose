// ----------------------- OpenPose C++ API Tutorial - Example 3 - Body from image -----------------------
// It reads an image, process it, and displays it with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Third-party dependencies
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/core/types_c.h>
#include <vector>
#include <chrono>
#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#define CAMERA_DEVICE 1

#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>


// window size
#define WIDTH_WINDOWS 640
#define HEIGHT_WINDOWS 480

// Face Detection file 
#define OPENCV_CASCADE_FILENAME "haarcascade_frontalface_alt.xml"

// Display
DEFINE_bool(no_display, false,
    "Enable to disable the visual display.");


//using namespace std;
//using namespace cv;

//std::stringstream folderName, strName;
int counterImg = 1;
//std::string filename;
//std::string strCounter;


int rightWrist_x1;
int rightWrist_y1;
int rightWrist_x2;
int rightWrist_y2;
int stomatch;
int rightWrist;

cv::Mat handCropped;

//save 
void writeCSV(std::string filename, cv::Mat& m)
{
    std::ofstream myfile;
    myfile.open(filename.c_str());
    myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
    std::cout << "success save : " << filename << std::endl;

}

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, cv::Mat& Img)
{
    try
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            
            // Display image

            auto fps = op::getCvCapPropFrameFps();

            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            std::cout << "fps = " << fps << std::endl;

            if (!cvMat.empty())
            {
                //cv::imshow(OPEN_POSE_NAME_AND_VERSION + " Openpose", cvMat);
                //cv::imshow("Openpose", cvMat);
                const auto numberPeopleDetected = datumsPtr->at(0)->poseKeypoints.getSize(0);

                std::cout << "People Detected = " << numberPeopleDetected << std::endl;
  
                if (numberPeopleDetected != 0) {

                    //LEFT HAND

                    /*int pt1x = datumsPtr->at(0)->poseKeypoints[12] + 50;
                    int pt1y = datumsPtr->at(0)->poseKeypoints[13] + 50;
                    int pt2x = datumsPtr->at(0)->poseKeypoints[12] - 80;
                    int pt2y = datumsPtr->at(0)->poseKeypoints[13] - 120;*/
                    const auto opTimerHandDetection = op::getTimerInit();
                    //RIGHT WRIST
                    rightWrist_x1 = datumsPtr->at(0)->poseKeypoints[21] + 40;
                    rightWrist_y1 = datumsPtr->at(0)->poseKeypoints[22] + 40;
                    rightWrist_x2 = datumsPtr->at(0)->poseKeypoints[21] - 40;
                    rightWrist_y2 = datumsPtr->at(0)->poseKeypoints[22] - 60;

                    stomatch = datumsPtr->at(0)->poseKeypoints[4] + 60;
                    rightWrist = datumsPtr->at(0)->poseKeypoints[22];
                   /* std::cout << "Right Wrist = " << rightWrist << std::endl
                        << "Stomatch =" << stomatch << std::endl;*/

                    //std::cout << pt1x << "-" << pt1y << "|" << pt2x << "-" << pt2y << std::endl;
                    if (rightWrist_y2 > 0 && rightWrist_x2 > 0) {
                        if (rightWrist_y1 < HEIGHT_WINDOWS && rightWrist_x1 < WIDTH_WINDOWS) {
                            //std::cout << rightWrist_x1 << ":" << rightWrist_y1 << "|" << rightWrist_x2 << ":" << rightWrist_y2  << std::endl;
                            if (stomatch > rightWrist) {
                                cv::Rect Rec(rightWrist_x2, rightWrist_y2, 80, 80);
                                cv::rectangle(Img, Rec, cv::Scalar(0, 255, 0), 0, 8, 0);
                                cv::Mat Roi = Img(Rec);
                                //cv::Mat handCropped;
                                cv::resize(Roi, handCropped, cv::Size(64, 64));
                                //cv::imshow("Hand Detection", Img);

                                if (counterImg > 0 && counterImg <= 10) {
                                    std::string filename;
                                    std::string strCounter = std::to_string(counterImg);
                                    std::stringstream strName;
                                    strName << "./testing/gesture" << strCounter << ".jpg";

                                    strName >> filename;
                                    //cv::imshow("Crop", handCropped);
                                    std::cout << filename << std::endl;
                                    //writeCSV(filename, handCropped);
                                    cv::imwrite(filename, handCropped);

                                    counterImg++;

                                }
                                else {
                                    counterImg = 1;
                                }
                            }
                            else {
                                std::cout << "don't saved" << std::endl;
                                cv::Rect Rec(rightWrist_x2, rightWrist_y2, 80, 80);
                                cv::rectangle(Img, Rec, cv::Scalar(0, 0, 255), 0, 8, 0);
                                //cv::Mat Roi = Img(Rec);
                            }

                        }

                    }
                    op::printTime(opTimerHandDetection, "Time Hand Detection : ", " seconds.", op::Priority::High);
                }
                //cv::waitKey(0);
            }
            else
                op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
            FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging };
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold };
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold };
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port) };
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(1,FLAGS_3d)
        };

        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int Openpose()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        cv::VideoCapture webcam;
        webcam.open(CAMERA_DEVICE);
        webcam.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH_WINDOWS);
        webcam.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT_WINDOWS);


        if (!webcam.isOpened()) {
            std::cout << "ERROR! unable to open camera" << std::endl;
        }
        cv::Mat frame;

        //Face detection using HAAR CASCADE
        std::cout << "Face Detection file: ";
        cv::CascadeClassifier face_cascade;
        bool face_xml = face_cascade.load(OPENCV_CASCADE_FILENAME);


        if (face_xml == 0) {
            std::cerr << "Face xml did not load successfully..." << std::endl;
            return -1;
        }
        else
            std::cout << "Face xml was successfully loaded..." << std::endl;
        std::vector<cv::Rect> faces;
        cv::Mat frame_gray;

        op::Matrix imageToProcess;
        while (true) {
            const auto opTimer = op::getTimerInit();

            webcam.read(frame);
            // check if we succeeded
            if (frame.empty()) {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            //std::chrono::high_resolution_clock::time_point start_face = std::chrono::high_resolution_clock::now();
         
            const auto opTimerFaceDetection = op::getTimerInit();
            cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
            face_cascade.detectMultiScale(frame_gray, faces, 2.1, 3, 0, cv::Size(30, 30));
            std::cout << std::endl;
            op::printTime(opTimerFaceDetection, "Time Face Detection : ", " seconds.", op::Priority::High);
            //std::chrono::high_resolution_clock::time_point end_face = std::chrono::high_resolution_clock::now();
            //auto face_ms = std::chrono::duration_cast<std::chrono::milliseconds> (end_face - start_face).count();
            for (auto&& feature : faces) {
                cv::rectangle(frame, feature, cv::Scalar(0, 255, 0), 2);
            }

            // show live and wait for a key with timeout long enough to show images
 /*           for (int i = 0; i < faces.size(); i++)
            {
                cv::Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);*/
                if (faces.size() != 0) {


                    // Display Face Detected
                    //cv::ellipse(frame, center, cv::Size(faces[i].width * 0.5, faces[i].height * 0.5), 2, 0, 360, cv::Scalar( 0, 255, 0));
                    
                    const auto opTimerSkeleton = op::getTimerInit();

                    // Process and display image
                    imageToProcess = OP_CV2OPCONSTMAT(frame);
                    auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
                    if (datumProcessed != nullptr)
                    {
                        if (!FLAGS_no_display)
                            display(datumProcessed, frame);
                    }
                    else
                        op::opLog("Image could not be processed.", op::Priority::High);

                    op::printTime(opTimerSkeleton, "Time Skeleton : ", " seconds.", op::Priority::High);

                }
            //}
            cv::imshow("RGB", frame);

            //std::cout << "Time Face Detection = " << face_ms << std::endl;

            op::printTime(opTimer, "Total time: ", " seconds.", op::Priority::High);
            if (cv::waitKey(5) >= 0) {
                break;
            }
        }
        // Measuring total time
      

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char* argv[])
{
    // Parsing command line flags
    //gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return Openpose();
}
