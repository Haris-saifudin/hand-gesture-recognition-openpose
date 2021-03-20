#include "SVM.h"
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

// <- Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#define CAMERA_DEVICE 1

#include <openpose/flags.hpp>
// <- OpenPose dependencies
#include <openpose/headers.hpp>

// <- window size
#define WIDTH_WINDOWS 640
#define HEIGHT_WINDOWS 480

// <- Path Face Detection 
#define OPENCV_CASCADE_FILENAME "haarcascade_frontalface_alt.xml"

// <- Display
DEFINE_bool(no_display, false,
    "Enable to disable the visual display.");


using namespace std;
using namespace cv;

// <- Inisialization Global Variable
int counterImg = 1;

// <- HAND ROI
int stomatch;
int rightWrist;
int leftWrist;

int rightWrist_x1;
int rightWrist_y1;
int rightWrist_x2;
int rightWrist_y2;

int leftWrist_x1;
int leftWrist_y1;
int leftWrist_x2;
int leftWrist_y2;


// <- Inisialization HOG
// <- 64x64 Pixel, normalisasi blok 16x16, cell 8x8, col 8x8, bin 9
cv::HOGDescriptor handFeature(cv::Size(64, 64), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
std::vector<float> featureVector;
std::vector <cv::Point> locHand;
cv::Mat dynamicGesture;
cv::Mat tmp;


// ,- SVM Declaration
int h_klasifikasi = 0;
static int(*info)(const char* fmt, ...) = &printf;
struct svm_node* x;
int max_nr_attr = 64;
struct svm_model* model;
int predict_probability = 0;
static char* baris;
static int max_line_len;



static char* readline(FILE* input)
{
    int len;

    if (fgets(baris, max_line_len, input) == NULL)
        return NULL;

    while (strrchr(baris, '\n') == NULL)
    {
        max_line_len *= 2;
        baris = (char*)realloc(baris, max_line_len);
        len = (int)strlen(baris);
        if (fgets(baris + len, max_line_len - len, input) == NULL)
            break;
    }
    return baris;
}

void exit_input_error(int line_num)
{
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(1);
}

void predict(FILE* input, FILE* output) //svm predict
{

    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int svm_type = svm_get_svm_type(model);
    int nr_class = svm_get_nr_class(model);
    double* prob_estimates = NULL;
    int j;

    if (predict_probability)
    {
        if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
            info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n", svm_get_svr_probability(model));
        else
        {
            int* labels = (int*)malloc(nr_class * sizeof(int));
            svm_get_labels(model, labels);
            prob_estimates = (double*)malloc(nr_class * sizeof(double));
            fprintf(output, "labels");
            for (j = 0; j < nr_class; j++) {
                fprintf(output, " %d", labels[j]);
                h_klasifikasi = labels[j];
            }
            fprintf(output, "\n");
            free(labels);
        }
    }

    max_line_len = 1024;
    baris = (char*)malloc(max_line_len * sizeof(char));
    while (readline(input) != NULL)
    {
        int i = 0;
        double target_label, predict_label;
        char* idx, * val, * label, * endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        label = strtok(baris, " \t\n");
        if (label == NULL) // empty line
            exit_input_error(total + 1);

        target_label = strtod(label, &endptr);
        if (endptr == label || *endptr != '\0')
            exit_input_error(total + 1);

        while (1)
        {
            if (i >= max_nr_attr - 1)	// need one more for index = -1
            {
                max_nr_attr *= 2;
                x = (struct svm_node*)realloc(x, max_nr_attr * sizeof(struct svm_node));
            }

            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL)
                break;
            errno = 0;
            x[i].index = (int)strtol(idx, &endptr, 10);
            if (endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
                exit_input_error(total + 1);
            else
                inst_max_index = x[i].index;

            errno = 0;
            x[i].value = strtod(val, &endptr);
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(total + 1);

            ++i;
        }
        x[i].index = -1;

        if (predict_probability && (svm_type == C_SVC || svm_type == NU_SVC))
        {
            predict_label = svm_predict_probability(model, x, prob_estimates);
            fprintf(output, "%g", predict_label);
            for (j = 0; j < nr_class; j++)
                fprintf(output, " %g", prob_estimates[j]);
            fprintf(output, "\n");
        }
        else
        {
            predict_label = svm_predict(model, x);
            h_klasifikasi = predict_label;
            fprintf(output, "%.17g\n", predict_label);
        }

        if (predict_label == target_label)
            ++correct;
        error += (predict_label - target_label) * (predict_label - target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label * predict_label;
        sumtt += target_label * target_label;
        sumpt += predict_label * target_label;
        ++total;
    }
    if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
    {
        info("Mean squared error = %g (regression)\n", error / total);
        info("Squared correlation coefficient = %g (regression)\n",
            ((total * sumpt - sump * sumt) * (total * sumpt - sump * sumt)) /
            ((total * sumpp - sump * sump) * (total * sumtt - sumt * sumt))
        );
    }
    else
        /*info("Accuracy = %g%% (%d/%d) (classification) -> ",
            (double)correct / total * 100, correct, total);*/
        if (predict_probability)
            free(prob_estimates);
}



// <- Save to format .csv
void writeCSV(std::string filename, cv::Mat& m)
{
    std::ofstream myfile;
    myfile.open(filename.c_str());
    myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
    std::cout << "success save : " << filename << std::endl;

}


// <- This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, cv::Mat& Img)
{
    try
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            
            // <- Display image
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty())
            {
                //cv::imshow(OPEN_POSE_NAME_AND_VERSION + " Openpose", cvMat);
                cv::imshow("Openpose", cvMat);

                // <- Get people detection
                const auto numberPeopleDetected = datumsPtr->at(0)->poseKeypoints.getSize(0);

                std::cout << "People Detected = " << numberPeopleDetected << std::endl;

                if (numberPeopleDetected != 0) {

                    const auto opTimerHandDetection = op::getTimerInit();
                    
                    // <- LEFT WRIST
                    leftWrist_x1 = datumsPtr->at(0)->poseKeypoints[12] + 40;
                    leftWrist_y1 = datumsPtr->at(0)->poseKeypoints[13] + 40;
                    leftWrist_x2 = datumsPtr->at(0)->poseKeypoints[12] - 40;
                    leftWrist_y2 = datumsPtr->at(0)->poseKeypoints[13] - 60;
                    
                    // <- RIGHT WRIST
                    rightWrist_x1 = datumsPtr->at(0)->poseKeypoints[21] + 40;
                    rightWrist_y1 = datumsPtr->at(0)->poseKeypoints[22] + 40;
                    rightWrist_x2 = datumsPtr->at(0)->poseKeypoints[21] - 40;
                    rightWrist_y2 = datumsPtr->at(0)->poseKeypoints[22] - 60;

                    stomatch = datumsPtr->at(0)->poseKeypoints[4] + 80;
                    rightWrist = datumsPtr->at(0)->poseKeypoints[22];
                    leftWrist = datumsPtr->at(0)->poseKeypoints[13];

                   /* std::cout << "Right Wrist = " << rightWrist << std::endl
                        << "Stomatch =" << stomatch << std::endl;*/

                    //std::cout << pt1x << "-" << pt1y << "|" << pt2x << "-" << pt2y << std::endl;

                    // <- Boundary Image
                    if (/*(rightWrist_y2 > 0 && rightWrist_x2 > 0) &&*/ leftWrist_y2 > 0 && leftWrist_x2 > 0) {
                        if (/*(rightWrist_y1 < HEIGHT_WINDOWS && rightWrist_x1 < WIDTH_WINDOWS) && */
                            leftWrist_y1 < HEIGHT_WINDOWS && leftWrist_x1 < WIDTH_WINDOWS) {
                            //std::cout << rightWrist_x1 << ":" << rightWrist_y1 << "|" << rightWrist_x2 << ":" << rightWrist_y2  << std::endl;
                            
                            if (/*stomatch > rightWrist &&*/ stomatch > leftWrist) {
                                
                                //Bounding Box Hand Detection
                                //cv::Rect Rec(rightWrist_x2, rightWrist_y2, 80, 80);
                                cv::Rect Rec1(leftWrist_x2, leftWrist_y2, 80, 80);
                                //cv::rectangle(Img, Rec, cv::Scalar(0, 255, 0), 0, 8, 0);
                                cv::rectangle(Img, Rec1, cv::Scalar(0, 255, 0), 0, 8, 0);

                                // Get ROI Image
                                //cv::Mat rightRoi = Img(Rec);
                                cv::Mat leftRoi = Img(Rec1);

                                //Resize Image 
                                //cv::resize(rightRoi, rightRoi, cv::Size(64, 64));
                                cv::resize(leftRoi, leftRoi, cv::Size(64, 64));

                                //cv::imshow("Hand Detection", Img);

                                if (counterImg > 0 && counterImg <= 100) {

                                    // <- path to save hand detection
                                    std::string filename;
                                    std::string strCounter = std::to_string(counterImg);
                                    std::stringstream strName;
                                    strName << "./Dataset/kontrol_ac1/gesture" << strCounter << ".jpg";

                                    strName >> filename;
                                    //cv::imshow("Crop", handCropped);
                                    std::cout << filename << std::endl;
                                    //writeCSV(filename, handCropped);
                                    //cv::imwrite(filename, leftRoi);

                                    // <= Compute Static left hand using HOG
                                    handFeature.compute(leftRoi, featureVector, Size(0, 0), Size(0, 0), locHand);

                                    Mat pose(featureVector, true);
                                    
                                    // <- Set dynamic gesture right hand
                                    if (counterImg == 1) {
                                        dynamicGesture = pose;
                                    }
                                    else {
                                        hconcat(pose, dynamicGesture, tmp);
                                        dynamicGesture = tmp;
                                        cout << counterImg << endl;
                                    }
                                    counterImg++;
                                }
                                else if (counterImg == 101) {
                                    writeCSV("./Dataset/dataset_kontrol_ac.csv", dynamicGesture);
                                }
                                else {
                                    //counterImg = 1;
                                }
                            }
                            else {

                                // <- image can't saved
                                std::cout << "don't saved" << std::endl;
                                //cv::Rect Rec(rightWrist_x2, rightWrist_y2, 80, 80);
                                cv::Rect Rec1(leftWrist_x2, leftWrist_y2, 80, 80);

                                //cv::rectangle(Img, Rec, cv::Scalar(0, 0, 255), 0, 8, 0);
                                cv::rectangle(Img, Rec1, cv::Scalar(0, 0, 255), 0, 8, 0);
                            }

                        }

                    }
                    op::printTime(opTimerHandDetection, "Time Hand Detection : ", " seconds.", op::Priority::High);
                }
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
        // <- Configuring OpenPose

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

        // <- Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
        configureWrapper(opWrapper);

        // <- Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // <- input camera
        cv::VideoCapture webcam;
        webcam.open(CAMERA_DEVICE);

        // <- set window resolution
        webcam.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH_WINDOWS);
        webcam.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT_WINDOWS);

        if (!webcam.isOpened()) {
            std::cout << "ERROR! unable to open camera" << std::endl;
        }
        cv::Mat frame;

        // <- Face Detection using HAAR CASCADE
        std::cout << "Face Detection file: ";
        cv::CascadeClassifier face_cascade;
        bool face_xml = face_cascade.load(OPENCV_CASCADE_FILENAME);

        if (face_xml == 0) {
            std::cerr << "Face xml did not load successfully..." << std::endl;
            return -1;
        }
        else {
            std::cout << "Face xml was successfully loaded..." << std::endl;
        }
        std::vector<cv::Rect> faces;
        cv::Mat frame_gray;
        op::Matrix imageToProcess;

        while (true) {
            const auto opTimer = op::getTimerInit();
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            
            webcam.read(frame);

            // <- check if we succeeded
            if (frame.empty()) {
                std::cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            
            const auto opTimerFaceDetection = op::getTimerInit();

            // <- face detecmultiscale
            cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
            face_cascade.detectMultiScale(frame_gray, faces, 2, 3, 0, cv::Size(30, 30));
            std::cout << std::endl;
            op::printTime(opTimerFaceDetection, "Time Face Detection : ", " seconds.", op::Priority::High);

            // <- Bounding box face detection
            for (auto&& feature : faces) {
                cv::rectangle(frame, feature, cv::Scalar(0, 255, 0), 2);
            }

            // <- show live and wait for a key with timeout long enough to show images
            if (faces.size() != 0) {
                // <- Display Face Detected
                    
                const auto opTimerSkeleton = op::getTimerInit();

                // <- Process and display image
                imageToProcess = OP_CV2OPCONSTMAT(frame);
                auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
                if (datumProcessed != nullptr)
                {
                    if (!FLAGS_no_display)
                        display(datumProcessed, frame);
                }
                else {
                    op::opLog("Image could not be processed.", op::Priority::High);
                }
                op::printTime(opTimerSkeleton, "Time Skeleton : ", " seconds.", op::Priority::High);
            }

            // <- Measuring total time
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            auto gpuFps = (1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
            op::printTime(opTimer, "Total time: ", " seconds.", op::Priority::High);

            // <- add text FPS to window
            std::stringstream ss;
            ss << "FPS : " << gpuFps;
            std::string strFps = ss.str();
            cv::putText(frame, strFps, cv::Point(10, 40), cv::FONT_HERSHEY_COMPLEX, 0.6, cvScalar(255, 255, 255));

            if (counterImg >= 100) {
                cv::putText(frame, "COMPLETE", cv::Point(500, 40), cv::FONT_HERSHEY_COMPLEX, 0.6, cvScalar(0, 0, 0));
            }
            cv::imshow("RGB", frame);

            if ( '1' == cv::waitKey(5)) {
                break;
            }
        }
      
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

    // Running OpenPose
    return Openpose();
}
