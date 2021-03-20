#include "SVM.h"
#include <vector>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// <- path load dataset
#define PATH "./Dataset/dataset_kontrol_tv.txt"

// <- path image
#define PATH_IMAGE "./Dataset/kontrol_tv1/gesture"

// <- label class dataset
#define LABEL 1

using namespace std;
using namespace cv;


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

int generateDataset() {
	Mat frame;

	// <- inisialization HOG
	HOGDescriptor handFeature(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	vector<float> featureVector;
	vector<Point> locHand;

	Mat dataset, tmp;

	// <- Load dataset
	ofstream loadDataset;
	loadDataset.open(PATH);

	for (int i = 1; i <= 100; i++) {
		// <- convert to string
		string count = to_string(i);
		string filename;
		stringstream strName;

		// <- path image
		strName << PATH_IMAGE << count << ".jpg";
		strName >> filename;
		cout << filename << endl;

		// <- read image
		frame = imread(filename, 1);
		cout << i << endl;
		if (frame.empty()) {
			return -1;
		}

		// <- compute HOG
		handFeature.compute(frame, featureVector, Size(0, 0), Size(0, 0), locHand);

		// <- convert vector to matrix
		Mat pose(featureVector, true);

		// <- add column matrix
		if (i == 1) {
			dataset = pose;
		}
		else {
			hconcat(pose, dataset, dataset);
		}
	}
	cout << dataset.cols << " || " << dataset.rows << endl;

	// <- load dataset
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 1764; j++) {
			if (j == 0) {
				loadDataset << LABEL << " ";
			}
			else {
				loadDataset << j << ":" << /*tmpData.col(i-1)*/ dataset.at<float>(j, i) << " ";
			}
		}
		loadDataset << endl;
	}
	loadDataset.close();
	return 0;
}




//int main() {
//	generateDataset();
//}


int main() {
	// <- Clasifier SVM

	FILE* input, * output;


	chrono::high_resolution_clock::time_point start_classification = chrono::high_resolution_clock::now();


	output = fopen("./output.txt", "w");
	if (output == NULL)
	{
		fprintf(stderr, "Output does not open\n");
		exit(1);
	}
	if ((model = svm_load_model("./dataset.model")) == 0)
	{
		fprintf(stderr, "Model does not open\n");
		exit(1);
	}
	x = (struct svm_node*)malloc(max_nr_attr * sizeof(struct svm_node));
	if (predict_probability)
	{
		if (svm_check_probability_model(model) == 0)
		{
			fprintf(stderr, "Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if (svm_check_probability_model(model) != 0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

		// <- input data
	input = fopen("./testing/realtime_predict.txt", "r");
	if (input == NULL)
	{
		fprintf(stderr, "File input does not open\n");

		exit(1);
	}

	predict(input, output);
	svm_free_and_destroy_model(&model);
	free(x);
	free(baris);
	fclose(input);
	fclose(output);

	chrono::high_resolution_clock::time_point end_classification = chrono::high_resolution_clock::now();
	auto clssification_ms = chrono::duration_cast<chrono::milliseconds>(end_classification - start_classification).count();
	cout << "hasil klasifikasi = ";
	if (h_klasifikasi == 1)
	{
		cout << "Kontrol TV" << endl;
		//show_text(Mat_Depth, Point(300, 20), "Berdiri", "", 255, 255, 255);
	}
	if (h_klasifikasi == 2)
	{
		cout << "Kontrol AC" << endl;
		//show_text(Mat_Depth, Point(300, 20), "Duduk", "", 255, 255, 255);
	}

	cout << "Time Classification : " << clssification_ms << " ms" << endl;


	return 0;
}