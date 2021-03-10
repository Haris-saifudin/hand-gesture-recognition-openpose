//#include "stdafx.h"
#include "stdio.h"
#include <vector>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>
#include <valarray>
#include <complex>
#include <ctime>
#include <chrono>
#include "SVM.h"

#define FILE_INPUT "./data_training/realtime_data.txt"
#define FILE_OUTPUT "./data_training/realtime_data.txt"
#define MODEL "./data_training/realtime_data.txt"


using namespace std;

//##### SVM Declaration #####*/
int h_klasifikasi = 0;
static int(*info)(const char* fmt, ...) = &printf;
struct svm_node* x;
int max_nr_attr = 64;
struct svm_model* model;
int predict_probability = 0;
static char* baris;
static int max_line_len;


//### Create dataset ###

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
		info("Accuracy = %g%% (%d/%d) (classification) -> ",
			(double)correct / total * 100, correct, total);
	if (predict_probability)
		free(prob_estimates);
}

int main() {
	//##### SVM Classifier #####
	FILE* input, * output;
	input = fopen(FILE_INPUT, "r"); //data input
	if (input == NULL)
	{
		cout << "Can't open file input..." << endl
			 << "Path : " << FILE_INPUT << endl;
		return -1;
	}
	chrono::high_resolution_clock::time_point start_classification = chrono::high_resolution_clock::now();


	output = fopen(FILE_OUTPUT, "w");
	if (output == NULL)
	{
		cout << "Can't open file output..." << endl
			<< "Path : " << FILE_OUTPUT << endl;
		return -1;
	}
	if ((model = svm_load_model(MODEL)) == 0)
	{
		cout << "Can't open file model..." << endl
			<< "Path : " << MODEL << endl;
		return -1;
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
	cout << "Load model..." << endl;

	predict(input, output);
	svm_free_and_destroy_model(&model);
	free(x);
	free(baris);
	fclose(input);
	fclose(output);

	chrono::high_resolution_clock::time_point end_classification = chrono::high_resolution_clock::now();
	auto clssification_ms = chrono::duration_cast<chrono::milliseconds>(end_classification - start_classification).count();
	if (h_klasifikasi == 1){
		cout << "Control TV.." << endl;
		//show_text(Mat_Depth, Point(300, 20), "Berdiri", "", 255, 255, 255);
	}
	else if (h_klasifikasi == 2){
		cout << "Control AC.." << endl;
		//show_text(Mat_Depth, Point(300, 20), "Duduk", "", 255, 255, 255);
	}
	else {
		cout << "Negative Prediction.." << endl;
	}

	cout << "Time Classification : " << clssification_ms << " ms" << endl;

	return 0;
}