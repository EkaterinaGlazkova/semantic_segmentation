#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

template <typename T>
void Output(std::vector<std::vector<T>> array) {
    for (std::vector<T> line : array) {
        for (T elem: line) {
            std::cout << elem << " ";
        }
        std::cout << "\n";
    }
}

template <typename T>
void Output(std::vector<T> array) {
    for (T elem : array) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

int main()
{
    int CATEGORIES_NUMBER = 6;
    std::string FIRST_FILE_NAME = "top_potsdam_2_11_label.tif";
    std::string SECOND_FILE_NAME = "top_potsdam_2_11_label_1.tif";
    Mat ideal_result, predicted_result;
    ideal_result = imread(FIRST_FILE_NAME, CV_LOAD_IMAGE_COLOR);
    predicted_result = imread(SECOND_FILE_NAME, CV_LOAD_IMAGE_COLOR);
    
    imshow("image", ideal_result);
    
    cv::Size size = ideal_result.size();
    int lines = size.height;
    int rows = size.width;
    
    std::vector<std::vector <int>> results(CATEGORIES_NUMBER);
    std::vector<std::vector<double>> normalized_results(CATEGORIES_NUMBER);
    for (int line  = 0; line < CATEGORIES_NUMBER; ++line) {
        results[line].resize(CATEGORIES_NUMBER, 0);
        normalized_results[line].resize(CATEGORIES_NUMBER, 0);
    }
    
    std::vector<long long> TP(6, 0), FP(6, 0), FN(6, 0);
    
    std::vector<Vec3b> categories(6);
    categories[0]  = Vec3b(255, 255, 255);
    categories[1]  = Vec3b(0, 0, 255);
    categories[2]  = Vec3b(0, 255, 255);
    categories[3]  = Vec3b(0, 255, 0);
    categories[4]  = Vec3b(255, 255, 0);
    categories[5]  = Vec3b(255, 0, 0);
    
    for (int line = 0; line < lines; ++line) {
        for (int row = 0; row < rows; ++row){
            size_t ideal_pixel_category = 0, predicted_pixel_category = 0;
            for (size_t category_index = 0; category_index < categories.size(); ++category_index){
                if (ideal_result.at<Vec3b>(line, row) == categories[category_index]) {
                    ideal_pixel_category = category_index;
                }
                if (predicted_result.at<Vec3b>(line, row) == categories[category_index]) {
                    predicted_pixel_category = category_index;
                }
            }
            ++results[ideal_pixel_category][predicted_pixel_category];
            if (ideal_pixel_category == predicted_pixel_category) {
                ++TP[ideal_pixel_category];
            } else {
                ++FN[ideal_pixel_category];
                ++FP[predicted_pixel_category];
            }
        }
    }
    
    std::vector<double> precision(CATEGORIES_NUMBER),
    recall(CATEGORIES_NUMBER),
    accuracy(CATEGORIES_NUMBER),
    f1(CATEGORIES_NUMBER);
    
    for (size_t category_index = 0; category_index < categories.size(); ++category_index){
        precision[category_index] = (double) TP[category_index] /(TP[category_index] + FP[category_index]);
        recall[category_index] = (double) TP[category_index] /(TP[category_index] + FN[category_index]);
        accuracy[category_index] = (double) (lines*rows - (FN[category_index]+FP[category_index]))/ (lines*rows);
        f1[category_index] = (double) 2*TP[category_index]/(2*TP[category_index] + FP[category_index] + FN[category_index]);
    }
    
    Output(results);
    std::cout << "TP\n";
    Output(TP);
    std::cout << "FP\n";
    Output(FP);
    std::cout << "FN\n";
    Output(FN);
    std::cout << "precision\n";
    Output(precision);
    std::cout << "recall\n";
    Output(recall);
    std::cout << "accuracy\n";
    Output(accuracy);
    std::cout << "f1\n";
    Output(f1);
    
    
    return 0;
}
