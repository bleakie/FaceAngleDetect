#include <vector>
#include <iostream>
#include <fstream>
#include "ldmarkmodel.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"


using namespace std;
using namespace cv;


int main()
{
	//使用libface的68个点人脸对齐
    ldmarkmodel modelt;

    cv::VideoCapture mCamera(0);
    if(!mCamera.isOpened()){
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
	cv::Mat Image;
    for(;;){
		mCamera >> Image;
		cv::Mat current_shape(1, 136, CV_32FC1);//用于存放人脸对齐点
        modelt.track(Image, current_shape);
        cv::Vec3d eav;
        modelt.EstimateHeadPose(current_shape, eav);
        modelt.drawPose(Image, current_shape, 50);

        int numLandmarks = current_shape.cols/2;
        for(int j=0; j<numLandmarks; j++){
            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);
            std::stringstream ss;
            ss << j;
            cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Camera", Image);
        if(27 == cv::waitKey(5)){
            cv::destroyAllWindows();
            break;
        }
    }

    system("pause");
    return 0;
}






















