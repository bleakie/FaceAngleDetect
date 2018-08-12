#pragma once


#include "ldmarkmodel.h"

LinearRegressor::LinearRegressor() : weights(), meanvalue(), x(), isPCA(false)
{

}

cv::Mat LinearRegressor::predict(cv::Mat values)
{
	if (this->isPCA){
		cv::Mat mdata = values.colRange(0, values.cols - 2).clone();
		//            assert(mdata.cols==this->weights.rows && mdata.cols==this->meanvalue.cols);
		if (mdata.rows == 1){
			mdata = (mdata - this->meanvalue)*this->eigenvectors;
			cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols + 1, mdata.type());
			for (int i = 0; i < mdata.cols; i++){
				A.at<float>(i) = mdata.at<float>(i);
			}
			A.at<float>(A.cols - 1) = 1.0f;
			return A*this->x;
		}
		else{
			for (int i = 0; i < mdata.rows; i++){
				mdata.row(i) = mdata.row(i) - this->meanvalue;
			}
			mdata = mdata*this->eigenvectors;
			cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols + 1, mdata.type());
			for (int i = 0; i < mdata.rows; i++){
				for (int j = 0; j < mdata.cols; j++){
					A.at<float>(i, j) = mdata.at<float>(i, j);
				}
			}
			A.col(A.cols - 1) = cv::Mat::ones(A.rows, 1, A.type());
			return A*this->x;
		}
	}
	else{
		assert(values.cols == this->weights.rows);
		return  values*this->weights;
	}
}


ldmarkmodel::ldmarkmodel(){
	//{36,39,42,45,30,48,54};   {7,16,17,8,9,10,11};
	static int HeadPosePointIndexs[] = { 36, 39, 42, 45, 30, 48, 54 };
	estimateHeadPosePointIndexs = HeadPosePointIndexs;
	static float estimateHeadPose2dArray[] = {
		-0.208764, -0.140359, 0.458815, 0.106082, 0.00859783, -0.0866249, -0.443304, -0.00551231, -0.0697294,
		-0.157724, -0.173532, 0.16253, 0.0935172, -0.0280447, 0.016427, -0.162489, -0.0468956, -0.102772,
		0.126487, -0.164141, 0.184245, 0.101047, 0.0104349, -0.0243688, -0.183127, 0.0267416, 0.117526,
		0.201744, -0.051405, 0.498323, 0.0341851, -0.0126043, 0.0578142, -0.490372, 0.0244975, 0.0670094,
		0.0244522, -0.211899, -1.73645, 0.0873952, 0.00189387, 0.0850161, 1.72599, 0.00521321, 0.0315345,
		-0.122839, 0.405878, 0.28964, -0.23045, 0.0212364, -0.0533548, -0.290354, 0.0718529, -0.176586,
		0.136662, 0.335455, 0.142905, -0.191773, -0.00149495, 0.00509046, -0.156346, -0.0759126, 0.133053,
		-0.0393198, 0.307292, 0.185202, -0.446933, -0.0789959, 0.29604, -0.190589, -0.407886, 0.0269739,
		-0.00319206, 0.141906, 0.143748, -0.194121, -0.0809829, 0.0443648, -0.157001, -0.0928255, 0.0334674,
		-0.0155408, -0.145267, -0.146458, 0.205672, -0.111508, 0.0481617, 0.142516, -0.0820573, 0.0329081,
		-0.0520549, -0.329935, -0.231104, 0.451872, -0.140248, 0.294419, 0.223746, -0.381816, 0.0223632,
		0.176198, -0.00558382, 0.0509544, 0.0258391, 0.050704, -1.10825, -0.0198969, 1.1124, 0.189531,
		-0.0352285, 0.163014, 0.0842186, -0.24742, 0.199899, 0.228204, -0.0721214, -0.0561584, -0.157876,
		-0.0308544, -0.131422, -0.0865534, 0.205083, 0.161144, 0.197055, 0.0733392, -0.0916629, -0.147355,
		0.527424, -0.0592165, 0.0150818, 0.0603236, 0.640014, -0.0714241, -0.0199933, -0.261328, 0.891053 };
	estimateHeadPoseMat = cv::Mat(15, 9, CV_32FC1, estimateHeadPose2dArray);
	static float estimateHeadPose2dArray2[] = {
		0.139791, 27.4028, 7.02636,
		-2.48207, 9.59384, 6.03758,
		1.27402, 10.4795, 6.20801,
		1.17406, 29.1886, 1.67768,
		0.306761, -103.832, 5.66238,
		4.78663, 17.8726, -15.3623,
		-5.20016, 9.29488, -11.2495,
		-25.1704, 10.8649, -29.4877,
		-5.62572, 9.0871, -12.0982,
		-5.19707, -8.25251, 13.3965,
		-23.6643, -13.1348, 29.4322,
		67.239, 0.666896, 1.84304,
		-2.83223, 4.56333, -15.885,
		-4.74948, -3.79454, 12.7986,
		-16.1, 1.47175, 4.03941 };
	estimateHeadPoseMat2 = cv::Mat(15, 3, CV_32FC1, estimateHeadPose2dArray2);
}


int ldmarkmodel::track(const cv::Mat& src, cv::Mat& current_shape, bool isDetFace){
	cv::Mat grayImage;
	if (src.channels() == 1){
		grayImage = src;
	}
	else if (src.channels() == 3){
		cv::cvtColor(src, grayImage, CV_BGR2GRAY);
	}
	else if (src.channels() == 4){
		cv::cvtColor(src, grayImage, CV_RGBA2GRAY);
	}
	else{
		return SDM_ERROR_IMAGE;
	}

	if (!current_shape.empty()){
		faceBox = get_enclosing_bbox(current_shape);
	}
	else{
		faceBox = cv::Rect(0, 0, 0, 0);
	}
	cv::Rect mfaceBox = faceBox & cv::Rect(0, 0, grayImage.cols, grayImage.rows);

	int error_code = SDM_NO_ERROR;
	vector<Point> ldmark_point;
	std::vector<cv::Rect> mFaceRects;
	//my add face_detect
	int * pResults = NULL;
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
		return 0;
	pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(grayImage.ptr(0)), grayImage.cols, grayImage.rows, (int)grayImage.step,
		1.2f, 2, 60, 0, 1);
	int peopleNUM = (pResults ? *pResults : 0);//几张人脸
	float Area = 0.0f;
	for (int i = 0; i < peopleNUM; i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		float area = p[2] * p[3];
		if (Area < area)
		{
			cv::Rect opencvRect(p[0], p[1], p[2], p[3]); 
			if (calcSafeRect(opencvRect, src))
			{
				cv::Mat face = src;
				rectangle(face, opencvRect, Scalar(0, 255, 0), 2);
				mFaceRects.push_back(opencvRect);
				for (int j = 0; j < 68; j++)
					ldmark_point.push_back(Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]));
			}
		}
	}

	free(pBuffer);
	if (mFaceRects.size() <= 0){
		current_shape = cv::Mat();
		return SDM_ERROR_FACENO;
	}
	faceBox = mFaceRects[0];
	for (int i = 1; i < mFaceRects.size(); i++){
		if (faceBox.area() < mFaceRects[i].area())
			faceBox = mFaceRects[i];
	}
	error_code = SDM_ERROR_FACEDET;

	int numLandmarks = current_shape.cols / 2;
	for (int j = 0; j < numLandmarks; j++){
		current_shape.at<float>(j) = ldmark_point[j].x;
		current_shape.at<float>(j + numLandmarks) = ldmark_point[j].y;
	}

	for (int i = 0; i < LinearRegressors.size(); i++){
		cv::Mat Descriptor = CalculateHogDescriptor(grayImage, current_shape, LandmarkIndexs.at(i), eyes_index, HoGParams.at(i));
		cv::Mat update_step = LinearRegressors.at(i).predict(Descriptor);
		if (isNormal){
			float lx = (current_shape.at<float>(eyes_index.at(0)) + current_shape.at<float>(eyes_index.at(1)))*0.5;
			float ly = (current_shape.at<float>(eyes_index.at(0) + numLandmarks) + current_shape.at<float>(eyes_index.at(1) + numLandmarks))*0.5;
			float rx = (current_shape.at<float>(eyes_index.at(2)) + current_shape.at<float>(eyes_index.at(3)))*0.5;
			float ry = (current_shape.at<float>(eyes_index.at(2) + numLandmarks) + current_shape.at<float>(eyes_index.at(3) + numLandmarks))*0.5;
			float distance = sqrt((rx - lx)*(rx - lx) + (ry - ly)*(ry - ly));
			update_step = update_step*distance;
		}
		current_shape = current_shape + update_step;
	}
	return error_code;
}


bool ldmarkmodel::calcSafeRect(cv::Rect& roi_rect, const Mat& src)
{
	// boudRect的左上的x和y有可能小于0
	float tl_x = roi_rect.x > 0 ? roi_rect.x : 0;
	float tl_y = roi_rect.y > 0 ? roi_rect.y : 0;
	// boudRect的右下的x和y有可能大于src的范围
	float br_x = roi_rect.x + roi_rect.width < src.cols ?
		roi_rect.x + roi_rect.width - 1 : src.cols - 1;
	float br_y = roi_rect.y + roi_rect.height < src.rows ?
		roi_rect.y + roi_rect.height - 1 : src.rows - 1;

	float roi_width = br_x - tl_x;
	float roi_height = br_y - tl_y;

	if (roi_width <= 0 || roi_height <= 0)
		return false;

	// 新建一个mat，确保地址不越界，以防mat定位roi时抛异常
	roi_rect = Rect_<float>(tl_x, tl_y, roi_width, roi_height);
	//ExpandRect(roi_rect, src.size());
	return true;
}

void ldmarkmodel::EstimateHeadPose(cv::Mat &current_shape, cv::Vec3d &eav){
	if (current_shape.empty())
		return;
	static const int samplePdim = 7;
	float miny = 10000000000.0f;
	float maxy = 0.0f;
	float sumx = 0.0f;
	float sumy = 0.0f;
	for (int i = 0; i<samplePdim; i++){
		sumx += current_shape.at<float>(estimateHeadPosePointIndexs[i]);
		float y = current_shape.at<float>(estimateHeadPosePointIndexs[i] + current_shape.cols / 2);
		sumy += y;
		if (miny > y)
			miny = y;
		if (maxy < y)
			maxy = y;
	}
	float dist = maxy - miny;
	sumx = sumx / samplePdim;
	sumy = sumy / samplePdim;
	cv::Mat tmp(1, 2 * samplePdim + 1, CV_32FC1);
	for (int i = 0; i < samplePdim; i++){
		tmp.at<float>(i) = (current_shape.at<float>(estimateHeadPosePointIndexs[i]) - sumx) / dist;
		tmp.at<float>(i + samplePdim) = (current_shape.at<float>(estimateHeadPosePointIndexs[i] + current_shape.cols / 2) - sumy) / dist;
	}
	tmp.at<float>(2 * samplePdim) = 1.0f;

	cv::Mat predict = tmp*estimateHeadPoseMat2;
	eav[0] = predict.at<float>(0);
	eav[1] = predict.at<float>(1);
	eav[2] = predict.at<float>(2);
	tmp.release();
	return;
}

void ldmarkmodel::drawPose(cv::Mat& img, const cv::Mat& current_shape, float lineL)
{
	if (current_shape.empty())
		return;
	static const int samplePdim = 7;
	float miny = 10000000000.0f;
	float maxy = 0.0f;
	float sumx = 0.0f;
	float sumy = 0.0f;
	for (int i = 0; i<samplePdim; i++){
		sumx += current_shape.at<float>(estimateHeadPosePointIndexs[i]);
		float y = current_shape.at<float>(estimateHeadPosePointIndexs[i] + current_shape.cols / 2);
		sumy += y;
		if (miny > y)
			miny = y;
		if (maxy < y)
			maxy = y;
	}
	float dist = maxy - miny;
	sumx = sumx / samplePdim;
	sumy = sumy / samplePdim;
	static cv::Mat tmp(1, 2 * samplePdim + 1, CV_32FC1);
	for (int i = 0; i < samplePdim; i++){
		tmp.at<float>(i) = (current_shape.at<float>(estimateHeadPosePointIndexs[i]) - sumx) / dist;
		tmp.at<float>(i + samplePdim) = (current_shape.at<float>(estimateHeadPosePointIndexs[i] + current_shape.cols / 2) - sumy) / dist;
	}
	tmp.at<float>(2 * samplePdim) = 1.0f;
	cv::Mat predict = tmp*estimateHeadPoseMat;
	cv::Mat rot(3, 3, CV_32FC1);
	for (int i = 0; i < 3; i++){
		rot.at<float>(i, 0) = predict.at<float>(3 * i);
		rot.at<float>(i, 1) = predict.at<float>(3 * i + 1);
		rot.at<float>(i, 2) = predict.at<float>(3 * i + 2);
	}
	//we have get the rot mat
	int loc[2] = { 70, 70 };
	int thickness = 2;
	int lineType = 8;

	cv::Mat P = (cv::Mat_<float>(3, 4) <<
		0, lineL, 0, 0,
		0, 0, -lineL, 0,
		0, 0, 0, -lineL);
	P = rot.rowRange(0, 2)*P;
	P.row(0) += loc[0];
	P.row(1) += loc[1];
	cv::Point p0(P.at<float>(0, 0), P.at<float>(1, 0));

	line(img, p0, cv::Point(P.at<float>(0, 1), P.at<float>(1, 1)), cv::Scalar(255, 0, 0), thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0, 2), P.at<float>(1, 2)), cv::Scalar(0, 255, 0), thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0, 3), P.at<float>(1, 3)), cv::Scalar(0, 0, 255), thickness, lineType);

	//printf("%f %f %f\n", rot.at<float>(0, 0), rot.at<float>(0, 1), rot.at<float>(0, 2));
	//printf("%f %f %f\n", rot.at<float>(1, 0), rot.at<float>(1, 1), rot.at<float>(1, 2));

	cv::Vec3d eav;
	cv::Mat tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
	double _pm[12] = { rot.at<float>(0, 0), rot.at<float>(0, 1), rot.at<float>(0, 2), 0,
		rot.at<float>(1, 0), rot.at<float>(1, 1), rot.at<float>(1, 2), 0,
		rot.at<float>(2, 0), rot.at<float>(2, 1), rot.at<float>(2, 2), 0 };
	cv::decomposeProjectionMatrix(cv::Mat(3, 4, CV_64FC1, _pm), tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, eav);
	std::stringstream ss;
	ss << eav[0];
	std::string txt = "Pitch: " + ss.str();
	cv::putText(img, txt, cv::Point(60, 20), 0.5, 0.5, cv::Scalar(0, 0, 255));
	std::stringstream ss1;
	ss1 << eav[1];
	std::string txt1 = "Yaw: " + ss1.str();
	cv::putText(img, txt1, cv::Point(60, 40), 0.5, 0.5, cv::Scalar(0, 0, 255));
	std::stringstream ss2;
	ss2 << eav[2];
	std::string txt2 = "Roll: " + ss2.str();
	cv::putText(img, txt2, cv::Point(60, 60), 0.5, 0.5, cv::Scalar(0, 0, 255));

	predict = tmp*estimateHeadPoseMat2;
	std::stringstream ss3;
	ss3 << predict.at<float>(0);
	txt = "Pitch: " + ss3.str();
	cv::putText(img, txt, cv::Point(340, 20), 0.5, 0.5, cv::Scalar(255, 255, 255));
	std::stringstream ss4;
	ss4 << predict.at<float>(1);
	txt1 = "Yaw: " + ss4.str();
	cv::putText(img, txt1, cv::Point(340, 40), 0.5, 0.5, cv::Scalar(255, 255, 255));
	std::stringstream ss5;
	ss5 << predict.at<float>(2);
	txt2 = "Roll: " + ss5.str();
	cv::putText(img, txt2, cv::Point(340, 60), 0.5, 0.5, cv::Scalar(255, 255, 255));
	//        Pitch = eav[0];
	//        Yaw	  = eav[1];
	//        Roll  = eav[2];
}
