// Kinect2.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include <iostream>
#include <Windows.h>
#include <kinect.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/contrib/contrib.hpp>
#include "opencv2/video/video.hpp"
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>

#include <iterator>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;
using namespace cv;


typedef struct BodyInfoStructs
{
	Mat bodyInfoMat;
	float skeletonInfo[9 * JointType_Count];
};


#define CHANNEL 4
int desiredColorFrameHeight = 270;
int desiredColorFrameWidth = 480;

UINT16 *depthFrameBuffer;

Mat drawAperson(CvPoint *jointsPoints, CvScalar skeletonColor, int skeletonThickness, Mat tmpSkeletonMat);


int bodyFrameHeight = 1080;
int bodyFrameWidth = 1920;  

int depthFrameHeight = 1080;
int depthFrameWidth = 1920;

int colorFrameHeight = 1080;
int colorFrameWidth = 1920;


int bodyIndexFrameHeight = 1080;
int bodyIndexFrameWidth = 1920;

UINT8 *colorFrameBuffer = new UINT8[colorFrameHeight*colorFrameWidth*CHANNEL];
RGBQUAD* depthFrameRGBXMat = new RGBQUAD[depthFrameWidth * depthFrameHeight];
RGBQUAD* bodyIndexFrameRGBXMat = new RGBQUAD[bodyIndexFrameWidth * bodyIndexFrameHeight];



IKinectSensor *kinectSensor = nullptr;
IMultiSourceFrame * multiSourceFrame = nullptr;
IMultiSourceFrameReader * multiSourceFrameReader = nullptr;
ICoordinateMapper* multisourceCoordinateMapper;

string DESTINATIONPATH = "C:\\Users\\appl\\Desktop\\SampleOutput";




bool InitializingKinect() {

	if (FAILED(GetDefaultKinectSensor(&kinectSensor))) {
		return false;
	}

	if (kinectSensor) {
		kinectSensor->get_CoordinateMapper(&multisourceCoordinateMapper);

		kinectSensor->Open();
		kinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Depth |
			FrameSourceTypes::FrameSourceTypes_Color |
			FrameSourceTypes::FrameSourceTypes_Body |
			FrameSourceTypes::FrameSourceTypes_BodyIndex,
			&multiSourceFrameReader);
		return multiSourceFrameReader;
	}
	else {
		return false;
	}
}

BOOLEAN tracked;


BodyInfoStructs GetBodyFrame(IMultiSourceFrame *multiSourceFrame, int mappingDestFlag) {

	IBodyFrame* bodyFrame = nullptr;
	IBodyFrameReference* bodyFrameReference = nullptr;

	Mat resultMat;
	BodyInfoStructs bodyInfo;
	for (int i =0; i< 9 * JointType_Count; i++)
		bodyInfo.skeletonInfo[i] = 0;
	//if (mappingDestFlag == 1)
		//resultMat = Mat::zeros(colorFrameHeight, colorFrameWidth, CV_8UC4);
	//if (mappingDestFlag == 2)
		//resultMat = Mat::zeros(depthFrameHeight, depthFrameWidth, CV_8UC4);
	resultMat = Mat::zeros(0, 0, CV_8UC4);
	if (SUCCEEDED(multiSourceFrame->get_BodyFrameReference(&bodyFrameReference))) {
		if (SUCCEEDED(bodyFrameReference->AcquireFrame(&bodyFrame))) {

			IBody* bodies[BODY_COUNT] = { 0 };

			if (SUCCEEDED(bodyFrame->GetAndRefreshBodyData(BODY_COUNT, bodies))) {

				IBody* body;


				//Currently Just get one body
				for (int i = 0; i < BODY_COUNT; i++) {
					body = bodies[i];
					Joint joints[JointType_Count];
					JointOrientation jointsOrientations[JointType_Count];
					UINT colorStep = (i / 3 + 1) << 6;
					UINT colorMask = 1 << (i % 3);
					CvScalar skeletonColor = cvScalar((float)(((colorMask) & 1) * colorStep),
						(float)((((colorMask) & 2) >> 1) * colorStep),
						(float)((((colorMask) & 4) >> 2) * colorStep));

					if (body) {
						BOOLEAN bodyTracked = false;
						CvPoint jointsPoints[JointType_Count] = { cvPoint(-1,-1) };
						if (SUCCEEDED(body->get_IsTracked(&bodyTracked))) {

							// Both of joints position and orientations should be gotten. 

							if (bodyTracked) {
								if (SUCCEEDED(body->GetJoints(JointType_Count, joints))) {
									if (mappingDestFlag == 1) {
										ColorSpacePoint tmpColorSpacePoint;

										for (int i = 0; i < JointType_Count; i++) {
											if (joints[i].TrackingState > 0) {

												bodyInfo.skeletonInfo[0 + 9 * i] = joints[i].Position.X;
												bodyInfo.skeletonInfo[1 + 9 * i] = joints[i].Position.Y;
												bodyInfo.skeletonInfo[2 + 9 * i] = joints[i].Position.Z;

												if (SUCCEEDED(body->GetJointOrientations(JointType_Count, jointsOrientations))) {
													bodyInfo.skeletonInfo[3 + 9 * i] = jointsOrientations[i].Orientation.x;
													bodyInfo.skeletonInfo[4 + 9 * i] = jointsOrientations[i].Orientation.y;
													bodyInfo.skeletonInfo[5 + 9 * i] = jointsOrientations[i].Orientation.z;
													bodyInfo.skeletonInfo[6 + 9 * i] = jointsOrientations[i].Orientation.w;
												}

												if (SUCCEEDED(multisourceCoordinateMapper->MapCameraPointToColorSpace(joints[i].Position, &tmpColorSpacePoint))) {
													jointsPoints[i].x = (int)tmpColorSpacePoint.X;
													jointsPoints[i].y = (int)tmpColorSpacePoint.Y;
													bodyInfo.skeletonInfo[8] = jointsPoints[i].y;
													bodyInfo.skeletonInfo[7] = jointsPoints[i].x;
												}
											}
										}
									}

									if (mappingDestFlag == 2) {

										DepthSpacePoint tmpDepthSpacePoint;

										for (int i = 0; i < JointType_Count; i++) {
											if (joints[i].TrackingState > 0) {

												bodyInfo.skeletonInfo[0 + 9 * i] = joints[i].Position.X;
												bodyInfo.skeletonInfo[1 + 9 * i] = joints[i].Position.Y;
												bodyInfo.skeletonInfo[2 + 9 * i] = joints[i].Position.Z;
												if (SUCCEEDED(body->GetJointOrientations(JointType_Count, jointsOrientations))) {
													bodyInfo.skeletonInfo[3 + 9 * i] = jointsOrientations[i].Orientation.x;
													bodyInfo.skeletonInfo[4 + 9 * i] = jointsOrientations[i].Orientation.y;
													bodyInfo.skeletonInfo[5 + 9 * i] = jointsOrientations[i].Orientation.z;
													bodyInfo.skeletonInfo[6 + 9 * i] = jointsOrientations[i].Orientation.w;
												}

												if (SUCCEEDED(multisourceCoordinateMapper->MapCameraPointToDepthSpace(joints[i].Position, &tmpDepthSpacePoint))) {
													bodyInfo.skeletonInfo[8 + 9 * i] = jointsPoints[i].x = (int)tmpDepthSpacePoint.X;
													bodyInfo.skeletonInfo[7 + 9 * i] = jointsPoints[i].y = (int)tmpDepthSpacePoint.Y;
												}
											}
										}
									}

								}
								if (mappingDestFlag == 1)
									resultMat = Mat::zeros(colorFrameHeight, colorFrameWidth, CV_8UC4);
								if (mappingDestFlag == 2)
									resultMat = Mat::zeros(depthFrameHeight, depthFrameWidth, CV_8UC4);
								resultMat = drawAperson(jointsPoints, skeletonColor, 10, resultMat);
								break;
								body->Release();
							}
						}
					}
					body->Release();
				}
			}
			if (bodyFrame) bodyFrame->Release();
		}
		if (bodyFrameReference) bodyFrameReference->Release();
	}
	bodyInfo.bodyInfoMat = resultMat;
	return bodyInfo;
}




Mat drawAperson(CvPoint *jointsPoints, CvScalar skeletonColor, int skeletonThickness, Mat tmpSkeletonMat) {


	//HEAD + NECK
	if ((jointsPoints[JointType_Head].x >= 0) && (jointsPoints[JointType_Neck].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_Head], jointsPoints[JointType_Neck], skeletonColor, skeletonThickness);

	//NECK + SPINE SHOULDER;
	if ((jointsPoints[JointType_Neck].x >= 0) && (jointsPoints[JointType_SpineShoulder].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_Neck], jointsPoints[JointType_SpineShoulder], skeletonColor, skeletonThickness);

	//SPINE SHOULDER + RIGHT SHOULDER
	if ((jointsPoints[JointType_SpineShoulder].x >= 0) && (jointsPoints[JointType_ShoulderRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_SpineShoulder], jointsPoints[JointType_ShoulderRight], skeletonColor, skeletonThickness);

	//SPINE SHOULDER + LEFT SHOULDER
	if ((jointsPoints[JointType_SpineShoulder].x >= 0) && (jointsPoints[JointType_ShoulderLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_SpineShoulder], jointsPoints[JointType_ShoulderLeft], skeletonColor, skeletonThickness);
	//RIGHT SHOULDER + RIGHT ELBOW

	if ((jointsPoints[JointType_ShoulderRight].x >= 0) && (jointsPoints[JointType_ElbowRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_ShoulderRight], jointsPoints[JointType_ElbowRight], skeletonColor, skeletonThickness);

	//LEFT SHOULDER + LEFT ELBOW
	if ((jointsPoints[JointType_ShoulderLeft].x >= 0) && (jointsPoints[JointType_ElbowLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_ShoulderLeft], jointsPoints[JointType_ElbowLeft], skeletonColor, skeletonThickness);

	// RIGHT ELBOW + RIGHT WRIST
	if ((jointsPoints[JointType_ElbowRight].x >= 0) && (jointsPoints[JointType_WristRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_ElbowRight], jointsPoints[JointType_WristRight], skeletonColor, skeletonThickness);

	//LEFT ELBOW + LEFT WRIST
	if ((jointsPoints[JointType_ElbowLeft].x >= 0) && (jointsPoints[JointType_WristLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_ElbowLeft], jointsPoints[JointType_WristLeft], skeletonColor, skeletonThickness);

	//RIGHT WRIST + RIGHT HAND
	if ((jointsPoints[JointType_WristRight].x >= 0) && (jointsPoints[JointType_HandRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_WristRight], jointsPoints[JointType_HandRight], skeletonColor, skeletonThickness);

	//LEFT WRIST + LEFT HAND
	if ((jointsPoints[JointType_WristLeft].x >= 0) && (jointsPoints[JointType_HandLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_WristLeft], jointsPoints[JointType_HandLeft], skeletonColor, skeletonThickness);

	//RIGHT HAND + RIGHT THUMB
	if ((jointsPoints[JointType_HandRight].x >= 0) && (jointsPoints[JointType_ThumbRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_HandRight], jointsPoints[JointType_ThumbRight], skeletonColor, skeletonThickness);

	//LEFT HAND + LEFT THUMB
	if ((jointsPoints[JointType_HandLeft].x >= 0) && (jointsPoints[JointType_ThumbLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_HandLeft], jointsPoints[JointType_ThumbLeft], skeletonColor, skeletonThickness);

	//RIGHT HAND + RIGHT HANDTIP
	if ((jointsPoints[JointType_HandRight].x >= 0) && (jointsPoints[JointType_HandTipRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_HandRight], jointsPoints[JointType_HandTipRight], skeletonColor, skeletonThickness);

	//LEFT HAND + LEFT HANDTIP
	if ((jointsPoints[JointType_HandLeft].x >= 0) && (jointsPoints[JointType_HandTipLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_HandLeft], jointsPoints[JointType_HandTipLeft], skeletonColor, skeletonThickness);

	//SPINE SHOULDER + SPINE MID
	if ((jointsPoints[JointType_SpineShoulder].x >= 0) && (jointsPoints[JointType_SpineMid].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_SpineShoulder], jointsPoints[JointType_SpineMid], skeletonColor, skeletonThickness);

	//SPINE MID + SPINE BASE
	if ((jointsPoints[JointType_SpineMid].x >= 0) && (jointsPoints[JointType_SpineBase].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_SpineMid], jointsPoints[JointType_SpineBase], skeletonColor, skeletonThickness);

	//SPINE BASE + RIGHT HIP
	if ((jointsPoints[JointType_SpineBase].x >= 0) && (jointsPoints[JointType_HipRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_SpineBase], jointsPoints[JointType_HipRight], skeletonColor, skeletonThickness);

	//SPINE BASE + LEFT HIP
	if ((jointsPoints[JointType_SpineBase].x >= 0) && (jointsPoints[JointType_HipLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_SpineBase], jointsPoints[JointType_HipLeft], skeletonColor, skeletonThickness);

	//RIGHT HIP + RIGHT KNEE
	if ((jointsPoints[JointType_HipRight].x >= 0) && (jointsPoints[JointType_KneeRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_HipRight], jointsPoints[JointType_KneeRight], skeletonColor, skeletonThickness);

	//LEFT HIP + LEFT KNEE
	if ((jointsPoints[JointType_HipLeft].x >= 0) && (jointsPoints[JointType_KneeLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_HipLeft], jointsPoints[JointType_KneeLeft], skeletonColor, skeletonThickness);

	//RIGHT KNEE + RIGHT ANKLE
	if ((jointsPoints[JointType_KneeRight].x >= 0) && (jointsPoints[JointType_AnkleRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_KneeRight], jointsPoints[JointType_AnkleRight], skeletonColor, skeletonThickness);

	//LEFT KNEE + LEFT ANKLE
	if ((jointsPoints[JointType_KneeLeft].x >= 0) && (jointsPoints[JointType_AnkleLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_KneeLeft], jointsPoints[JointType_AnkleLeft], skeletonColor, skeletonThickness);

	//RIGHT ANKLE + RIGHT FOOT
	if ((jointsPoints[JointType_AnkleRight].x >= 0) && (jointsPoints[JointType_FootRight].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_AnkleRight], jointsPoints[JointType_FootRight], skeletonColor, skeletonThickness);

	//LEFT ANKLE + LEFT FOOT
	if ((jointsPoints[JointType_AnkleLeft].x >= 0) && (jointsPoints[JointType_FootLeft].x >= 0))
		line(tmpSkeletonMat, jointsPoints[JointType_AnkleLeft], jointsPoints[JointType_FootLeft], skeletonColor, skeletonThickness);

	return tmpSkeletonMat;

}


Mat GetBodyIndexFrame(IMultiSourceFrame *multiSourceFrame) {


	IBodyIndexFrame *bodyIndexFrame = nullptr;
	IBodyIndexFrameReference *bodyIndexFrameReference = nullptr;
	IFrameDescription *bodyIndexFrameDescription = nullptr;
	BYTE *bodyIndexFrameBuffer = nullptr;
	UINT bodyIndexFrameBufferSize = 0;

	Mat resultMat = Mat::zeros(0, 0, CV_16UC1);


	if (SUCCEEDED(multiSourceFrame->get_BodyIndexFrameReference(&bodyIndexFrameReference))) {

		if (SUCCEEDED(bodyIndexFrameReference->AcquireFrame(&bodyIndexFrame))) {


			if (SUCCEEDED(bodyIndexFrame->get_FrameDescription(&bodyIndexFrameDescription))) {
				if (SUCCEEDED(bodyIndexFrameDescription->get_Height(&bodyIndexFrameHeight)) &&
					SUCCEEDED(bodyIndexFrameDescription->get_Width(&bodyIndexFrameWidth))) {


					if (SUCCEEDED(bodyIndexFrame->AccessUnderlyingBuffer(&bodyIndexFrameBufferSize, &bodyIndexFrameBuffer))) {

						RGBQUAD * bodyIndexFrameRGBXPointer = bodyIndexFrameRGBXMat;

						for (UINT i = 0; i < bodyIndexFrameBufferSize; i++) {
							bodyIndexFrameRGBXPointer->rgbRed = (*bodyIndexFrameBuffer) & 0x01 ? 0x00 : 0xFF;
							bodyIndexFrameRGBXPointer->rgbGreen = (*bodyIndexFrameBuffer) & 0x02 ? 0x00 : 0xFF;
							bodyIndexFrameRGBXPointer->rgbBlue = (*bodyIndexFrameBuffer) & 0x04 ? 0x00 : 0xFF;
							bodyIndexFrameRGBXPointer->rgbReserved = 0XFF;
							bodyIndexFrameRGBXPointer++;
							bodyIndexFrameBuffer++;
						}
						Mat bodyIndexFrameMat(bodyIndexFrameHeight, bodyIndexFrameWidth, CV_8UC4, bodyIndexFrameRGBXMat);

						resultMat = bodyIndexFrameMat;

					}
				}
				if (bodyIndexFrameDescription) bodyIndexFrameDescription->Release();
			}
			if (bodyIndexFrame) bodyIndexFrame->Release();
		}
		if (bodyIndexFrameReference) bodyIndexFrameReference->Release();
	}
	return resultMat;

}


Mat GetDepthFrame(IMultiSourceFrame *multiSourceFrame) {

	IDepthFrame *depthFrame = nullptr;
	IDepthFrameReference *depthFrameReference = nullptr;
	IFrameDescription *depthFrameDescription = nullptr;
	Mat resultMat = Mat::zeros(0, 0, CV_16UC1);

	if (SUCCEEDED(multiSourceFrame->get_DepthFrameReference(&depthFrameReference))) {
		if (SUCCEEDED(depthFrameReference->AcquireFrame(&depthFrame))) {

			UINT16 *depthFrameBuffer = NULL;
			UINT depthFrameBufferSize;

			if (SUCCEEDED(depthFrame->get_FrameDescription(&depthFrameDescription))) {
				USHORT depthFrameMaxReliableDistance;
				USHORT depthFrameMinReliableDistance;

				depthFrame->get_DepthMaxReliableDistance(&depthFrameMaxReliableDistance);
				depthFrame->get_DepthMinReliableDistance(&depthFrameMinReliableDistance);

				if (SUCCEEDED(depthFrameDescription->get_Height(&depthFrameHeight)) &&
					SUCCEEDED(depthFrameDescription->get_Width(&depthFrameWidth))) {

					// IF FRAME SIZE SETTED, GET THE DATA
					if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(&depthFrameBufferSize, &depthFrameBuffer))) {


						RGBQUAD* depthFrameRGBXPointer = depthFrameRGBXMat;
						const UINT16* depthFrameBufferEnd = depthFrameBuffer + (depthFrameWidth * depthFrameHeight);

						while (depthFrameBuffer < depthFrameBufferEnd) {
							USHORT depth = *depthFrameBuffer;

							if (depth < 0) {
								depthFrameRGBXPointer->rgbRed = 0xFF;
								depthFrameRGBXPointer->rgbGreen = 0;
								depthFrameRGBXPointer->rgbBlue = 0;
								depthFrameRGBXPointer->rgbReserved = 0xFF;
							}
							else if (depth < depthFrameMinReliableDistance) {
								depthFrameRGBXPointer->rgbRed = 0;
								depthFrameRGBXPointer->rgbGreen = depth & 0x7F + 0x80;
								depthFrameRGBXPointer->rgbBlue = 0;
								depthFrameRGBXPointer->rgbReserved = 0xFF;
							}
							else if (depth < depthFrameMaxReliableDistance) {
								depthFrameRGBXPointer->rgbRed = depth & 0xFF;
								depthFrameRGBXPointer->rgbGreen = depthFrameRGBXPointer->rgbRed;
								depthFrameRGBXPointer->rgbBlue = depthFrameRGBXPointer->rgbRed;
								depthFrameRGBXPointer->rgbReserved = 0xFF;
							}
							else {
								depthFrameRGBXPointer->rgbRed = 0;
								depthFrameRGBXPointer->rgbGreen = 0;
								depthFrameRGBXPointer->rgbBlue = depth & 0x7F + 0x80;
								depthFrameRGBXPointer->rgbReserved = 0xFF;
							}
							++depthFrameRGBXPointer;
							++depthFrameBuffer;
						}

						Mat depthFrameMat(depthFrameHeight, depthFrameWidth, CV_8UC4, depthFrameRGBXMat);
						resultMat = depthFrameMat;
					}
				}

				if (depthFrameDescription) depthFrameDescription->Release();
			}

			if (depthFrame) depthFrame->Release();

		}

		if (depthFrameReference) depthFrameReference->Release();
	}

	return resultMat;
}

Mat GetColorFrame(IMultiSourceFrame *multiSourceFrame) {

	IColorFrame *colorFrame = nullptr;
	IColorFrameReference *colorFrameReference = nullptr;
	IFrameDescription *colorFrameDescription = nullptr;
	Mat resultMat = Mat::zeros(0, 0, CV_16UC1);

	if (SUCCEEDED(multiSourceFrame->get_ColorFrameReference(&colorFrameReference))) {
		if (SUCCEEDED(colorFrameReference->AcquireFrame(&colorFrame))) {

			if (SUCCEEDED(colorFrame->get_FrameDescription(&colorFrameDescription))) {

				if (SUCCEEDED(colorFrameDescription->get_Height(&colorFrameHeight)) &&
					SUCCEEDED(colorFrameDescription->get_Width(&colorFrameWidth))) {


					if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(colorFrameHeight * colorFrameWidth * CHANNEL,
						colorFrameBuffer,
						ColorImageFormat::ColorImageFormat_Bgra))) {
						Mat colorFrameMat = Mat(colorFrameHeight, colorFrameWidth, CV_8UC4, colorFrameBuffer);

						resultMat = Mat::zeros(desiredColorFrameHeight, desiredColorFrameWidth, CV_8UC4);

						resize(colorFrameMat, resultMat, resultMat.size(), 0, 0, INTER_LINEAR);

					}

				}

				if (colorFrameDescription) colorFrameDescription->Release();
			}

			if (colorFrame) colorFrame->Release();
		}

		if (colorFrameReference) colorFrameReference->Release();
	}

	return resultMat;
}



int main() {

	int skeletonMapingMode = 2;
	bool initializingStatus = InitializingKinect();
	if (initializingStatus) {

		string sampleNum;
		cout << "put sample number here (please include all the zeros):" << endl;
		cin >> sampleNum;
		string path = DESTINATIONPATH + "\\Sample" + sampleNum;
		_mkdir(path.c_str());
		string colorPath = path + "\\Sample" + sampleNum + "_color.avi";
		string depthPath = path + "\\Sample" + sampleNum + "_depth.avi";
		string dataPath = path + "\\Sample" + sampleNum + "_data.csv";
		string skeletonPath = path + "\\Sample" + sampleNum + "_skeleton.csv";
		string userPath = path + "\\Sample" + sampleNum + "_user.avi";
		string skeletonFPath = path + "\\Sample" + sampleNum + "_skeleton.avi";

		Mat depthFrameMat = Mat::zeros(0, 0, CV_16UC1);
		Mat colorFrameMat = Mat::zeros(0, 0, CV_16UC1);
		Mat maskFrameMat = Mat::zeros(0, 0, CV_16UC1);
		BodyInfoStructs bodyInfo;
		bodyInfo.bodyInfoMat = Mat::zeros(0, 0, CV_16UC1);

		while (!(!depthFrameMat.empty() && !maskFrameMat.empty() && !colorFrameMat.empty() && !bodyInfo.bodyInfoMat.empty())) {

			//check if bodyinfo is empty
			if (SUCCEEDED(multiSourceFrameReader->AcquireLatestFrame(&multiSourceFrame))) {
				depthFrameMat = GetDepthFrame(multiSourceFrame);
				maskFrameMat = GetBodyIndexFrame(multiSourceFrame);
				colorFrameMat = GetColorFrame(multiSourceFrame);
				bodyInfo = GetBodyFrame(multiSourceFrame, skeletonMapingMode);
			}
			if (multiSourceFrame) {
				multiSourceFrame->Release();
			}
		}



		Size colorFrameSize(desiredColorFrameWidth, desiredColorFrameHeight);
		Size depthFrameSize(depthFrameWidth, depthFrameHeight);
		Size maskFrameSize(bodyIndexFrameWidth, bodyIndexFrameHeight);
		Size skeletonFrameSize;

		if (skeletonMapingMode == 1)
			skeletonFrameSize = colorFrameSize;
		if (skeletonMapingMode == 2)
			skeletonFrameSize = depthFrameSize;


		//int fourcc = CV_FOURCC('X','V','I','D');
		int fourcc = -1;
		VideoWriter oVideoWriter(colorPath.c_str(), fourcc, 20, colorFrameSize, true);
		VideoWriter dVideoWriter(depthPath.c_str(), fourcc, 20, depthFrameSize, true);
		VideoWriter uVideoWriter(userPath.c_str(), fourcc, 20, maskFrameSize, true);
		VideoWriter sVideoWriter(skeletonFPath.c_str(), fourcc, 20, skeletonFrameSize, true);


		ofstream csvFile(skeletonPath.c_str());

		namedWindow("depth image", CV_WINDOW_AUTOSIZE);
		namedWindow("color image", CV_WINDOW_AUTOSIZE);
		namedWindow("mask image", CV_WINDOW_AUTOSIZE);
		namedWindow("skeleton image", CV_WINDOW_AUTOSIZE);

		bool r = false;

		if (!oVideoWriter.isOpened() || !dVideoWriter.isOpened() || !uVideoWriter.isOpened() || !sVideoWriter.isOpened())
		{
			cout << "!!! Output video could not be opened" << std::endl;
			return -1;
		}
		else cout << "video opened successfully" << endl << "use r-key to record" << endl;

		while (1) {
			if (SUCCEEDED(multiSourceFrameReader->AcquireLatestFrame(&multiSourceFrame))) {

				depthFrameMat = GetDepthFrame(multiSourceFrame);
				maskFrameMat = GetBodyIndexFrame(multiSourceFrame);
				colorFrameMat = GetColorFrame(multiSourceFrame);
				bodyInfo = GetBodyFrame(multiSourceFrame, skeletonMapingMode);

				if (!depthFrameMat.empty() && !maskFrameMat.empty() && !colorFrameMat.empty() && !bodyInfo.bodyInfoMat.empty()) {
					if (r == true) {
						oVideoWriter.write(colorFrameMat);
						dVideoWriter.write(depthFrameMat);
						uVideoWriter.write(maskFrameMat);
						sVideoWriter.write(bodyInfo.bodyInfoMat);

						for (int i = 0; i < JointType_Count - 1; i++) {
							csvFile
								<< to_string(bodyInfo.skeletonInfo[9 * i + 0]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 1]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 2]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 3]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 4]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 5]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 6]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 7]) << ","
								<< to_string(bodyInfo.skeletonInfo[9 * i + 8]) << ",";
						}
						csvFile
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 0]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 1]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 2]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 3]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 4]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 5]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 6]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 7]) << ","
							<< to_string(bodyInfo.skeletonInfo[9 * (JointType_Count - 1) + 8]) << endl;
					}

					int c = cvWaitKey(1);
					if (c == 27 || c == 'q' || c == 'Q')
						break;
					if (c == 'r' || c == 'R') {
						r = true;
						cout << "recoridng" << endl << "use s-key to stop" << endl;
					}
					if (c == 's' || c == 'S') {
						r = false;
						cout << "stoped" << endl;
					}

					//depthFrameMat = depthFrameMat + bodyInfo.bodyInfoMat;
					imshow("depth image", depthFrameMat);
					imshow("mask image", maskFrameMat);
					imshow("color image", colorFrameMat);
					imshow("skeleton image", bodyInfo.bodyInfoMat);
					c = waitKey(1);
				}
			}
			if (multiSourceFrame) {
				multiSourceFrame->Release();
			}
		}
		oVideoWriter.release();
		dVideoWriter.release();
		uVideoWriter.release();
		sVideoWriter.release();
		/*cvReleaseImageHeader(&depth);
		cvReleaseImageHeader(&color);
		cvReleaseImage(&skeleton);*/
		cvDestroyWindow("depth image");
		cvDestroyWindow("color image");
		cvDestroyWindow("mask image");
		cvDestroyWindow("skeleton image");


		//NuiShutdown();
		string colorPathNew = path + "\\Sample" + sampleNum + "_color.mp4";
		string depthPathNew = path + "\\Sample" + sampleNum + "_depth.mp4";
		string userPathNew = path + "\\Sample" + sampleNum + "_user.mp4";
		string skeletonFPathNew = path + "\\Sample" + sampleNum + "_skeleton.mp4";
		rename(colorPath.c_str(), colorPathNew.c_str());
		rename(depthPath.c_str(), depthPathNew.c_str());
		rename(userPath.c_str(), userPathNew.c_str());
		rename(skeletonFPath.c_str(), skeletonFPathNew.c_str());
	}
}