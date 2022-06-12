#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


void getContours(Mat imgDil, Mat img);
int differentCars(Rect);
Mat recognizeShapes();

string path = "shapes.png";
Mat img = imread(path);
Mat img2 = imread(path);
int cannyMin = 25, cannyMax = 100, karnelSize = 10;
int counter = 0;

void main() {
	path = "shapes.png";
	img = imread(path);
	Mat firstImg = recognizeShapes();


	path = "Capture.JPG";
	img = imread(path);
	Mat secondImg = recognizeShapes();

	imshow("first Image", firstImg);
	imshow("second Image", secondImg);

	waitKey(0);
}

Mat recognizeShapes() {
	Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

	cvtColor(img, imgGray, COLOR_BGR2GRAY);

	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);

	Canny(imgBlur, imgCanny, cannyMin, cannyMax);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(karnelSize, karnelSize));
	dilate(imgCanny, imgDil, kernel);

	getContours(imgDil, img);

	return img;
}

string carOrCircle(InputArray contour, OutputArray conPolyOut, double epsilon) {
	vector<Point> conPoly;
	approxPolyDP(contour, conPoly, epsilon, true);
	Rect boundRect = boundingRect(conPoly);

	int	objCor = (int)conPoly.size();



	if (objCor == 8) {
		return "Circle";
	}
	int carClassNumber = differentCars(boundRect);
	if (carClassNumber == -1) return "Unknown Class Number";

	return "Car Class" + to_string(carClassNumber);
}

void getContours(Mat imgDil, Mat imgOut) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());
	int area, objCor;
	string objectType;
	float peri;
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);

		if (area > 500)
		{
			objectType = "";
			peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.043 * peri, true);
			boundRect[i] = boundingRect(conPoly[i]);

			objCor = (int)conPoly[i].size();

			if (objCor == 3) {
				objectType = "triangle";
			}
			else if (objCor == 4) {
				float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
				if (aspRatio > 0.95 && aspRatio < 1.05) objectType = "Square";
				else									objectType = "rectangle";
			}
			else {
				objectType = carOrCircle(contours[i], conPoly[i], 0.018 * peri);
			}



			rectangle(imgOut, boundRect[i].tl(), boundRect[i].br(), Scalar(21, 21, 112), 2);

			putText(imgOut, objectType, { boundRect[i].x + boundRect[i].width / 3 ,boundRect[i].y - 4 }, FONT_HERSHEY_PLAIN, 1, Scalar(21, 21, 112), 2);
		}
	}
}


int differentCars(Rect boundRect) {
	Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

	float width = boundRect.width, height = boundRect.height;
	float x = boundRect.x, y = boundRect.y;

	Point2f src[4] = { {x,y} , {x + width, y}, {x,y + height},{x + width, y + height} };
	Point2f dst[4] = { {0.0f,0.0f},{width,0.0f},{0.0f,height},{width,height} };

	Mat imgWarp, matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(width, height));



	cvtColor(imgWarp, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, cannyMin, cannyMax);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(10, 1));
	dilate(imgCanny, imgDil, kernel);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundaryRect(contours.size());

	int area, objCor;
	string objectType;
	float peri;


	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		if (area > 500)
		{
			objectType = "";

			peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.035 * peri, true);
			boundaryRect[i] = boundingRect(conPoly[i]);

			objCor = (int)conPoly[i].size();

			if (objCor == 3) {
				objectType = "triangle";
			}
			else if (objCor == 4) {
				float aspRatio = (float)boundaryRect[i].width / (float)boundaryRect[i].height;
				if (aspRatio > 0.95 && aspRatio < 1.05) objectType = "Square";
				else									objectType = "rectangle";
			}

			cout << "contour #" << i << " ==> " << boundaryRect[i].width << endl;

			putText(imgWarp, to_string(boundaryRect[i].width), { boundaryRect[i].x + boundaryRect[i].width / 3 ,boundaryRect[i].y - 4 }, FONT_HERSHEY_PLAIN, 1, Scalar(21, 21, 112), 2);
		}
	}

	if (boundaryRect.size() < 4) return -1;

	float par1 = boundaryRect[2].width;
	float par2 = boundaryRect[3].width;
	float diffPars = par1 / par2;

	//imshow("imgWarp #" + to_string(counter++), imgWarp);
	//waitKey(1);

	if (diffPars >= 0.95 && diffPars <= 1.05) return 3;
	else if (diffPars > 1.5)			      return 2;
	else									  return 1;

}

