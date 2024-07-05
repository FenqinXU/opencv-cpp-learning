// Qt_window.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
/*右键单击项目，选择“属性”。
在属性窗口中，选择“VC++ 目录” -> “包含目录”，然后添加 D:\Software\OpenCV\build\include 路径，以包含 OpenCV 的头文件。
选择“链接器” -> “常规” -> “附加库目录”，然后添加 D:\Software\OpenCV\build\x64\vc17\lib 路径，以包含 OpenCV 的库文件。
选择“链接器” -> “输入” -> “附加依赖项”，然后添加 opencv_core.lib。添加库文件路径：

在“附加库目录”字段中，添加OpenCV库文件的路径。这些库文件通常位于OpenCV安装目录的 lib 文件夹中。
点击字段右侧的下拉箭头，然后选择“编辑”。
在弹出的窗口中，添加OpenCV库文件的路径，然后点击“确定”。*/
/*opencv_world490d.lib
opencv_img_hash490d.lib,不能同时带xxd.lib和xx.lib,需要单独添加根据Debug和Release模式不同配置不同*/

#include<iostream>
#include<string>
#include<sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/opencv_modules.hpp>
#include<opencv2/imgproc.hpp>
#include<math.h>
#include <memory>
#include <filesystem>
//#include "resource.h"
#include "MultipleImageWindow.h"
using namespace cv;
using namespace std;
shared_ptr<MultipleImageWindow> miw;
#if _DEBUG
#pragma comment(lib,"opencv_world454d.lib")
#else
#pragma comment(lib,"opencv_world454.lib")
#endif // _DEBUG




/*
int main() {
    // 图片文件路径
    string imagePath = "D:\\Users\\xfq\\source\\repos\\Qt_window\\2.jpg";
    cout << "Image file path: " << imagePath << endl;

    // 读取图片文件
    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Failed to read image!" << endl;
        return -1;
    }

    // 显示图像
    imshow("Image", img);
    waitKey(0);

    return 0;
}*/
/*
int main() {
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
		return -1;

	namedWindow("video", 1);
	for (;;)
	{
		Mat frame;//视频帧
		cap >> frame;

		imshow("video", frame);
		if (waitKey(30) >= 0) break;
	}
	cap.release();
	return 0;
}*/
/*

int main() {
    // 图片文件路径
    string imagePath = "D:\\Users\\xfq\\source\\repos\\Qt_window\\2.jpg";
    //cout << "Image file path: " << imagePath << endl;

    // 读取图片文件
    Mat img = imread(imagePath, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Failed to read image!" << endl;
        return -1;
    }
    Mat npp_photo = imread("D:\\Users\\xfq\\source\\repos\\Qt_window\\npp.jpg", IMREAD_COLOR);
    if (npp_photo.empty()) {
        cerr << "Failed to read image!" << endl;
        return -1;
    }
    //创建窗口
    namedWindow("test", WINDOW_NORMAL);
    namedWindow("npp", WINDOW_AUTOSIZE);


    //移动窗口
    moveWindow("test", 10, 10);
    moveWindow("npp", 520, 10);


    // 显示图像
    imshow("test", img);
    imshow("npp", npp_photo);

    //重新调整窗口大小
    resizeWindow("test", 512, 512);

    waitKey(0);
    destroyWindow("test");
    destroyWindow("npp");
    
    for (int i = 0; i < 5; i++) {
        ostringstream ss;
        ss << "npp" << i;
        namedWindow(ss.str());
        moveWindow(ss.str(), 20 * i, 20 * i);
        imshow(ss.str(), npp_photo);
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}
*/
/*用户界面创建，用滑块调整模糊度，用圆函数添加圆*/
/*
int blurAmount = 15;
static void onChange(int pos, void* userInput) 
{
    if (pos <= 0)return;
    Mat imgBlur;
    Mat* image = (Mat*)userInput;
    blur(*image, imgBlur, Size(pos, pos));
    imshow("Lena", imgBlur);
}


static void onMouse(int event, int x, int y, int, void* userInput)
{
    if (event != EVENT_LBUTTONDOWN) return;
    Mat* image = (Mat*)userInput;
    circle(*image, Point(x, y), 10, Scalar(0, 255, 0), 3);//创建圆函数，scalar是圆的颜色和圆的大小
    onChange(blurAmount, image);
}

int main(int argc, const char** argv)
{
    Mat img = imread("D:\\Users\\xfq\\source\\repos\\Qt_window\\npp.jpg", IMREAD_COLOR);
    // Create windows
    namedWindow("Lena");

    // create a trackbark
    createTrackbar("Lena", "Lena", &blurAmount, 30, onChange, &img);

    setMouseCallback("Lena", onMouse, &img);

    // Call to onChange to init
    onChange(blurAmount, &img);

    // wait app for a key to exit
    waitKey(0);

    // Destroy the windows
    destroyWindow("Lena");

    return 0;
    return 0;
}
*/



/*
Mat img;
bool applyGray = false;
bool applyBlur = false;
bool applySobel = false;

void applyFilters()
{
    Mat result;

    img.copyTo(result);
    if (applyGray) {
        cvtColor(result, result, COLOR_BGR2GRAY);

    }
    if (applyBlur) {
        blur(result, result, Size(5, 5));

    }
    if (applySobel) {
        Sobel(result, result, CV_8U, 1, 1);
    }
    imshow("npp", result);

}



void grayCallback(int state, void* userData) 
{
    applyGray = true;
    applyFilters();

}
void bgrCallback(int state, void* userData) {
    applyGray = false;
    applyFilters();
}
void blurCallback(int state, void* userData) {
    applyBlur = (bool)state;
    applyFilters();
}

void sobelCallback(int state, void* userData) {
    applySobel = !applySobel;
    applyFilters();
}



int main(int argc,const char** argv) 
{
    img= imread("D:\\Users\\xfq\\source\\repos\\Qt_window\\npp.jpg", IMREAD_COLOR);
    namedWindow("npp");
    createButton("Blur", blurCallback, NULL, QT_CHECKBOX, 0); 
    createButton("Gray", grayCallback, NULL,QT_RADIOBOX, 0);
    createButton("RGB", bgrCallback, NULL, QT_RADIOBOX, 1 );
    createButton("Sobel", sobelCallback, NULL, QT_PUSH_BUTTON, 0);
    waitKey(0);
    destroyWindow("npp");
    return 0;
}
*/

/*
int main()
{
    string imageFilename = "D:\\Users\\xfq\\source\\repos\\Qt_window\\npp.jpg"; // 替换为实际的图像文件路径

    Mat img = imread(imageFilename, IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Failed to load image: " << imageFilename << endl;
        return -1;
    }

    //namedWindow("npp");

    // 绘制直方图
    vector<Mat> bgr;
    split(img, bgr);

    int numbins = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange);
    calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins, &histRange);
    calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins, &histRange);

    int width = 512;
    int height = 300;
    Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));

    normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, height, NORM_MINMAX);

    int binStep = cvRound((float)width / (float)numbins);
    for (int i = 1; i < numbins; i++)
    {
        line(histImage,
            Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
            Point(binStep * (i), height - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0)
        );
        line(histImage,
            Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
            Point(binStep * (i), height - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0)
        );
        line(histImage,
            Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
            Point(binStep * (i), height - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255)
        );
    }

    // 显示直方图
    imshow("Histogram", histImage);

    waitKey(0);
    //destroyWindow("npp");
    return 0;
}
*/

// OpenCV command line parser functions
// Keys accecpted by command line parser
/*
const char* keys =
{
    "{help h usage ? | | print this message}"
    "{@video | | Video file, if not defined try to use webcamera}"
};

int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Chapter 2. v1.0.0");
    //If requires help show
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String videoFile = parser.get<String>(0);

    // Check if params are correctly parsed in his variables
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    VideoCapture cap; // open the default camera
    if (videoFile != "")
        cap.open(videoFile);
    else
        cap.open(0);
    if (!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("Video", 1);
    for (;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        if (frame.empty())
            return 0;
        imshow("Video", frame);
        if (waitKey(30) >= 0) break;
    }
    // Release the camera or video cap
    cap.release();

    return 0;
}
*/
//读取kunkun视频并边缘检测
/*
int main(int argc, const char** argv)
{
    
    String videoFile = "D:\\Users\\xfq\\source\\repos\\Qt_window\\ikun2.mp4";
    Mat eages;
        VideoCapture cap;
        cap.open(videoFile);
        namedWindow("Video", 1);
        for (;;)
        {
            Mat frame;
            cap >> frame; // get a new frame from camera
            if (frame.empty())
                return 0;
             Canny(frame,eages, 20, 40);//eages dectetion
            imshow("Video", eages);
            if (waitKey(30) >= 0) break;
        }
        // Release the camera or video cap
        cap.release();

    return 0;
}*/

static Scalar randomColor(RNG& rng)
{
    auto icolor = (unsigned)rng;
    return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

/**
 * Calcualte image pattern from an input image
 * @param img Mat input image to calculate the light pattern
 * @return a Mat pattern image
 */
Mat calculateLightPattern(Mat img)
{
    Mat pattern;
    // Basic and effective way to calculate the light pattern from one image
    blur(img, pattern, Size(img.cols / 3, img.cols / 3));
    return pattern;
}

/**
 * Calcualte image pattern from an input image
 * @param img Mat input image to calculate the light pattern
 * @return a Mat pattern image
 */
/*
Mat calculateLightPattern(Mat img)
{
    Mat pattern;
    // Basic and effective way to calculate the light pattern from one image
    blur(img, pattern, Size(img.cols / 3, img.cols / 3));
    return pattern;
}
*/

void ConnectedComponents(Mat img)
{
    // Use connected components to divide our possibles parts of images 
    Mat labels;
    auto num_objects = connectedComponents(img, labels);
    // Check the number of objects detected
    if (num_objects < 2) {
        cout << "No objects detected" << endl;
        return;
    }
    else {
        cout << "Number of objects detected: " << num_objects - 1 << endl;
    }
    // Create output image coloring the objects
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    RNG rng(0xFFFFFFFF);
    for (auto i = 1; i < num_objects; i++) {
        Mat mask = labels == i;
        output.setTo(randomColor(rng), mask);
    }
    //imshow("Result", output);
    miw->addImage("Result", output);
}
/*
void ConnectedComponentsStats(Mat img)
{
    // Use connected components with stats
    Mat labels;
    Mat stats;
    Mat centroids;
    auto num_objects = connectedComponentsWithStats(img, labels, stats, centroids);
    // Check the number of objects detected
    if (num_objects < 2) {
        cout << "No objects detected" << endl;
        return;
    }
    else {
        cout << "Number of objects detected: " << num_objects - 1 << endl;
    }
    // Create output image coloring the objects and show area
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    RNG rng(0xFFFFFFFF);
    for (auto i = 1; i < num_objects; i++) {
        cout << "Object " << i << " with pos: " << centroids.at<Point2d>(i) << " with area " << stats.at<int>(i, CC_STAT_AREA) << endl;
        Mat mask = labels == i;
        output.setTo(randomColor(rng), mask);
        // draw text with area
        stringstream ss;
        ss << "area: " << stats.at<int>(i, CC_STAT_AREA);

        putText(output,
            ss.str(),
            centroids.at<Point2d>(i),
            FONT_HERSHEY_SIMPLEX,
            0.4,
            Scalar(255, 255, 255));
    }
    imshow("Result", output);
    miw->addImage("Result", output);
}
*/

void ConnectedComponentsStats(Mat binary_img)
{
    // Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Check the number of objects detected
    if (contours.empty()) {
        cout << "No objects detected" << endl;
        return;
    }
    else {
        cout << "Number of objects detected: " << contours.size() << endl;
    }

    // Create output image
    Mat output = Mat::zeros(binary_img.size(), CV_8UC3);

    // Draw objects and show area
    RNG rng(0xFFFFFFFF);
    for (size_t i = 0; i < contours.size(); i++) {
        // Draw object
        drawContours(output, contours, static_cast<int>(i), randomColor(rng), FILLED);

        // Calculate object area
        double area = contourArea(contours[i]);

        // Calculate centroid
        Moments mu = moments(contours[i]);
        Point2f centroid(mu.m10 / mu.m00, mu.m01 / mu.m00);

        // Print information
        cout << "Object " << i + 1 << " with pos: " << centroid << " with area " << area << endl;

        // Draw text with area
        stringstream ss;
        ss << "area: " << area;
        putText(output, ss.str(), centroid, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
    }

    // Show result
    imshow("Result", output);
    miw->addImage("Result", output);
}



void FindContoursBasic(Mat img)
{
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    // Check the number of objects detected
    if (contours.size() == 0) {
        cout << "No objects detected" << endl;
        return;
    }
    else {
        cout << "Number of objects detected: " << contours.size() << endl;
    }
    RNG rng(0xFFFFFFFF);
    for (auto i = 0; i < contours.size(); i++)
        drawContours(output, contours, i, randomColor(rng));
    //imshow("Result", output);
    miw->addImage("Result", output);
}


/**
 * Remove th light and return new image without light
 * @param img Mat image to remove the light pattern
 * @param pattern Mat image with light pattern
 * @return a new image Mat without light
 */
Mat removeLight(Mat img, Mat pattern, int method)
{
    Mat aux;
    // if method is normalization
    if (method == 1)
    {
        // Require change our image to 32 float for division
        Mat img32, pattern32;
        img.convertTo(img32, CV_32F);
        pattern.convertTo(pattern32, CV_32F);
        // Divide the imabe by the pattern
        aux = 1 - (img32 / pattern32);
        // Convert 8 bits format
        aux.convertTo(aux, CV_8U, 255);
    }
    else {
        aux = pattern - img;
    }
    //equalizeHist( aux, aux );
    return aux;
}
/*
int method_light = 1;
void oneCallback(int, void*) {
    method_light = 1;
}

void twoCallback(int, void*) {
    method_light = 2;
}


int method_seg = 2;
void segCallback_1(int, void*) {
    method_seg = 1;
}

void segCallback_2(int, void*) {
    method_seg = 2;
}


void segCallback_3(int, void*) {
    method_seg = 3;
}
*/

int main(int argc, const char** argv) {
   
    Mat img= imread("D:\\Users\\xfq\\source\\repos\\Qt_window\\noisy_image.jpg", 0);
   
    // Create the Multiple Image Window
    miw = make_shared<MultipleImageWindow>("Main window", 3, 2, WINDOW_AUTOSIZE);

    // Remove noise
    Mat img_noise, img_box_smooth;
    medianBlur(img, img_noise, 3);
    blur(img, img_box_smooth, Size(3, 3));
    // 创建两个按钮
    /*
    createButton("ONE", oneCallback, NULL, QT_PUSH_BUTTON, 1);
    createButton("TWO", twoCallback, NULL, QT_PUSH_BUTTON, 1);
    createButton("SEG1", segCallback_1, NULL, QT_PUSH_BUTTON, 1);
    createButton("SEG2", segCallback_2, NULL, QT_PUSH_BUTTON, 1);
    createButton("SEG3", segCallback_3, NULL, QT_PUSH_BUTTON, 1);
   */
    // Load image to process
    //Mat light_pattern= imread("D:\\Users\\xfq\\source\\repos\\Qt_window\\light.pgm", 0);
    Mat light_pattern;
    if (light_pattern.data == NULL) {
        // Calculate light pattern
        blur(img_noise, light_pattern, Size(img.cols / 3, img.cols / 3));
    }
    medianBlur(light_pattern, light_pattern, 3);
    int method_light =2;
        //Apply the light pattern
        Mat img_no_light;
        img_noise.copyTo(img_no_light);
        if (method_light != 2) {
            img_no_light = removeLight(img_noise, light_pattern, method_light);
        }


        // Binarize image for segment
        Mat img_thr;
        if (method_light != 2) {
            threshold(img_no_light, img_thr, 30, 255, THRESH_BINARY);
            
        }
        else {
            threshold(img_no_light, img_thr, 100, 255, THRESH_BINARY_INV);//140, 255,
            
        }
        //img_thr.convertTo(img_thr, CV_8U);
        //cout << "Image type: " << img_thr.type() << endl;
        // Show images
        miw->addImage("Input", img);
        miw->addImage("Input without noise", img_noise);
        //miw->addImage("Input without noise with box smooth", img_box_smooth);
        miw->addImage("Light Pattern", light_pattern);
        //imshow("Light pattern", light_pattern);
        //imshow("No Light", img_no_light);
        miw->addImage("No Light", img_no_light);
        miw->addImage("Threshold", img_thr);
        int method_seg = 3;
        
        switch (method_seg) {
        case 1:
            ConnectedComponents(img_thr);
            break;
        case 2:
            ConnectedComponentsStats(img_thr);
          
     
     
          break;
        case 3:
            FindContoursBasic(img_thr);
            break;
        }


        miw->render();

       waitKey(0);
    

    return 0;
}