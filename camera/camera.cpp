#include <opencv2/opencv.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
int main()
{
    VideoCapture cap(1);
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera" << endl;
        return -1;
    }
 
    Mat img; 
    namedWindow("img", WINDOW_NORMAL);
 
    while (1)
    {
        cap >> img;
        if (img.empty())
            break;
 
        imshow("img", img);
 
        char key = waitKey(20);
        if (key == 27 || key == 'q') // 按 ESC 或 'q' 退出
            break;
    }
 
    cap.release();
    destroyAllWindows();
 
    return 0;
}
