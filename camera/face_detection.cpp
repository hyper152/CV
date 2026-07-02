#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 创建一个简单的图像处理示例
    cv::Mat image = cv::Mat::zeros(400, 600, CV_8UC3);
    
    // 绘制一个笑脸 
    cv::circle(image, cv::Point(300, 200), 100, cv::Scalar(0, 255, 255), -1); // 脸
    cv::circle(image, cv::Point(270, 170), 10, cv::Scalar(0, 0, 0), -1);     // 左眼
    cv::circle(image, cv::Point(330, 170), 10, cv::Scalar(0, 0, 0), -1);     // 右眼
    cv::ellipse(image, cv::Point(300, 230), cv::Size(40, 20), 0, 0, 180, cv::Scalar(0, 0, 0), 3); // 嘴
    
    cv::putText(image, "New OpenCV Project!", 
                cv::Point(150, 350), 
                cv::FONT_HERSHEY_COMPLEX, 1.0, 
                cv::Scalar(255, 255, 255), 2);
    
    cv::imshow("Face Detection", image);
    std::cout << "新项目运行成功！" << std::endl;
    cv::waitKey(0);
    return 0;
}