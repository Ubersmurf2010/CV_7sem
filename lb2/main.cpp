#include <iostream>
#include <opencv2/opencv.hpp>


// Timers's class
class CV_EXPORTS TickMeter
{
public:
  TickMeter();
  void start();
  void stop();

  int64 getTimeTicks() const;
  double getTimeMicro() const;
  double getTimeMilli() const;
  double getTimeSec()   const;
  int64 getCounter() const;

  void reset();
private:
  int64 counter;
  int64 sumTime;
  int64 startTime;
};

std::ostream& operator << (std::ostream& out, const TickMeter& tm);

TickMeter::TickMeter() { reset(); }
int64 TickMeter::getTimeTicks() const { return sumTime; }
double TickMeter::getTimeMicro() const { return  getTimeMilli() * 1e3; }
double TickMeter::getTimeMilli() const { return getTimeSec() * 1e3; }
double TickMeter::getTimeSec() const { return (double)getTimeTicks() / cv::getTickFrequency(); }
int64 TickMeter::getCounter() const { return counter; }
void TickMeter::reset() { startTime = 0; sumTime = 0; counter = 0; }

void TickMeter::start() { startTime = cv::getTickCount(); }
void TickMeter::stop()
{
  int64 time = cv::getTickCount();
  if (startTime == 0)
    return;
  ++counter;
  sumTime += (time - startTime);
  startTime = 0;
}

std::ostream& operator << (std::ostream& out, const TickMeter& tm) { return out << tm.getTimeSec() << "sec"; }


void box_filter(cv::Mat& src, cv::Mat& dst, int kernel_size) {
    assert(kernel_size % 2 == 1);
    dst.create(src.size(), src.type());
    unsigned int sum = 0;

    for (int col = (kernel_size-1) / 2; col < dst.cols - (kernel_size-1) / 2; col++) {
        for (int row = (kernel_size-1) / 2; row < dst.rows - (kernel_size-1) / 2; row++) {
            sum = 0;
            for (int temp_col = col - (kernel_size-1) / 2; temp_col <= col + (kernel_size-1) / 2; temp_col++) {
                for (int temp_row = row - (kernel_size-1) / 2; temp_row <= row + (kernel_size-1) / 2; temp_row++) {
                    sum += (int)src.at<uchar>(temp_row, temp_col);
                }
            }
            dst.at<uchar>(row, col) = std::round(sum / (kernel_size * kernel_size));
        }
    }
}

void compare(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, double& interest) {
    result = img1.clone();

    double count = 0;

    for (int col = 0; col < img1.cols; col++) {
        for (int row = 0; row < img1.rows; row++) {
            if (img1.at<uchar>(row, col) == img2.at<uchar>(row, col) ||
            img1.at<uchar>(row, col) == img2.at<uchar>(row, col) + 1 ||
            img1.at<uchar>(row, col) == img2.at<uchar>(row, col) - 1)
             {
                result.at<uchar>(row, col) = 0;
                count++;
            }
        }
    }
    interest = count / (img1.rows * img1.cols);
    return;
}


void unsharpMasking(cv::Mat& src, cv::Mat& sharping, cv::Mat& dst, double k = 0.5) {
    dst = src.clone();
    for (int col = 0; col < src.cols; col++)
        for (int row = 0; row < src.rows; row++) {
            dst.at<uchar>(row, col) = src.at<uchar>(row, col) + k * (src.at<uchar>(row, col) - sharping.at<uchar>(row, col));
        }
}

void laplasFiltration(cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    for (int col = 1; col < src.cols - 1; col++)
        for(int row = 1; row < src.rows - 1; row++) {
            dst.at<uchar>(row, col) = std::round((src.at<uchar>(row, col) * (4) + 
            (-1) * src.at<uchar>(row + 1, col) + 
            (-1) * src.at<uchar>(row - 1, col) + 
            (-1) * src.at<uchar>(row, col + 1) + 
            (-1) * src.at<uchar>(row, col - 1)) / 9); 
        }
}

void logTransform(cv::Mat& img1, cv::Mat& img2, cv::Mat& result, double c = 0.5) {
    result = img1.clone();
    for (int col = 0; col < img1.cols; col++)
        for (int row = 0; row < img2.rows; row++) {
            result.at<uchar>(row, col) = c * std::log(1 + std::abs(img1.at<uchar>(row, col) - img2.at<uchar>(row, col)));
    }
}

int main(int argc, char** argv) {
    // set timer
    TickMeter timer;

    cv::Mat img = cv::imread("/home/gerzeg/Polytech/TV/lb2/images/plank.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return -1;
    }

    int startX = 375;
    int startY = 0;
    int width = 400;
    int height = 400;
    cv::Rect rp(startX, startY, width, height);
    cv::Mat roi = img(rp);

    cv::imshow("OI", roi);
    cv::waitKey(-1);

    //box_filter
    cv::Mat box_filtered;
    timer.start();
    box_filter(roi, box_filtered, 3);
    timer.stop();
    std::cout << "My time: " << timer.getTimeSec() << std::endl;
    timer.reset();
    cv::imshow("box_filtered", box_filtered);
    cv::waitKey(-1);

    //opencv box_filter
    cv::Mat opencvFiltered;
    timer.start();
    blur(roi, opencvFiltered, cv::Size(3, 3));
    timer.stop();
    std::cout << "Blur time: " << timer.getTimeSec() << std::endl;
    timer.reset();
    cv::imshow("opencvFiltered", opencvFiltered);
    cv::waitKey(-1);

    //Gauss filter
    cv::Mat GaussFiltered;
    GaussianBlur(roi, GaussFiltered, cv::Size(3, 3), (0, 0));
    cv::imshow("Gauss", GaussFiltered);
    cv::waitKey(-1);

    //compare images 1 own box filter and opencv box filter
    cv::Mat comparsion_1;
    double interest;
    compare(box_filtered, opencvFiltered, comparsion_1, interest);
    std::cout << interest <<"" << std::endl;
    cv::imshow("comparison BOX 1", comparsion_1);
    cv::waitKey(-1);

    //compare images 2 opencv box filter and gauss filter
    cv::Mat comparsion_2;
    logTransform(GaussFiltered, opencvFiltered, comparsion_2);
    cv::imshow("comparison BOX 2", comparsion_2);
    cv::waitKey(-1);

    //unsharp masking
    cv::Mat boxSharp;
    cv::Mat gaussSharp;
    unsharpMasking(roi, opencvFiltered, boxSharp);
    unsharpMasking(roi, GaussFiltered, gaussSharp);
    cv::imshow("box sharping", boxSharp);
    cv::imshow("gauss sharping", gaussSharp);
    cv::waitKey(-1);

    // comparsion log between box and gauss
    cv::Mat comparsion_3;
    logTransform(boxSharp, gaussSharp, comparsion_3);
    cv::imshow("comparison BOX 3", comparsion_3);
    cv::waitKey(-1);

    // laplas filter
    cv::Mat laplas;
    laplasFiltration(roi, laplas);
    cv::imshow("laplasFiltration", laplas);
    cv::waitKey(-1);

    // unsharp for laplas
    cv::Mat laplasSharp;
    unsharpMasking(roi, laplas, laplasSharp);
    cv::imshow("laplasSharp", laplasSharp);
    cv::waitKey(-1);

    // comparsion log between unsharp laplas and box
    cv::Mat comparsion_4;
    logTransform(boxSharp, laplasSharp, comparsion_4);
    cv::imshow("comparison BOX 4", comparsion_4);
    cv::waitKey(-1);
    return 0;
}