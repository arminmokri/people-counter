//#include <cstdlib>
#include <iostream>
#include <stdio.h>
using namespace std;

#include <opencv/cv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#define CONST_ZERO 0
#define CONST_BLACK 0
#define CONST_WHITE 255

//struct rectangle

void extractForegroundInMat(
        Mat matForeground,
        Mat matBackground,
        Mat matFrame,
        int threshold,
        bool removeBlack
        ) {
    for (int i = 0; i < matForeground.rows; i++) {
        for (int j = 0; j < matForeground.cols; j++) {
            int diff = abs((int) matFrame.at<uchar> (i, j) - (int) matBackground.at<uchar> (i, j));
            if (diff < threshold) {
                matForeground.at<uchar> (i, j) = CONST_WHITE;
            } else if (removeBlack && matFrame.at<uchar> (i, j) >= 0 && matFrame.at<uchar>(i, j) <= 0) {
                matForeground.at<uchar> (i, j) = CONST_WHITE;
            } else if (false && matFrame.at<uchar> (i, j) >= 230 && matFrame.at<uchar>(i, j) < CONST_WHITE) {
                matForeground.at<uchar> (i, j) = CONST_WHITE;
            } else {
                matForeground.at<uchar> (i, j) = matFrame.at<uchar> (i, j);
            }
        }
    }
}

void matMapFuturesToMatrix(Mat mat,
        int segmentationParts,
        int** foregroundMin,
        int** foregroundMax,
        int** foregroundNumberOf255,
        int** foregroundSum) {

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            int p = i / segmentationParts;
            int q = j / segmentationParts;

            // find min in sq
            if (foregroundMin[p][q] > mat.at<uchar> (i, j)) {
                foregroundMin[p][q] = (int) mat.at<uchar> (i, j);
            }

            // find max in sq
            if (foregroundMax[p][q] < mat.at<uchar> (i, j)) {
                foregroundMax[p][q] = (int) mat.at<uchar> (i, j);
            }

            // count number of white pixels
            if (mat.at<uchar> (i, j) == CONST_WHITE) {
                foregroundNumberOf255[p][q]++;
            }

            // sum of pixels
            foregroundSum[p][q] = foregroundSum[p][q] +(int) mat.at<uchar> (i, j);
        }
    }
}

void checkeredMat(Mat mat,
        int segmentationParts
        ) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (i % segmentationParts == 0 || j % segmentationParts == 0) {
                mat.at<uchar> (i, j) = 0;
            }
        }
    }
}

void cleanMat(Mat mat, int** foreground_numberof255, int segmentation_square, int percent) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (i == 0 || j == 0 || i == mat.rows - 1 || j == mat.cols - 1) {
                mat.at<uchar> (i, j) = CONST_WHITE;
            } else if ((foreground_numberof255[i][j]*100) / segmentation_square > percent) {
                mat.at<uchar> (i, j) = CONST_WHITE;
            } //else if (false && MatForegroundAvg.at<uchar> (i, j) < color_range) {
            //    MatForegroundAvg.at<uchar> (i, j) = CONST_WHITE;
            //}
        }
    }
}

void reconstructionMat(Mat mat, bool removeAloneBlack, bool fillAloneWhite) {
    int numberOfSearchNeighbors = 1;
    int numberOfNeighbors = pow((2 * numberOfSearchNeighbors) + 1, 2) - 1;
    int numberOfDifferentNeighbors = (numberOfNeighbors / 2) + 1;
    bool have_change_removeAloneBlack = true;
    bool have_change_fillAloneWhite = true;
    while (have_change_removeAloneBlack || have_change_fillAloneWhite) {
        Mat mat_temp = mat.clone();
        have_change_removeAloneBlack = false;
        have_change_fillAloneWhite = false;
        for (int i = 1; i < mat.rows - 1; i++) {
            for (int j = 1; j < mat.cols - 1; j++) {
                if ((removeAloneBlack && mat_temp.at<uchar> (i, j) != CONST_WHITE) ||
                        (fillAloneWhite && mat_temp.at<uchar> (i, j) == CONST_WHITE)
                        ) {
                    //
                    int y_start = i - numberOfSearchNeighbors;
                    if (y_start < 0) {
                        y_start = 0;
                    }
                    //
                    int y_end = i + numberOfSearchNeighbors;
                    if (y_end > mat.rows - 1) {
                        y_end = mat.rows - 1;
                    }
                    //
                    int x_start = j - numberOfSearchNeighbors;
                    if (x_start < 0) {
                        x_start = 0;
                    }
                    //
                    int x_end = j + numberOfSearchNeighbors;
                    if (x_end > mat.cols - 1) {
                        x_end = mat.cols - 1;
                    }
                    //
                    int numer_of_whites = 0;
                    int number_of_blacks = 0;
                    int sum_of_blacks = 0;
                    //
                    for (int p = y_start; p <= y_end; p++) {
                        for (int q = x_start; q <= x_end; q++) {
                            // skip center pixel
                            if (p == i && q == j) {
                                continue;
                            }
                            // count number of whites
                            if (removeAloneBlack) {
                                if (mat_temp.at<uchar> (p, q) == CONST_WHITE) {
                                    numer_of_whites++;
                                }
                            }
                            // count number of blacks
                            if (removeAloneBlack) {
                                if (mat_temp.at<uchar> (p, q) != CONST_WHITE) {
                                    number_of_blacks++;
                                    sum_of_blacks = sum_of_blacks + mat_temp.at<uchar> (p, q);
                                }
                            }
                        }
                    }
                    //
                    if (removeAloneBlack) {
                        if (mat_temp.at<uchar> (i, j) != CONST_WHITE && numer_of_whites >= numberOfDifferentNeighbors) {
                            mat.at<uchar> (i, j) = CONST_WHITE;
                            have_change_removeAloneBlack = true;
                        }
                    }
                    //
                    if (removeAloneBlack) {
                        if (mat_temp.at<uchar> (i, j) == CONST_WHITE && number_of_blacks >= numberOfDifferentNeighbors) {
                            mat.at<uchar> (i, j) = sum_of_blacks / number_of_blacks;
                            have_change_fillAloneWhite = true;
                        }
                    }
                }
            }
        }
        removeAloneBlack = have_change_removeAloneBlack;
        fillAloneWhite = have_change_fillAloneWhite;
    }
}

Rect getRectObject(Mat mat) {
    int x1 = mat.cols - 1, x2 = 0;
    int y1 = mat.rows - 1, y2 = 0;
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (mat.at<uchar> (i, j) != CONST_WHITE) {
                if (j < x1) {
                    x1 = j;
                }
                if (j > x2) {
                    x2 = j;
                }
                if (i < y1) {
                    y1 = i;
                }
                if (i > y2) {
                    y2 = i;
                }
            }
        }
    }
    Rect rect = Rect(x1, y1, x2 - x1, y2 - y1);
    return rect;
}

void getRandomPoint(Mat mat, Rect rect, vector<Point_<int> >* randomPoints) {
    int nrand = (rect.area() * 10) / 100;
    randomPoints->clear();
    for (int i = 0; i < nrand; i++) {
        Point_<int> point;
        point.x = (rand() % (rect.width + 1)) + rect.x;
        point.y = (rand() % (rect.height + 1)) + rect.y;
        if (mat.at<uchar> (point) != CONST_WHITE) {
            randomPoints->push_back(point);
        }
    }
}

void aaa(Mat mat, Rect rect, vector<Point_<int> >* randomPoints, int neighborNumber) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < randomPoints->size(); j++) {

            //
            int ys = randomPoints->at(j).y - neighborNumber;
            if (ys < rect.y) {
                ys = rect.y;
            }
            //
            int ye = randomPoints->at(j).y + neighborNumber;
            if (ye > rect.y + rect.height) {
                ye = rect.y + rect.height;
            }
            //
            int xs = randomPoints->at(j).x - neighborNumber;
            if (xs < rect.x) {
                xs = rect.x;
            }
            //
            int xe = randomPoints->at(j).x + neighborNumber;
            if (xe > rect.x + rect.width) {
                xe = rect.x + rect.width;
            }

            Point_<int> point;
            point.x = xs;
            point.y = ys;

            for (int p = ys; p <= ye; p++) {
                for (int q = xs; q <= xe; q++) {
                    if (mat.at<uchar> (point) > mat.at<uchar> (p, q)) {
                        point.x = q;
                        point.y = p;
                    } else if (false && mat.at<uchar> (point) == mat.at<uchar> (p, q) && (rand() % 100) > 50) {
                        point.x = q;
                        point.y = p;
                    }

                }
            }
            randomPoints->at(j) = point;
        }
    }
}

void bbb(Mat mat,
        Mat matExtremum,
        vector<Point_<int> >* randomPoints,
        vector<Point_<int> >* extremumPoints,
        int y1Track, int y2Track) {
    extremumPoints->clear();
    for (int i = 0; i < randomPoints->size(); i++) {
        if (matExtremum.at<uchar> (randomPoints->at(i)) == 0) {
            if (randomPoints->at(i).y > y1Track && randomPoints->at(i).y < y2Track) {
                matExtremum.at<uchar> (randomPoints->at(i)) = 255;
                extremumPoints->push_back(randomPoints->at(i));
            }
        }
    }
}

void ccc(Mat mat, Mat matExtremum, vector<Point_<int> >* ExtremumPoints, int threshold) {
    for (int i = 0; i < ExtremumPoints->size(); i++) {

        vector<Point_<int> > temp;
        temp.push_back(ExtremumPoints->at(i));

        int x1temp = mat.cols - 1, x2temp = 0;
        int y1temp = ExtremumPoints->at(i).y, y2temp = 0;

        while (!temp.empty()) {
            Point_<int> point = temp[temp.size() - 1];
            temp.erase(temp.end() - 1);

            int ys = point.y - 1;
            if (ys < 0) {
                ys = 0;
            }
            //
            int ye = point.y + 1;
            if (ye > mat.rows - 1) {
                ye = mat.rows - 1;
            }
            //
            int xs = point.x - 1;
            if (xs < 0) {
                xs = 0;
            }
            //
            int xe = point.x + 1;
            if (xe > mat.cols - 1) {
                xe = mat.cols - 1;
            }

            for (int p = ys; p <= ye; p++) {
                for (int q = xs; q <= xe; q++) {
                    if (p == point.y && q == point.x) {
                        continue;
                    }
                    if (mat.at<uchar> (p, q) <= mat.at<uchar>(ExtremumPoints->at(i)) + threshold
                            && matExtremum.at<uchar> (p, q) != 255) {

                        if (q < x1temp) {
                            x1temp = q;
                        }
                        if (q > x2temp) {
                            x2temp = q;
                        }
                        if (p < y1temp) {
                            y1temp = p;
                        }
                        if (p > y2temp) {
                            y2temp = p;
                        }

                        matExtremum.at<uchar> (p, q) = 255;
                        Point_<int> pointtemp;
                        pointtemp.x = q;
                        pointtemp.y = p;
                        temp.push_back(pointtemp);
                    }
                }
            }
            ExtremumPoints->at(i).y = y1temp + ((y2temp - y1temp) / 2);
            ExtremumPoints->at(i).x = x1temp + ((x2temp - x1temp) / 2);
        }
    }
}

void invertMat(Mat mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            mat.at<uchar> (i, j) = CONST_WHITE - mat.at<uchar> (i, j);
        }
    }
}

void matrixToMat(int** input, Mat output) {
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            output.at<uchar>(i, j) = input[i][j];
        }
    }
}

int** creatMatrix(int rowCount, int colCount) {
    int** matrix = new int*[rowCount];
    for (int i = 0; i < rowCount; i++) {
        matrix[i] = new int[colCount];
    }
    return matrix;
}

void initMatrix(int** matrix, int rowCount, int colCount, int value) {
    for (int i = 0; i < rowCount; i++) {
        for (int j = 0; j < colCount; j++) {
            matrix[i][j] = value;
        }
    }
}

void printMatrix(int** matrix, int rowCount, int colCount) {
    for (int i = 0; i < rowCount; i++) {
        for (int j = 0; j < colCount; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void matrixScalerMultiplaction(int** matrix, int rowCount, int colCount, int** divisionMatrix, int n) {
    for (int i = 0; i < rowCount; i++) {
        for (int j = 0; j < colCount; j++) {
            divisionMatrix[i][j] = matrix[i][j] / n;
        }
    }
}


//#define size_matrix 200
//

typedef struct position {
    Point_<int> prev_pos;
    int direction;
    int seq_notUse;
    bool inUse;
} Position;

//#include <chrono>
#include <sys/time.h>
//using namespace std::chrono;

int64 currentTimeMillis() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time_in_mill =
            (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000; // convert tv_sec & tv_usec to millisecond
    return time_in_mill;

}

void People_Counter(String fileName) {
    //
    vector<Position> TrackingPoints;
    //
    int CounterIn = 0;
    int CounterOut = 0;



    //try {

    CvScalar cvscalar_black = cvScalar(CONST_BLACK);
    CvScalar cvscalar_white = cvScalar(CONST_WHITE);

    int64 time_current, time_last = 0;

    //int delay = 1000 / (int) videoCapture.get(CV_CAP_PROP_FPS);

    //videoCapture.set(CV_CAP_PROP_POS_FRAMES, 10);


    //
    char key;
    bool play = true;
    bool stop = false;
    //
    int frame_printtofile = 831;
    int printtofile = 0;
    int invert = 0;
    //
    int segmentation_parts = 10;
    int remove_black = 1;
    int threshold1 = 30;
    int percent = 30;
    int reconstruction = 1;
    int ksize = 3;
    //
    int FindExtremum = 1;
    //
    int NeighborNumber = 10;
    int threshold2 = 5;
    //
    int track = 1;
    //
    const string setting_window_name = "setting";
    const string hot_keys = "ESC exit";
    const string hot_keys2 = "space pause/play";
    //
    Mat massage = Mat::zeros(60, 450, CV_8UC1);
    putText(massage, hot_keys, cvPoint(15, 20), FONT_HERSHEY_SIMPLEX, 0.4, cvscalar_white, 0);
    putText(massage, hot_keys2, cvPoint(15, 20 * 2), FONT_HERSHEY_SIMPLEX, 0.4, cvscalar_white, 0);
    //
    namedWindow(setting_window_name, WINDOW_NORMAL);
    //
    createTrackbar("print", setting_window_name, &printtofile, 1);
    createTrackbar("invert", setting_window_name, &invert, 1);
    createTrackbar("segmentation", setting_window_name, &segmentation_parts, 100);
    createTrackbar("remove black", setting_window_name, &remove_black, 1);
    createTrackbar("threshold1", setting_window_name, &threshold1, 100);
    createTrackbar("percent", setting_window_name, &percent, 100);
    createTrackbar("reconstruction", setting_window_name, &reconstruction, 1);
    createTrackbar("median", setting_window_name, &ksize, 100);
    createTrackbar("FindExtremum", setting_window_name, &FindExtremum, 1);
    createTrackbar("NeighborNumber", setting_window_name, &NeighborNumber, 100);
    createTrackbar("threshold2", setting_window_name, &threshold2, 100);
    createTrackbar("track", setting_window_name, &track, 1);
    //createTrackbar("erod xy", setting_window_name, &erod_x_y, 100);
    //createTrackbar("open xy", setting_window_name, &open_x_y, 100);
    //
    imshow(setting_window_name, massage);
    resizeWindow(setting_window_name, 500, 0);

    //
    vector<Point_<int> > RandomPoints;
    vector<Point_<int> > ExtremumPoints;

    // Mats
    Mat MatFrame;
    Mat MatBackground;
    Mat MatForeground;
    Mat MatForegroundAvg;
    Mat MatExtremum;
    Mat MatForegroundAvgForShow;

    // Start Video
    VideoCapture videoCapture(fileName);
    videoCapture.set(CV_CAP_PROP_MODE, CV_CAP_MODE_GRAY);

    // get background
    videoCapture.read(MatBackground);
    cvtColor(MatBackground, MatBackground, CV_BGR2GRAY, CV_8UC1);

    // map size
    int map_mat_rows = MatBackground.rows / segmentation_parts;
    int map_mat_cols = MatBackground.cols / segmentation_parts;

    // Matrixs
    int** foreground_min = creatMatrix(map_mat_rows, map_mat_cols);
    int** foreground_max = creatMatrix(map_mat_rows, map_mat_cols);
    int** foreground_numberof255 = creatMatrix(map_mat_rows, map_mat_cols);
    int** foreground_sum = creatMatrix(map_mat_rows, map_mat_cols);
    int** foreground_avg = creatMatrix(map_mat_rows, map_mat_cols);


    MatForeground = Mat(MatBackground.rows, MatBackground.cols, CV_8UC1);

    // y1 y2 track
    int y1track = map_mat_rows / 4;
    int y2track = (map_mat_rows * 3) / 4;

    while (!stop) {

        time_current = currentTimeMillis();

        int segmentation_square = segmentation_parts*segmentation_parts;

        if (play) {
            if (videoCapture.read(MatFrame)) {
            } else {
                //videoCapture.set(CV_CAP_PROP_POS_FRAMES, 0);
                break;
            }
            cvtColor(MatFrame, MatFrame, CV_BGR2GRAY, CV_8UC1);
        }

        MatForegroundAvg = Mat(MatFrame.rows / segmentation_parts, MatFrame.cols / segmentation_parts, CV_8UC1);


        initMatrix(foreground_min, map_mat_rows, map_mat_cols, CONST_WHITE);
        initMatrix(foreground_max, map_mat_rows, map_mat_cols, CONST_BLACK);
        initMatrix(foreground_numberof255, map_mat_rows, map_mat_cols, CONST_ZERO);
        initMatrix(foreground_sum, map_mat_rows, map_mat_cols, CONST_ZERO);

        extractForegroundInMat(MatForeground,
                MatBackground,
                MatFrame,
                threshold1,
                remove_black
                );

        matMapFuturesToMatrix(
                MatForeground,
                segmentation_parts,
                foreground_min,
                foreground_max,
                foreground_numberof255,
                foreground_sum
                );

        matrixScalerMultiplaction(foreground_sum, map_mat_rows, map_mat_cols, foreground_avg, segmentation_square);
        matrixToMat(foreground_avg, MatForegroundAvg);

        cleanMat(MatForegroundAvg, foreground_numberof255, segmentation_square, percent);

        if (reconstruction == 1) {
            reconstructionMat(MatForegroundAvg, true, true);
        }

        if (ksize % 2 == 0) {
            ksize++;
        }
        medianBlur(MatForegroundAvg, MatForegroundAvg, ksize);

        // gereftane darsade sefid va siah to nahye xy
        Rect rect = getRectObject(MatForegroundAvg);
        //cout << "x=" << " y=" << " x1=" << x1 << " x2=" << x2 << " y1=" << y1 << " y2=" << y2 << "\n";

        bool small_object = rect.area() >= 0 && rect.area() <= 36;

        if (rect.empty()) {
            if (play) {
                //MatBackground=MatFrame;
                //background = frame;
            }
        } else if (small_object) {
        } else if (FindExtremum == 1) {

            getRandomPoint(MatForegroundAvg, rect, &RandomPoints);

            aaa(MatForegroundAvg, rect, &RandomPoints, NeighborNumber);

            MatExtremum = Mat::zeros(MatForegroundAvg.rows, MatForegroundAvg.cols, CV_8UC1);
            bbb(MatForegroundAvg, MatExtremum, &RandomPoints, &ExtremumPoints, y1track, y2track);

            ccc(MatForegroundAvg, MatExtremum, &ExtremumPoints, threshold2);

            // for video
            for (int i = 0; i < MatForegroundAvg.rows; i++) {
                for (int j = 0; j < MatForegroundAvg.cols; j++) {
                    if (MatExtremum.at<uchar> (i, j) == 255) {
                        MatForegroundAvg.at<uchar> (i, j) = 255;
                    }
                }
            }

            // for video
            MatForegroundAvg.at<uchar> (rect.y, rect.x) = 0;
            MatForegroundAvg.at<uchar> (rect.y, rect.x + rect.width) = 0;
            MatForegroundAvg.at<uchar> (rect.y + rect.height, rect.x) = 0;
            MatForegroundAvg.at<uchar> (rect.y + rect.height, rect.x + rect.width) = 0;
        }

        // for video
        checkeredMat(MatForeground, segmentation_parts);



        for (int i = 0; i < MatForegroundAvg.cols; i++) {
            MatForegroundAvg.at<uchar> (y1track, i) = 0;
            MatForegroundAvg.at<uchar> (y2track, i) = 0;
        }

        for (int i = 0; i < ExtremumPoints.size(); i++) {
            if (MatForegroundAvg.at<uchar> (ExtremumPoints[i]) != 255) {
                MatForegroundAvg.at<uchar> (ExtremumPoints[i]) = 255;
            } else {
                MatForegroundAvg.at<uchar> (ExtremumPoints[i]) = 0;
            }
        }

        if (invert == 1) {
            invertMat(MatForegroundAvg);
        }

        if (printtofile == 1) {
            for (int i = 0; i < MatForegroundAvg.rows; i++) {
                for (int j = 0; j < MatForegroundAvg.cols; j++) {
                    if (videoCapture.get(CV_CAP_PROP_POS_FRAMES) == frame_printtofile) {
                        //printf("%3d ", MatForegroundAvg.at<uchar> (i, j));
                    }
                }
                if (videoCapture.get(CV_CAP_PROP_POS_FRAMES) == frame_printtofile) {
                    cout << "\n";
                }
            }
        }


        MatForegroundAvgForShow = Mat(MatFrame.rows, MatFrame.cols, CV_8UC1);
        for (int i = 0; i < MatFrame.rows; i++) {
            for (int j = 0; j < MatFrame.cols; j++) {
                int p = i / segmentation_parts;
                int q = j / segmentation_parts;
                MatForegroundAvgForShow.at<uchar> (i, j) = MatForegroundAvg.at<uchar> (p, q);
                if (invert) {
                    if (i % segmentation_parts == 0 || j % segmentation_parts == 0) {
                        MatForegroundAvgForShow.at<uchar> (i, j) = 255;
                    }
                } else {
                    if (i % segmentation_parts == 0 || j % segmentation_parts == 0) {
                        MatForegroundAvgForShow.at<uchar> (i, j) = 0;
                    }
                }
            }
        }





        //setMouseCallback("Play", Distance_16bit, &frame);



        stringstream str;

        str = stringstream();
        int i = time_current - time_last;
        if (i == 0) {
            str << "Frame Rate: " << "inf";
        } else {
            str << "Frame Rate: " << (int) (1000 / i);
        }
        putText(MatForegroundAvgForShow, str.str(), cvPoint(2, 25), FONT_HERSHEY_PLAIN, 2, cvscalar_black, 2);

        str = stringstream();
        str << "Frame Counter: " << videoCapture.get(CV_CAP_PROP_POS_FRAMES);
        putText(MatForegroundAvgForShow, str.str(), cvPoint(2, 50), FONT_HERSHEY_PLAIN, 2, cvscalar_black, 2);
        cout << str.str() << "\n";

        str = stringstream();
        str << "In: " << CounterIn;
        putText(MatForegroundAvgForShow, str.str(), cvPoint(2, 75), FONT_HERSHEY_PLAIN, 2, cvscalar_black, 2);

        str = stringstream();
        str << "Out: " << CounterOut;
        putText(MatForegroundAvgForShow, str.str(), cvPoint(2, 100), FONT_HERSHEY_PLAIN, 2, cvscalar_black, 2);


        //imshow("MatBackground", MatBackground);
        imshow("MatFrame", MatFrame);
        imshow("MatForeground", MatForeground);
        imshow("MatForegroundAvg", MatForegroundAvg);
        imshow("MatForegroundAvgForShow", MatForegroundAvgForShow);



        key = waitKey(1);
        if (key == ',') {
            if (play) {
                int temp = (int) videoCapture.get(CV_CAP_PROP_POS_FRAMES);
                temp = temp - 31;
                if (temp < 0) {
                    temp = 0;
                }
                videoCapture.set(CV_CAP_PROP_POS_FRAMES, temp);
            } else {
                int temp = (int) videoCapture.get(CV_CAP_PROP_POS_FRAMES);
                temp = temp - 2;
                if (temp < 0) {
                    temp = 0;
                }
                videoCapture.set(CV_CAP_PROP_POS_FRAMES, temp);
                videoCapture.read(MatFrame);
                cvtColor(MatFrame, MatFrame, CV_BGR2GRAY, CV_8UC1);
            }
        } else if (key == '.') {
            if (play) {
                int temp = (int) videoCapture.get(CV_CAP_PROP_POS_FRAMES);
                temp = temp + 30;
                if (temp > videoCapture.get(CV_CAP_PROP_FRAME_COUNT)) {
                    temp = 0;
                }
                videoCapture.set(CV_CAP_PROP_POS_FRAMES, temp);
            } else {
                int temp = (int) videoCapture.get(CV_CAP_PROP_POS_FRAMES);
                //temp = temp + 1;
                if (temp > videoCapture.get(CV_CAP_PROP_FRAME_COUNT)) {
                    temp = 0;
                }
                videoCapture.set(CV_CAP_PROP_POS_FRAMES, temp);
                videoCapture.read(MatFrame);
                cvtColor(MatFrame, MatFrame, CV_BGR2GRAY, CV_8UC1);
            }
        } else if (key == ' ') {
            play = !play;
        } else if (key == 27) {
            stop = true;
        }

        time_last = time_current;
    }
    videoCapture.release();
    //} catch (exception& e) {
    //    cout << "Exception: " << e.what() << "\n";
    //}
}
