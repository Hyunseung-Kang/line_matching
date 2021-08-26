#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#define MATCHES_DIST_THRESHOLD 25

using namespace cv;
using namespace cv::line_descriptor;


int main(int argc, char** argv)
{

    /* load image */
    cv::Mat imageMat1 = imread("maps/m1_edit.png", 1);
    cv::Mat imageMat2 = imread("maps/m2_edit.png", 1);

    if (imageMat1.data == NULL || imageMat2.data == NULL)
    {
        std::cout << "Error, images could not be loaded. Please, check their path" << std::endl;
    }

    /* create binary masks */
    cv::Mat mask1 = Mat::ones(imageMat1.size(), CV_8UC1);
    cv::Mat mask2 = Mat::ones(imageMat2.size(), CV_8UC1);

    /* create a pointer to a BinaryDescriptor object with default parameters */

    /// Binary Descriptor는 무엇을 하는 놈인가?
    /// BinaryDescriptor형의 포인터 bd는 
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

    /* compute lines and descriptors */
    std::vector<KeyLine> keylines1, keylines2;
    cv::Mat descr1, descr2;


    /// 앞의 (*bd)는 무엇을 나타내는가?
    /// imageMat은 입력영상, mask는 입력영상과 크기가 같고 모든 값이 1인 Mat
    /// keylines는 KeyLine 형이 원소로 들어가는 벡터를 나타내고
    /// descr은 무엇을 나타내는지 확인이 필요하다.
    /// 
    /// 이 과정을 통해 keylines가 검출된다.
    std::cout <<"before"<< keylines1.size() << std::endl;
    std::cout << "before" << keylines2.size() << std::endl;
    (*bd)(imageMat1, mask1, keylines1, descr1, false, false);
    (*bd)(imageMat2, mask2, keylines2, descr2, false, false);

    std::cout << keylines1.size() << std::endl;
    std::cout << keylines2.size() << std::endl;
    /* select keylines from first octave and their descriptors */
    std::vector<KeyLine> lbd_octave1, lbd_octave2;
    Mat left_lbd, right_lbd;    // maybe lbd is line binary discriptor
    for (int i = 0; i < (int)keylines1.size(); i++)
    {
        if (keylines1[i].octave == 0)
        {
            lbd_octave1.push_back(keylines1[i]);
            left_lbd.push_back(descr1.row(i));
        }
    }

    for (int j = 0; j < (int)keylines2.size(); j++)
    {
        if (keylines2[j].octave == 0)
        {
            lbd_octave2.push_back(keylines2[j]);
            right_lbd.push_back(descr2.row(j));
        }
    }

    /* create a BinaryDescriptorMatcher object */
    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    std::vector<DMatch> matches;
    bdm->match(left_lbd, right_lbd, matches);

    /* select best matches */
    std::vector<DMatch> good_matches;
    for (int i = 0; i < (int)matches.size(); i++)
    {
        if (matches[i].distance < MATCHES_DIST_THRESHOLD)
            good_matches.push_back(matches[i]);
    }

    /* plot matches */
    cv::Mat outImg;
    cv::Mat scaled1, scaled2;
    std::vector<char> mask(matches.size(), 1);
    drawLineMatches(imageMat1, lbd_octave1, imageMat2, lbd_octave2, good_matches, outImg, Scalar::all(-1), Scalar::all(-1), mask,
        DrawLinesMatchesFlags::DEFAULT);

    imshow("Matches", outImg);
    waitKey();
    imwrite("matches.jpg", outImg);
    /* create an LSD detector */
    Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();

    /* detect lines */
    std::vector<KeyLine> klsd1, klsd2;
    Mat lsd_descr1, lsd_descr2;
    lsd->detect(imageMat1, klsd1, 2, 2, mask1);
    lsd->detect(imageMat2, klsd2, 2, 2, mask2);

    /* compute descriptors for lines from first octave */
    bd->compute(imageMat1, klsd1, lsd_descr1);
    bd->compute(imageMat2, klsd2, lsd_descr2);

    /* select lines and descriptors from first octave */
    std::vector<KeyLine> octave0_1, octave0_2;
    Mat leftDEscr, rightDescr;
    for (int i = 0; i < (int)klsd1.size(); i++)
    {
        if (klsd1[i].octave == 1)
        {
            octave0_1.push_back(klsd1[i]);
            leftDEscr.push_back(lsd_descr1.row(i));
        }
    }

    for (int j = 0; j < (int)klsd2.size(); j++)
    {
        if (klsd2[j].octave == 1)
        {
            octave0_2.push_back(klsd2[j]);
            rightDescr.push_back(lsd_descr2.row(j));
        }
    }

    /* compute matches */
    std::vector<DMatch> lsd_matches;
    bdm->match(leftDEscr, rightDescr, lsd_matches);

    /* select best matches */
    good_matches.clear();
    for (int i = 0; i < (int)lsd_matches.size(); i++)
    {
        if (lsd_matches[i].distance < MATCHES_DIST_THRESHOLD)
            good_matches.push_back(lsd_matches[i]);
    }

    /* plot matches */
    cv::Mat lsd_outImg;
    resize(imageMat1, imageMat1, Size(imageMat1.cols / 2, imageMat1.rows / 2));
    resize(imageMat2, imageMat2, Size(imageMat2.cols / 2, imageMat2.rows / 2));
    std::vector<char> lsd_mask(matches.size(), 1);
    drawLineMatches(imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all(-1), Scalar::all(-1), lsd_mask,
        DrawLinesMatchesFlags::DEFAULT);

    imshow("LSD matches", lsd_outImg);
    waitKey();


}