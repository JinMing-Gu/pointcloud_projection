// 加载标定数据
void loadCalibrationData(cv::Mat &P_rect_00, cv::Mat &RT, cv::Mat &distCoeffs)
{
    //! horizon-MVS version
    // extrinsic(horizon-MVS version)
    // RT.at<double>(0, 0) = 0;
    // RT.at<double>(0, 1) = -1;
    // RT.at<double>(0, 2) = 0;
    // RT.at<double>(0, 3) = 0;
    // RT.at<double>(1, 0) = 0;
    // RT.at<double>(1, 1) = 0;
    // RT.at<double>(1, 2) = -1;
    // RT.at<double>(1, 3) = 0;
    // RT.at<double>(2, 0) = 1;
    // RT.at<double>(2, 1) = 0;
    // RT.at<double>(2, 2) = 0;
    // RT.at<double>(2, 3) = 0;
    // RT.at<double>(3, 0) = 0.0;
    // RT.at<double>(3, 1) = 0.0;
    // RT.at<double>(3, 2) = 0.0;
    // RT.at<double>(3, 3) = 1.0;

    // intrinsic(horizon-MVS version)
    // 957.994  0.0  790.335
    // 0.0  955.3280  250.6631
    // 0.0  0.0  1.0
    // P_rect_00.at<double>(0, 0) = 957.994;
    // P_rect_00.at<double>(0, 1) = 0.0;
    // P_rect_00.at<double>(0, 2) = 790.335;
    // P_rect_00.at<double>(0, 3) = 0.0;
    // P_rect_00.at<double>(1, 0) = 0;
    // P_rect_00.at<double>(1, 1) = 955.3280;
    // P_rect_00.at<double>(1, 2) = 250.6631;
    // P_rect_00.at<double>(1, 3) = 0;
    // P_rect_00.at<double>(2, 0) = 0.0;
    // P_rect_00.at<double>(2, 1) = 0.0;
    // P_rect_00.at<double>(2, 2) = 1.0;
    // P_rect_00.at<double>(2, 3) = 0;

    // distCoeffs(horizon-MVS version)
    // k1: -0.12
    // k2: 0.1162
    // p1: 0.0
    // p2: 0.0
    // Ck3: 0.0

    // resolution(horizon-MVS version)
    // width: 1520
    // height: 568

    //! uisee version
    // extrinsic(uisee version)
    RT.at<double>(0, 0) = 0.0229327;
    RT.at<double>(0, 1) = -0.999535;
    RT.at<double>(0, 2) = -0.0200773;
    RT.at<double>(0, 3) = -0.0707031;
    RT.at<double>(1, 0) = -0.0372575;
    RT.at<double>(1, 1) = 0.0192141;
    RT.at<double>(1, 2) = -0.999121;
    RT.at<double>(1, 3) = -0.0533916;
    RT.at<double>(2, 0) = 0.999043;
    RT.at<double>(2, 1) = 0.0236606;
    RT.at<double>(2, 2) = -0.0367995;
    RT.at<double>(2, 3) = -0.0698497;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    // intrinsic(uisee version)
    P_rect_00.at<double>(0, 0) = 964.36052256;
    P_rect_00.at<double>(0, 1) = 0.0;
    P_rect_00.at<double>(0, 2) = 629.68157859;
    P_rect_00.at<double>(0, 3) = 0.0;
    P_rect_00.at<double>(1, 0) = 0;
    P_rect_00.at<double>(1, 1) = 963.81358811;
    P_rect_00.at<double>(1, 2) = 374.90701638;
    P_rect_00.at<double>(1, 3) = 0;
    P_rect_00.at<double>(2, 0) = 0.0;
    P_rect_00.at<double>(2, 1) = 0.0;
    P_rect_00.at<double>(2, 2) = 1.0;
    P_rect_00.at<double>(2, 3) = 0;

    // distCoeffs(uisee version)
    // k1: -0.37675309
    // k2: 0.13327285
    // p1: 0.00040131
    // p2: -0.00030038
    // k3: 0.0
    distCoeffs.at<double>(0, 0) = -0.37675309;
    distCoeffs.at<double>(1, 0) = 0.13327285;
    distCoeffs.at<double>(2, 0) = 0.00040131;
    distCoeffs.at<double>(3, 0) = -0.00030038;
    distCoeffs.at<double>(4, 0) = 0.0;

    // resolution(uisee version)
    // width: 1280
    // height: 720
}