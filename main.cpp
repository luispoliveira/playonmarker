//compile g++ $(pkg-config --cflags --libs opencv4) -lmpg123 -lao -std=c++11 main.cpp
//run ./a.out
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <nlohmann/json.hpp>

#include <mpg123.h> //decode mp3
#include <ao/ao.h> //make sound

#include <fstream>
#include <sstream>
#include <iostream>

using json = nlohmann::json;


using namespace cv;
using namespace aruco;
using namespace std;


int mainMenu();

void generateMarker();

int beginInteration();


class Marker {
public:

    const string IMAGE = "image";
    const string VIDEO = "video";


    string media_type;//need to be set
    string media_path;//need to be set
    string type; //{"single" or combine}

    string function;
    int slaveId;

    string placeholderFinal_path;
    string placeholderInicial_path;


    time_t detectedAt;//need to be set
    time_t lastTimeDetected;
    bool isDetected = false;
    int delay = 0;//seconds? need to be set


    bool toLoop = false;


    int markerId; //teste
    int markerIdPosition; //teste


    vector<Point2f> markerCorners;
    vector<Point> pts_dst;
    vector<Point> pts_src;

    Mat image_src;
    bool image_src_loaded = false;

    VideoCapture video;

    bool video_loaded = false;

    int timeToPause = 4;//segundos

    bool playOnMarker() {
        if (isDetected) {
            return true;
        } else {
            time_t current_time = time(nullptr);

            time_t stopPlayingAt = lastTimeDetected + timeToPause;

            return stopPlayingAt >= current_time;
        }
    }


    bool showMedia() const {
        time_t current_time = time(nullptr);
        time_t showAt = detectedAt + delay;
        return current_time >= showAt;
    }

    void setDetection() {
        isDetected = true;
        if (detectedAt == 0) {
            detectedAt = time(nullptr);
        }
    }

    void loopVideo() {
        /**
         * voltar a colocar o video no frame 0
         * passar o frame para o image_src
         */
        video.set(CAP_PROP_POS_FRAMES, 0);
        video >> image_src;
    }

    void setImgSrc() {
        if (showMedia()) { // para fazer o delay
            if (media_type == IMAGE) {
                if (!image_src_loaded) {
                    image_src = imread(media_path);
                    image_src_loaded = true;
                }
            } else if (media_type == VIDEO) {
                if (!video_loaded) {
                    video.open(media_path);
                    video_loaded = true;
                }
                /**
                 * faz load da frame do video para o image_src
                 * utilizamos a image_src para fazer a homografia
                 * */
                video >> image_src;

                /**
                 * quando acabamos o video verificamos se é para fazer loop
                 */

                if (image_src.empty()) {
                    if (toLoop) loopVideo();

                    if (!(placeholderFinal_path.empty())) {
                        image_src = imread(placeholderFinal_path);
                    }

                }
            }
        } else {
            if (!placeholderInicial_path.empty()) {
                image_src = imread(placeholderInicial_path);
            }
        }
    }

    void setPtsSrc() {
        pts_src.clear();
        pts_src.emplace_back(0, 0);
        pts_src.emplace_back(image_src.cols, 0);
        pts_src.emplace_back(image_src.cols, image_src.rows);
        pts_src.emplace_back(0, image_src.rows);
    }
};

/**
 * variaveis usadas a nivel global
 */
Marker markers[256];

Mat homographys[255];
Mat warpedImages[255];
Mat homography_mask[255];
Mat homography_element[255];
Mat imOut;


void computeHomography(int markerId, const Mat &frame) {

    homographys[markerId] = findHomography(markers[markerId].pts_src, markers[markerId].pts_dst);

    warpPerspective(markers[markerId].image_src, warpedImages[markerId], homographys[markerId], frame.size(),
                    INTER_CUBIC);

    homography_mask[markerId] = Mat::zeros(frame.rows, frame.cols, CV_8UC1);

    fillConvexPoly(homography_mask[markerId], markers[markerId].pts_dst, Scalar(255, 255, 255), LINE_AA);

    homography_element[markerId] = getStructuringElement(MORPH_RECT, Size(5, 5));

    erode(homography_mask[markerId], homography_mask[markerId], homography_element[markerId]);

    warpedImages[markerId].copyTo(imOut, homography_mask[markerId]);

}


void setMarkerProperties(int markerId) {
    /**
    * read json
    */
    ifstream config_json("../config.json");
    json ar_config;
    config_json >> ar_config;

    /**
     * carregar info do json para a class
     */

    json marker_config = ar_config[to_string(markerId)];


    markers[markerId].media_path = marker_config["media_path"];
    markers[markerId].media_type = marker_config["media_type"];
    markers[markerId].type = marker_config["type"];


    if (marker_config.contains("function")) {
        markers[markerId].function = marker_config["function"];
    }

    if (marker_config.contains("slaveId")) {
        markers[markerId].slaveId = marker_config["slaveId"];
    }

    if (marker_config.contains("toLoop")) {
        markers[markerId].toLoop = marker_config["toLoop"];
    }
    if (marker_config.contains("placeholderFinal_path")) {
        markers[markerId].placeholderFinal_path = marker_config["placeholderFinal_path"];
    }
    if (marker_config.contains("placeholderInicial_path")) {
        markers[markerId].placeholderInicial_path = marker_config["placeholderInicial_path"];
    }
    if (marker_config.contains("delay")) {
        markers[markerId].delay = marker_config["delay"];
    }

    markers[markerId].setDetection();
}


void generateMarker() {
    int idMarker;

    cout << "Que markers deseja gerar(1-250)?" << endl;
    cin >> idMarker;

    if (idMarker < 1 || idMarker > 250) {
        cout << "Esse marker não existe" << endl;
        generateMarker();
    } else {
        Mat markerImage;
        Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        aruco::drawMarker(dictionary, idMarker, 200, markerImage, 1);
        if (imwrite("markers/marker" + std::to_string(idMarker) + ".png", markerImage)) {
            cout << "Marker criado com sucesso. (Verificar diretório markers)" << endl;
        } else {
            cout << "Ocurreu um erro a criar o marker" << endl;
        }
    }

    mainMenu();
}

int beginInteration() {
    Ptr<Dictionary> dictionary = getPredefinedDictionary(DICT_6X6_250);
    Ptr<DetectorParameters> parameters = DetectorParameters::create();


//    parameters->cornerRefinementMethod = CORNER_REFINE_SUBPIX;
//    parameters->errorCorrectionRate = 0.1;
//    parameters->aprilTagDeglitch = 1;
//    parameters->maxErroneousBitsInBorderRate = 2.5;
    //void MarkerDetector::Params::setDetectionMode( DetectionMode dm,float minMarkerSize=0);

    VideoCapture cap;
    Mat frame;


    try {
        cap.open(0);

    } catch (...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }


    while (waitKey(1) < 0) {

        cap >> frame;
        imOut = frame.clone();
        try {
            vector<int> markersIds;
            vector<vector<Point2f>> markerCorners, rejectedCandidates;
            detectMarkers(frame, dictionary, markerCorners, markersIds, parameters);

            Mat outputImage = frame.clone();


            for (auto &marker : markers) {
                marker.isDetected = false;
            }


            if (!markersIds.empty()) {
                for (int i = 0; i < markersIds.size(); i++) {

                    int markerId = markersIds[i];


                    if (!markers[markerId].isDetected) {
                        setMarkerProperties(markerId);
                    }


                    markers[markerId].markerIdPosition = i;
                    markers[markerId].markerId = markerId;
                    markers[markerId].markerCorners = markerCorners.at(i);
                    markers[markerId].lastTimeDetected = time(nullptr);
                    //markers[markerId].setImgSrc();

                    /**
                     * pontos do markers
                     */

//                    Point refPt1, refPt2, refPt3, refPt4;
//
//                    refPt1 = markerCorners.at(i).at(3); //ponto superior direito
//                    refPt2 = markerCorners.at(i).at(0); //ponto superior esquerdo
//                    refPt3 = markerCorners.at(i).at(1); //ponto inferior direito
//                    refPt4 = markerCorners.at(i).at(2); //ponto inferior esquerdo
//
//
//                    markers[markerId].pts_dst.clear();
//                    markers[markerId].pts_dst.push_back(refPt1);
//                    markers[markerId].pts_dst.push_back(refPt2);
//                    markers[markerId].pts_dst.push_back(refPt3);
//                    markers[markerId].pts_dst.push_back(refPt4);


                    /**
                     * pontos da imagem
                     */
//                    markers[markerId].setPtsSrc();

                    /**
                     * computar homografia
                     */

//                    computeHomography(markerId, frame);

//                    imshow("warpedImage", imOut);
                }

                drawDetectedMarkers(outputImage, markerCorners, markersIds);
            }

            for (auto &marker : markers) {
                if (marker.playOnMarker() /*marker.isDetected*/) {
                    if (marker.type == "single") {

                        marker.setImgSrc();

                        Point refPt1, refPt2, refPt3, refPt4;

//                        refPt1 = markerCorners.at(marker.markerIdPosition).at(3); //ponto superior direito
//                        refPt2 = markerCorners.at(marker.markerIdPosition).at(0); //ponto superior esquerdo
//                        refPt3 = markerCorners.at(marker.markerIdPosition).at(1); //ponto inferior direito
//                        refPt4 = markerCorners.at(marker.markerIdPosition).at(2); //ponto inferior esquerdo

//                        refPt1 = markerCorners.at(marker.markerIdPosition).at(0); //ponto superior esquerdo
//                        refPt2 = markerCorners.at(marker.markerIdPosition).at(1); //ponto superior direito
//                        refPt3 = markerCorners.at(marker.markerIdPosition).at(2); //ponto inferior direito
//                        refPt4 = markerCorners.at(marker.markerIdPosition).at(3); //ponto inferior esquerdo

                        refPt1 = marker.markerCorners.at(0); //ponto superior esquerdo
                        refPt2 = marker.markerCorners.at(1); //ponto superior direito
                        refPt3 = marker.markerCorners.at(2); //ponto inferior direito
                        refPt4 = marker.markerCorners.at(3); //ponto inferior esquerdo



                        marker.pts_dst.clear();
                        marker.pts_dst.push_back(refPt1);
                        marker.pts_dst.push_back(refPt2);
                        marker.pts_dst.push_back(refPt3);
                        marker.pts_dst.push_back(refPt4);


                        /**
                         * pontos da imagem
                         */
                        marker.setPtsSrc();

                        /**
                         * computar homografia
                         */
                        computeHomography(marker.markerId, frame);
                    } else if (marker.type == "combine") {
                        if (marker.function == "master") {

                            int slaveId = marker.slaveId;

                            if (markers[slaveId].isDetected) {

                                Marker master = marker;
                                Marker slave = markers[slaveId];

                                marker.setImgSrc();


                                Point refPt1, refPt2, refPt3, refPt4;


                                refPt1 = markerCorners.at(marker.markerIdPosition).at(0);//ponto superior esquerdo
//
                                refPt2.x = markerCorners.at(markers[slaveId].markerIdPosition).at(
                                        1).x;//ponto superior direito
                                refPt2.y = markerCorners.at(marker.markerIdPosition).at(1).y;//ponto superior direito



                                refPt3 = markerCorners.at(markers[slaveId].markerIdPosition).at(
                                        2); //ponto inferior direito



                                refPt4.x = markerCorners.at(marker.markerIdPosition).at(3).x; //ponto inferior esquerdo
                                refPt4.y = markerCorners.at(markers[slaveId].markerIdPosition).at(
                                        3).y; //ponto inferior esquerdo



                                marker.pts_dst.clear();
                                marker.pts_dst.push_back(refPt1);
                                marker.pts_dst.push_back(refPt2);
                                marker.pts_dst.push_back(refPt3);
                                marker.pts_dst.push_back(refPt4);


                                /**
                                 * pontos da imagem
                                 */
                                marker.setPtsSrc();

                                /**
                                 * computar homografia
                                 */
                                computeHomography(marker.markerId, frame);
                            }


                        } else {
                            //do nothing
                        }

                    }
                }
            }
            imshow("out", outputImage);

            imshow("warpedImage", imOut);

        } catch (const std::exception &e) {
            cout << endl << " e : " << e.what() << endl;
            cout << "Could not do homography !! " << endl;
            //        return 0;
        }
    }
    return 0;
}


int mainMenu() {

    int menuChosen;


    cout << "1 - Criar marker" << endl;
    cout << "2 - Começar interatividade" << endl;
    cout << "3 - Fechar" << endl;
    cout << "Selecione o que deseja fazer:";

    cin >> menuChosen;
    cout << endl;


    switch (menuChosen) {
        case 1:
            generateMarker();
            break;
        case 2:
            beginInteration();
            break;
        case 3:
            return 0;
        default:
            mainMenu();
            break;
    }
    return 0;
}


int main(int argc, char **argv) {

    cout << "Bem Vindo ao protótipo do livro interativo." << endl;

    return mainMenu();
}