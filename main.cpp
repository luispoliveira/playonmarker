#include <lzma.h>
//compile g++ $(pkg-config --cflags --libs opencv4) -lsfml-audio -std=c++11 main.cpp
//run ./a.out
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <nlohmann/json.hpp>

#include <SFML/Audio.hpp>

#include <fstream>
#include <sstream>
#include <iostream>


using json = nlohmann::json;


using namespace cv;
using namespace aruco;
using namespace std;
using namespace sf;

int mainMenu();

void generateMarker();

int beginInteration();

const string IMAGE = "image";
const string VIDEO = "video";

const string TYPE_SINGLE = "single";
const string TYPE_COMBINE = "combine";

const string FUNCTION_SLAVE = "slave";
const string FUNCTION_MASTER = "master";

const string MARKER_TYPE_CONFIG = "config";
const string MARKER_TYPE_MEDIA = "media";


string CURRENT_LANGUAGE = "PT";
string LAST_LANGUAGE = "";

class Marker {
public:


    time_t detectedAt;//need to be set
    time_t lastTimeDetected;
    bool isDetected = false;


    int markerId; //teste
    int markerIdPosition; //teste


    vector <Point2f> markerCorners;
    vector <Point> pts_dst;
    vector <Point> pts_src;

    Mat image_src;
    bool image_src_loaded = false;


    void setDetection() {
        isDetected = true;
        if (detectedAt == 0) {
            detectedAt = time(nullptr);
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


class Config_Marker : public Marker {

public:
    bool isUsed = false;

    string media_path;
    string language; //vai definir a linguagem

    int timeToSelect = 2;

    void setImgSrc() {
        if (!image_src_loaded) {
            image_src = imread(media_path);
            image_src_loaded = true;
        }
    }

    bool selectLanguage() {
        time_t current_time = time(nullptr);

        time_t selectAt = lastTimeDetected + timeToSelect;

        return selectAt < current_time;
    }

};

Music music;

class Media_Marker : public Marker {

public:

    bool isUsed = false;

    string media_type;
    string media_path;
    string media_path_PT;
    string media_path_ES;
    string media_path_FR;
    string type;


    string function;
    int slaveId = 0;

    string placeholderFinal_path;
    string placeholderInicial_path;

    int delay = 0;

    bool toLoop = false;


    string audio_path = "";
    bool audio_loaded = false;
    bool audio_playing = false;

    VideoCapture video;
    bool video_loaded = false;
    double current_frame = 0;

    int timeToStop = 5;


    bool stopPlaying() {
        time_t current_time = time(nullptr);

        time_t stopPlayingAt = lastTimeDetected + timeToStop;

        if (stopPlayingAt < current_time) {
            if (media_type == VIDEO) {
                loopVideo();
                music.stop();
                audio_playing = false;
            }
        }
        return stopPlayingAt < current_time;
    }

    bool playOnMarker() {
        return isDetected;
    }

    bool showMedia() const {
        time_t current_time = time(nullptr);
        time_t showAt = detectedAt + delay;
        return current_time >= showAt;
    }


    void loopVideo() {
        /**
         * voltar a colocar o video no frame 0
         * passar o frame para o image_src
         */
        video.set(CAP_PROP_POS_FRAMES, 0);
        video >> image_src;
    }

    static void loopSound() {
        music.setLoop(true);
    }

    void setImgSrc() {
        if (showMedia()) { // para fazer o delay
            if (CURRENT_LANGUAGE != LAST_LANGUAGE) {
                LAST_LANGUAGE = CURRENT_LANGUAGE;
                video_loaded = false;
                image_src_loaded = false;
                if (CURRENT_LANGUAGE == "PT") {
                    media_path = media_path_PT;
                } else if (CURRENT_LANGUAGE == "ES") {
                    media_path = media_path_ES;
                } else if (CURRENT_LANGUAGE == "FR") {
                    media_path = media_path_FR;
                } else {
                    media_path = media_path_PT;
                }

            }


            if (media_type == IMAGE) {
                if (!image_src_loaded) {
                    image_src = imread(media_path);
                    image_src_loaded = true;
                }
            } else if (media_type == VIDEO) {
                if (!video_loaded) {
                    video.open(media_path);
                    video.set(CAP_PROP_POS_FRAMES, current_frame);
                    video_loaded = true;
                }

                /**
                 * carragr ficheiro de audio
                 */
                if (!audio_loaded) {
                    if (!music.openFromFile(audio_path))
                        cout << "erro ao carreagr som" << endl;
                    audio_loaded = true;
                }


                /**
                 * faz load da frame do video para o image_src
                 * utilizamos a image_src para fazer a homografia
                 * */
                if (playOnMarker()) {
                    video >> image_src;
                    if (!audio_playing) {
                        music.play();
                        audio_playing = true;
                    }
                    current_frame = video.get(CAP_PROP_POS_FRAMES);
                } else {
                    if (audio_playing) {
                        music.pause();
                        audio_playing = false;
                    }
                }

                /**
                 * quando acabamos o video verificamos se é para fazer loop
                 */

                if (image_src.empty()) {
                    if (toLoop) {
                        loopVideo();
                        loopSound();
                    }
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
};


/**
 * variaveis usadas a nivel global
 */
Media_Marker media_markers[256];
Config_Marker config_markers[256];

Mat homographys[255];
Mat warpedImages[255];
Mat homography_mask[255];
Mat homography_element[255];
Mat imOut;


void computeConfigHomography(int markerId, const Mat &frame) {

    homographys[markerId] = findHomography(config_markers[markerId].pts_src, config_markers[markerId].pts_dst);

    warpPerspective(config_markers[markerId].image_src, warpedImages[markerId], homographys[markerId], frame.size(),
                    INTER_CUBIC);

    homography_mask[markerId] = Mat::zeros(frame.rows, frame.cols, CV_8UC1);

    fillConvexPoly(homography_mask[markerId], config_markers[markerId].pts_dst, Scalar(255, 255, 255), LINE_AA);

    homography_element[markerId] = getStructuringElement(MORPH_RECT, Size(5, 5));

    erode(homography_mask[markerId], homography_mask[markerId], homography_element[markerId]);

    warpedImages[markerId].copyTo(imOut, homography_mask[markerId]);

}

void computeMediaHomography(int markerId, const Mat &frame) {

    homographys[markerId] = findHomography(media_markers[markerId].pts_src, media_markers[markerId].pts_dst);

    warpPerspective(media_markers[markerId].image_src, warpedImages[markerId], homographys[markerId], frame.size(),
                    INTER_CUBIC);

    homography_mask[markerId] = Mat::zeros(frame.rows, frame.cols, CV_8UC1);

    fillConvexPoly(homography_mask[markerId], media_markers[markerId].pts_dst, Scalar(255, 255, 255), LINE_AA);

    homography_element[markerId] = getStructuringElement(MORPH_RECT, Size(5, 5));

    erode(homography_mask[markerId], homography_mask[markerId], homography_element[markerId]);

    warpedImages[markerId].copyTo(imOut, homography_mask[markerId]);

}


/**
 * verifica se o marcador é de config ou de media
 * @param markerId
 * @return
 */
string checkMarkerType(int markerId) {
    /**
    * read json
    */
    ifstream config_json("./config.json");
    json ar_config;
    config_json >> ar_config;

    json marker_config = ar_config[to_string(markerId)];

    if (!marker_config.contains("marker_type")) {
        cout << "Ficheiro de configuração sem \"marker_type \" para o marcador : " << markerId << endl;
        return "ERRO";
    }

    if (marker_config["marker_type"] == MARKER_TYPE_CONFIG) {
        return MARKER_TYPE_CONFIG;
    } else if (marker_config["marker_type"] == MARKER_TYPE_MEDIA) {
        return MARKER_TYPE_MEDIA;
    } else {
        cout << "Ficheiro de configuração não reconhece o tipo \"" << marker_config["marker_type"]
             << "\"  para o marcador : " << markerId << endl;
        return "ERRO";
    }


}

bool setMarkerConfigProperties(int markerId) {
    /**
   * read json
   */
    ifstream config_json("./config.json");
    json ar_config;
    config_json >> ar_config;


    /**
     * carregar info do json para a class
     */

    json marker_config = ar_config[to_string(markerId)];
    if (!marker_config.contains("media_path")) {
        cout << "Ficheiro de configuração sem \"media_path \" para o marcador : " << markerId << endl;
        return false;
    }
    config_markers[markerId].media_path = marker_config["media_path"];
    if (!marker_config.contains("language")) {
        cout << "Ficheiro de configuração sem \"language \" para o marcador : " << markerId << endl;
        return false;
    }
    config_markers[markerId].language = marker_config["language"];
    config_markers[markerId].setDetection();
    config_markers[markerId].isUsed = true;
    return true;
}

/**
 * faz set das configurações para marcador do tipo media
 * @param markerId
 */
bool setMarkerMediaProperties(int markerId) {
    /**
    * read json
    */
    ifstream config_json("./config.json");
    json ar_config;
    config_json >> ar_config;

    /**
     * carregar info do json para a class
     */

    json marker_config = ar_config[to_string(markerId)];

    if (!marker_config.contains("media_path")) {
        cout << "Ficheiro de configuração sem \"media_path \" para o marcador : " << markerId << endl;
        return false;
    }
    json media_path = marker_config["media_path"];

    media_markers[markerId].media_path_PT = media_path["PT"];
    media_markers[markerId].media_path_FR = media_path["FR"];
    media_markers[markerId].media_path_ES = media_path["ES"];

    if (!marker_config.contains("media_type")) {
        cout << "Ficheiro de configuração sem \"media_type \" para o marcador : " << markerId << endl;
        return false;
    }
    media_markers[markerId].media_type = marker_config["media_type"];

    if (!marker_config.contains("type")) {
        cout << "Ficheiro de configuração sem \"type \" para o marcador : " << markerId << endl;
        return false;
    }

    string type = marker_config["type"];
    media_markers[markerId].type = type;

    if (type == TYPE_SINGLE) {
//do nothing
    } else if (type == TYPE_COMBINE) {

        if (!marker_config.contains("function")) {
            cout << "Ficheiro de configuração sem \"function \" para o marcador : " << markerId << endl;
            return false;
        }
        string function = marker_config["function"];
        media_markers[markerId].function = function;

        if (function == FUNCTION_SLAVE) {
            //do nothing
        } else if (function == FUNCTION_MASTER) {

            if (!marker_config.contains("slaveId")) {
                cout << "Ficheiro de configuração sem \"slaveId \" para o marcador : " << markerId << endl;
                return false;
            }
            media_markers[markerId].slaveId = marker_config["slaveId"];
        }
    }

    if (marker_config.contains("audio_path")) {
        media_markers[markerId].audio_path = marker_config["audio_path"];
    }

    if (marker_config.contains("toLoop")) {
        media_markers[markerId].toLoop = marker_config["toLoop"];
    }
    if (marker_config.contains("placeholderFinal_path")) {
        media_markers[markerId].placeholderFinal_path = marker_config["placeholderFinal_path"];
    }
    if (marker_config.contains("placeholderInicial_path")) {
        media_markers[markerId].placeholderInicial_path = marker_config["placeholderInicial_path"];
    }
    if (marker_config.contains("delay")) {
        media_markers[markerId].delay = marker_config["delay"];
    }


    media_markers[markerId].setDetection();
    media_markers[markerId].isUsed = true;
    return true;
}

/**
 * TODO: mudar o nome desta função, para ja fica assim
 * @return
 */
bool imTheOnlyOne() {

    int totalUndetected = 0;

    for (auto &marker : config_markers) {
        if (marker.isUsed) {
            if (!marker.isDetected) {
                totalUndetected++;
            }
        }
    }


    return totalUndetected == 1;
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
        Ptr <cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        aruco::drawMarker(dictionary, idMarker, 200, markerImage, 1);
        if (imwrite("./markers/marker" + std::to_string(idMarker) + ".png", markerImage)) {
            cout << "Marker criado com sucesso. (Verificar diretório markers)" << endl;
        } else {
            cout << "Ocurreu um erro a criar o marker" << endl;
        }
    }

    mainMenu();
}

int beginInteration() {
    Ptr <Dictionary> dictionary = getPredefinedDictionary(DICT_6X6_250);
    Ptr <DetectorParameters> parameters = DetectorParameters::create();


    VideoCapture cap;
    Mat frame;

    int capOpen;

    cout << "A câmara \"0\" normalmente é a câmara pricipal, se estiver a utilizar mais do que uma câmara utilize a 1."
         << endl;
    cout << "Escolher a câmara:";


    cin >> capOpen;
    try {
        cap.open(capOpen);

    } catch (...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }


    while (waitKey(1) < 0) {

        cap >> frame;
        imOut = frame.clone();
        try {
            vector<int> markersIds;
            vector <vector<Point2f>> markerCorners, rejectedCandidates;
            detectMarkers(frame, dictionary, markerCorners, markersIds, parameters);

            Mat outputImage = frame.clone();

            /**
             * colocar todos os markers como não detetados
             * foi a maneira que arranjeo para deixar de detatar markers
             */

            for (auto &marker : media_markers) {
                marker.isDetected = false;
            }

            for (auto &marker : config_markers) {
                marker.isDetected = false;
            }


            /**
             * detação dos markers
             */
            if (!markersIds.empty()) {
                for (int i = 0; i < markersIds.size(); i++) {

                    int markerId = markersIds[i];

                    string markerType = checkMarkerType(markerId);
                    if (markerType == MARKER_TYPE_CONFIG) {
                        if (!config_markers[markerId].isDetected) {
                            bool ok_config = setMarkerConfigProperties(markerId);
                            if (!ok_config) {
                                return -1;
                            }
                        }
                        config_markers[markerId].markerIdPosition = i;
                        config_markers[markerId].markerId = markerId;
                        config_markers[markerId].markerCorners = markerCorners.at(i);
                        config_markers[markerId].lastTimeDetected = time(nullptr);

                    } else if (markerType == MARKER_TYPE_MEDIA) {
                        if (!media_markers[markerId].isDetected) {
                            bool ok_media = setMarkerMediaProperties(markerId);
                            if (!ok_media) {
                                return -1;
                            }
                        }
                        media_markers[markerId].markerIdPosition = i;
                        media_markers[markerId].markerId = markerId;
                        media_markers[markerId].markerCorners = markerCorners.at(i);
                        media_markers[markerId].lastTimeDetected = time(nullptr);
                    } else {
                        return -1;
                    }
                }

                drawDetectedMarkers(outputImage, markerCorners, markersIds);
            }

            for (auto &marker : config_markers) {
                if (marker.isUsed) {
                    if (marker.isDetected) {
                        marker.setImgSrc();

                        Point refPt1, refPt2, refPt3, refPt4;

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
                        computeConfigHomography(marker.markerId, frame);
                    } else {
                        if (imTheOnlyOne() && marker.selectLanguage()) {
                            CURRENT_LANGUAGE = marker.language;
                        }
                    }
                }
            }
            //Point textOrigin(imOut.cols / 4, imOut.rows / 4);

            //putText(imOut, CURRENT_LANGUAGE, textOrigin, 1, 2, GREEN);
            //cout << "CURRENT :" + CURRENT_LANGUAGE << endl;
            //cout << "LAST :" + LAST_LANGUAGE << endl;

            /**
             * homografias dos marcadores do tipo media
             * faz as combinações de dois markers (master e slave)
             */
            for (auto &marker : media_markers) {
                if (!marker.stopPlaying()) {
                    if (marker.type == TYPE_SINGLE) {

                        marker.setImgSrc();


                        Point refPt1, refPt2, refPt3, refPt4;

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
                        computeMediaHomography(marker.markerId, frame);
                    } else if (marker.type == TYPE_COMBINE) {
                        if (marker.function == FUNCTION_MASTER) {

                            int slaveId = marker.slaveId;

                            if (media_markers[slaveId].isDetected) {

                                marker.setImgSrc();


                                Point refPt1, refPt2, refPt3, refPt4;


                                refPt1 = markerCorners.at(marker.markerIdPosition).at(0);//ponto superior esquerdo
                                refPt2.x = markerCorners.at(media_markers[slaveId].markerIdPosition).at(
                                        1).x;//ponto superior direito
                                refPt2.y = markerCorners.at(marker.markerIdPosition).at(1).y;//ponto superior direito



                                refPt3 = markerCorners.at(media_markers[slaveId].markerIdPosition).at(
                                        2); //ponto inferior direito

                                refPt4.x = markerCorners.at(marker.markerIdPosition).at(3).x; //ponto inferior esquerdo
                                refPt4.y = markerCorners.at(media_markers[slaveId].markerIdPosition).at(
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
                                computeMediaHomography(marker.markerId, frame);
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