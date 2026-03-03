#include <opencv2/opencv.hpp>
#include <vector>

extern "C" {
    cv::Mat old_gray;
    std::vector<cv::Point2f> p0;

    float process_frame(int width, int height, unsigned char* data) {
        cv::Mat frame(height, width, CV_8UC3, data);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        float movement = 0.0f;

        // Étape 1 : Si on n'a pas de points ou pas assez, on en cherche
        if (p0.size() < 20 || old_gray.empty()) {
            // Détecte les coins les plus nets pour les suivre
            cv::goodFeaturesToTrack(gray, p0, 100, 0.01, 10);
            gray.copyTo(old_gray);
            return 0.0f; // On attend le prochain frame pour comparer
        }

        // Étape 2 : Calculer le mouvement
        std::vector<cv::Point2f> p1;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(old_gray, gray, p0, p1, status, err);

        int count = 0;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                movement += cv::norm(p1[i] - p0[i]);
                count++;
            }
        }

        if (count > 0) {
            movement /= count;
            p0 = p1; // On garde les points valides pour le prochain coup
        } else {
            // Si on a tout perdu, on vide pour forcer la redétection au prochain frame
            p0.clear();
        }

        gray.copyTo(old_gray);
        return movement;
    }
}