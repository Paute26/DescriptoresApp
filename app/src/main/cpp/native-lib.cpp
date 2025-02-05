#include <locale>
#include <jni.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <android/log.h>


#define LOG_TAG "NDK_LOG"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


using namespace cv;
using namespace std;

//-------------------------------MOMENTOS HU-----------------------------------------------
Mat preprocessImage(const Mat& image) {
    Mat edges, dilatedEdges, filledImage;

    // Procesamiento de bordes con Canny
    Canny(image, edges, 50, 150);

    // Dilatar los bordes para hacerlos más visibles
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(edges, dilatedEdges, kernel);

    // Encontrar contornos en la imagen dilatada
    vector<vector<Point>> contours;
    findContours(dilatedEdges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Crear una imagen vacía para dibujar los contornos
    filledImage = Mat::zeros(image.size(), CV_8UC1);
    drawContours(filledImage, contours, -1, Scalar(255), FILLED);

    return filledImage;
}

vector<double> calculateHuMoments(const cv::Mat& image) {
    // Calcular los momentos de la imagen
    Mat binaryImage;
    threshold(image, binaryImage, 128, 255, THRESH_BINARY);
    cv::Moments moments = cv::moments(binaryImage);
    std::vector<double> huMoments(7);

    // Calcular los Momentos Hu
    cv::HuMoments(moments, huMoments.data());

    // Agregar log para cada momento Hu
    LOGI("Momento Hu 1: %f", huMoments[0]);
    LOGI("Momento Hu 2: %f", huMoments[1]);
    LOGI("Momento Hu 3: %f", huMoments[2]);
    LOGI("Momento Hu 4: %f", huMoments[3]);
    LOGI("Momento Hu 5: %f", huMoments[4]);
    LOGI("Momento Hu 6: %f", huMoments[5]);
    LOGI("Momento Hu 7: %f", huMoments[6]);

    return huMoments;
}

void normalizarMomentosHu(vector<double>& hu) {
    for (int i = 0; i < 7; i++) {
        if (hu[i] != 0)
            hu[i] = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));
        else
            hu[i] = 0;
    }
}

#include <vector>
#include <cmath>  // Para std::sqrt y std::accumulate

vector<double> standardizeHuMoments(const vector<double>& huMoments) {
    // Calcular la media de los momentos Hu de la imagen
    double mean = 0.0;
    for (size_t i = 0; i < huMoments.size(); i++) {
        mean += huMoments[i];
    }
    mean /= huMoments.size();

    // Calcular la desviación estándar de los momentos Hu de la imagen
    double stdDev = 0.0;
    for (size_t i = 0; i < huMoments.size(); i++) {
        stdDev += std::pow(huMoments[i] - mean, 2);
    }
    stdDev = std::sqrt(stdDev / huMoments.size());

    // Estandarizar los Momentos Hu de la nueva imagen
    vector<double> standardizedHuMoments(huMoments.size());
    for (size_t i = 0; i < huMoments.size(); i++) {
        // Estandarizar: (valor - media) / desviación estándar
        if (stdDev != 0) {  // Para evitar división por cero
            standardizedHuMoments[i] = (huMoments[i] - mean) / stdDev;
        } else {
            standardizedHuMoments[i] = huMoments[i] - mean;  // Si la desviación estándar es 0, solo restamos la media
        }
    }

    return standardizedHuMoments;
}

double euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

void loadTrainedHuMoments(const string& csvContent, vector<vector<double>>& trainedHuMoments, vector<string>& trainedLabels) {
    setlocale(LC_NUMERIC, "C");  // Asegura el formato correcto de números
    stringstream ss(csvContent);
    string line;
    getline(ss, line); // Saltar la cabecera
    LOGI("%s", "---csv_HU---");

    int lineNumber = 1;  // Para rastrear el número de línea
    while (getline(ss, line)) {
        stringstream lineStream(line);
        vector<double> huMoments;
        string value, label;
        // Configura el locale para asegurar el formato correcto
        setlocale(LC_ALL, "C");

        // Lee los primeros 7 valores de la línea (los momentos de Hu)
        for (int i = 0; i < 7; i++) {
            getline(lineStream, value, ',');
            //LOGI("Valor leido (antes de convertir a double): %s", value.c_str());  // Log de depuración
            huMoments.push_back(stod(value));  // Convierte el valor a double
            //LOGI("Momento Hu %d: %f", i + 1, huMoments.back());  // Imprime el valor del momento Hu

        }
        getline(lineStream, label, ',');
        //LOGI("Categoría: %s", label.c_str());  // Imprime la categoría

        // Agrega los momentos de Hu y la categoría a sus respectivos vectores
        trainedHuMoments.push_back(huMoments);
        trainedLabels.push_back(label);

        // Verifica que los datos se han agregado correctamente
        //LOGI("Línea %d leída: %d momentos Hu y categoría '%s'", lineNumber, huMoments.size(), label.c_str());

        lineNumber++;
    }
    //LOGI("Carga de momentos de Hu completada. Total de líneas procesadas: %d", lineNumber - 1);
}

string classifyImage(const Mat& newImage, const vector<vector<double>>& trainedHuMoments, const vector<string>& trainedLabels) {
    Mat processedImage = preprocessImage(newImage);
    vector<double> newHuMoments = calculateHuMoments(processedImage);
    normalizarMomentosHu(newHuMoments);
    vector<double> standardizedHuMoments = standardizeHuMoments(newHuMoments);

    // Normalizar los Momentos Hu de la nueva imagen
    //vector<double> normalizedNewHuMoments = normalizeHuMoments(newHuMoments, trainedHuMoments);

    // Imprimir los Momentos Hu de la nueva imagen normalizados
    string newHuMessage = "Momentos Hu normalizados de la nueva imagen: ";
    for (double hu : standardizedHuMoments) {
        newHuMessage += to_string(hu) + " ";
    }
    LOGI("%s", newHuMessage.c_str());

    double bestDistance = numeric_limits<double>::max();
    string predictedCategory;

    for (size_t i = 0; i < trainedHuMoments.size(); i++) {
        // Imprimir los Momentos Hu de los entrenados
        string trainedHuMessage = "Momentos Hu entrenados para la categoría " + trainedLabels[i] + ": ";
        for (double hu : trainedHuMoments[i]) {
            trainedHuMessage += to_string(hu) + " ";
        }
        LOGI("%s", trainedHuMessage.c_str());

        // Calcular la distancia euclidiana entre los momentos Hu normalizados
        double distance = euclideanDistance(standardizedHuMoments, trainedHuMoments[i]);

        // Imprimir la distancia calculada
        LOGI("Distancia Euclidiana con la categoría %s: %f", trainedLabels[i].c_str(), distance);

        if (distance < bestDistance) {
            bestDistance = distance;
            predictedCategory = trainedLabels[i];
        }
    }

    // Imprimir la categoría predicha
    LOGI("Categoría predicha: %s", predictedCategory.c_str());
    return predictedCategory;
}

//------------------------------------MOMENTOS DE ZERNIKEN----------------------------------------------
#define PI 3.14159265358979323846
#define MAX_L 20

void mb_zernike2D(const Mat& img, int order, double rad, vector<double>& zvalues) {
    int L, N, D;
    int cols = img.cols;
    int rows = img.rows;

    // Determinar el tamaño de la imagen
    N = min(cols, rows);
    L = (order > 0) ? order : 15;
    if (L >= MAX_L) L = MAX_L - 1; // Evitar desbordamiento

    if (!(rad > 0.0)) rad = N;
    D = static_cast<int>(rad * 2);

    static double H1[MAX_L][MAX_L] = { 0 };
    static double H2[MAX_L][MAX_L] = { 0 };
    static double H3[MAX_L][MAX_L] = { 0 };
    static bool init = true;

    double COST[MAX_L] = { 0 }, SINT[MAX_L] = { 0 }, R[MAX_L] = { 0 };

    double AR[MAX_L][MAX_L] = { 0 }, AI[MAX_L][MAX_L] = { 0 };

    // Cálculo de momentos de orden 0 y 1 para centrar el círculo unitario
    double moment10 = 0.0, moment00 = 0.0, moment01 = 0.0, intensity, sum = 0.0;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            intensity = static_cast<double>(img.at<uchar>(j, i));
            sum += intensity;
            moment10 += (i + 1) * intensity;
            moment00 += intensity;
            moment01 += (j + 1) * intensity;
        }
    }

    if (moment00 == 0) return; // Evitar división por cero

    double m10_m00 = moment10 / moment00;
    double m01_m00 = moment01 / moment00;

    // Inicialización de coeficientes H1, H2, H3
    if (init) {
        for (int n = 0; n < MAX_L; n++) {
            for (int m = 0; m <= n; m++) {
                if (n != m) {
                    H3[n][m] = -4.0 * (m + 2.0) * (m + 1.0) / ((n + m + 2.0) * (n - m));
                    H2[n][m] = (H3[n][m] * (n + m + 4.0) * (n - m - 2.0)) / (4.0 * (m + 3.0)) + (m + 2.0);
                    H1[n][m] = ((m + 4.0) * (m + 3.0)) / 2.0 - (m + 4.0) * H2[n][m] +
                               (H3[n][m] * (n + m + 6.0) * (n - m - 4.0)) / 8.0;
                }
            }
        }
        init = false;
    }

    double area = PI * rad * rad;

    for (int i = 0; i < cols; i++) {
        double x = (i + 1 - m10_m00) / rad;
        for (int j = 0; j < rows; j++) {
            double y = (j + 1 - m01_m00) / rad;
            double r2 = x * x + y * y;
            double r = sqrt(r2);
            if (r < DBL_EPSILON || r > 1.0) continue;

            R[0] = 1;
            for (int n = 1; n <= L; n++) R[n] = r * R[n - 1];

            COST[0] = x / r;
            SINT[0] = y / r;
            for (int m = 1; m <= L; m++) {
                COST[m] = x * COST[m - 1] - y * SINT[m - 1];
                SINT[m] = x * SINT[m - 1] + y * COST[m - 1];
            }

            double f = img.at<uchar>(j, i) / sum;

            double Rnm2 = 0, Rnm, Rnmp2 = 0, Rnmp4 = 0;
            for (int n = 0; n <= L; n++) {
                double const_t = (n + 1) * f / PI;
                double Rn = R[n];
                if (n >= 2) Rnm2 = R[n - 2];

                for (int m = n; m >= 0; m -= 2) {
                    if (m == n) {
                        Rnm = Rn;
                        Rnmp4 = Rn;
                    }
                    else if (m == n - 2) {
                        Rnm = n * Rn - (n - 1) * Rnm2;
                        Rnmp2 = Rnm;
                    }
                    else {
                        Rnm = H1[n][m] * Rnmp4 + (H2[n][m] + (H3[n][m] / r2)) * Rnmp2;
                        Rnmp4 = Rnmp2;
                        Rnmp2 = Rnm;
                    }
                    AR[n][m] += const_t * Rnm * COST[m];
                    AI[n][m] -= const_t * Rnm * SINT[m];
                }
            }
        }
    }

    // Cálculo final de momentos de Zernike
    zvalues.clear();
    for (int n = 0; n <= L; n++) {
        for (int m = 0; m <= n; m++) {
            if ((n - m) % 2 == 0) {
                double magnitude = sqrt(AR[n][m] * AR[n][m] + AI[n][m] * AI[n][m]);
                zvalues.push_back(fabs(magnitude));
            }
        }
    }
    // 📌 Imprimir los Momentos de Zernike calculados
    LOGI("Momentos de Zernike de la imagen:");
    for (size_t i = 0; i < zvalues.size(); i++) {
        LOGI("Zernike[%lu]: %f", i, zvalues[i]);
    }
}

// Cargar el dataset
vector<pair<vector<double>, string>> loadDataset(const string& csvContent) {
    stringstream ss(csvContent);
    string line;
    getline(ss, line); // Saltar la cabecera
    LOGI("%s", "---csv_ZERNIKE---");
    size_t maxRowsToLog = 5;  // Número máximo de filas a imprimir en Logcat

    vector<pair<vector<double>, string>> dataset;

    while (getline(ss, line)) {
        stringstream lineStream(line);
        vector<double> features;
        string value, label;
        vector<string> tokens;
        while (getline(lineStream, value, ',')) {
            tokens.push_back(value);
        }

        if (!tokens.empty()) {
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                features.push_back(stod(tokens[i]));
            }
            label = tokens.back();
            dataset.push_back({features, label});
        }
    }
    // 📌 Imprimir solo las primeras 5 filas
    LOGI("Dataset cargado con %lu filas. Mostrando las primeras %lu filas:", dataset.size(), min(dataset.size(), maxRowsToLog));
    for (size_t i = 0; i < min(dataset.size(), maxRowsToLog); i++) {
        string row = "Fila " + to_string(i + 1) + ": ";
        for (double feature : dataset[i].first) {
            row += to_string(feature) + " ";
        }
        row += "-> " + dataset[i].second; // Etiqueta
        LOGI("%s", row.c_str());
    }
    return dataset;
}

// Clasificar una nueva imagen
string classifyImageZERNIKE(const Mat& img, const vector<pair<vector<double>, string>>& dataset) {
    vector<double> zernikeValues;
    mb_zernike2D(img, 10, 50, zernikeValues);

    double minDist = DBL_MAX;
    string bestLabel;
    vector<double> closestZernike;

    LOGI("Clasificación con Momentos de Zernike:");
    for (const auto& entry : dataset) {
        double dist = euclideanDistance(zernikeValues, entry.first);

        LOGI("Comparando con clase '%s' - Distancia: %f", entry.second.c_str(), dist);

        if (dist < minDist) {
            minDist = dist;
            bestLabel = entry.second;
            closestZernike = entry.first;
        }
    }

    // 📌 Mostrar la mejor coincidencia
    LOGI("Clase asignada: %s", bestLabel.c_str());
    LOGI("Distancia mínima: %f", minDist);

    // 📌 Mostrar los Momentos de Zernike de la mejor coincidencia
    LOGI("Momentos de Zernike más cercanos en el dataset:");
    for (size_t i = 0; i < closestZernike.size(); i++) {
        LOGI("Zernike[%lu]: %f", i, closestZernike[i]);
    }

    return bestLabel;
}


//----------------------------------APLICACION-------------------------------------------------------
// Función JNI para clasificar la imagen usando los Momentos Hu
extern "C" JNIEXPORT jstring    JNICALL
Java_epautec_atlas_descriptoresapp_MainActivity_MomentsHU(
        JNIEnv *env,
        jobject /* this */,
        jbyteArray imageData,  // Imagen recibida en formato byte[]
        jstring csvContent) {  // Contenido del archivo CSV como String

    // Convertir el array de bytes de la imagen a un Mat
    jbyte* buffer = env->GetByteArrayElements(imageData, nullptr);
    jsize length = env->GetArrayLength(imageData);

    if (buffer == nullptr) {
        return env->NewStringUTF("Error al recibir la imagen");
    }

    // Crear una Mat a partir del array de bytes
    std::vector<uchar> bufferVec(buffer, buffer + length);
    env->ReleaseByteArrayElements(imageData, buffer, JNI_ABORT);

    // Decodificar la imagen a cv::Mat (en formato color o escala de grises según sea necesario)
    cv::Mat imageMat = cv::imdecode(bufferVec, cv::IMREAD_GRAYSCALE);
    if (imageMat.empty()) {
        return env->NewStringUTF("Error al decodificar la imagen");
    }
    //Mat imageMat = imdecode(Mat(1, length, CV_8UC1, buffer), IMREAD_GRAYSCALE);

    // Obtener el contenido del archivo CSV
    const char *csvChars = env->GetStringUTFChars(csvContent, nullptr);
    std::string csvString(csvChars);
    env->ReleaseStringUTFChars(csvContent, csvChars);

    // Cargar los momentos de Hu entrenados desde el contenido CSV
    vector<vector<double>> trainedHuMoments;
    vector<string> trainedLabels;
    loadTrainedHuMoments(csvString, trainedHuMoments, trainedLabels);

    // Clasificar la imagen
    string category = classifyImage(imageMat, trainedHuMoments, trainedLabels);

    // Liberar memoria
    env->ReleaseByteArrayElements(imageData, buffer, JNI_ABORT);
    env->ReleaseStringUTFChars(csvContent, csvChars);

    // Retornar la categoría como un string
    return env->NewStringUTF(category.c_str());
}


extern "C"
JNIEXPORT jstring JNICALL
Java_epautec_atlas_descriptoresapp_MainActivity_MomentsZernike(
        JNIEnv *env, jobject, jbyteArray imageBytes, jstring csvContent) {
    // Convertir jbyteArray a vector de bytes
    jsize length = env->GetArrayLength(imageBytes);
    jbyte* bytes = env->GetByteArrayElements(imageBytes, nullptr);
    vector<uchar> imgData(bytes, bytes + length);
    Mat img = imdecode(imgData, IMREAD_GRAYSCALE);  // Decodificar la imagen
    if (img.empty()) {
        LOGI("Error: No se pudo decodificar la imagen.");
        return env->NewStringUTF("Error: Imagen no válida");
    }

    // Liberar el array de bytes
    env->ReleaseByteArrayElements(imageBytes, bytes, 0);

    // Procesar el CSV (convertirlo en un vector de pares de características y etiquetas)
    const char *csvStr = env->GetStringUTFChars(csvContent, nullptr);
    stringstream ss(csvStr);
    string line;
    vector<pair<vector<double>, string>> dataset = loadDataset(csvStr);

    // Liberar memoria del CSV
    env->ReleaseStringUTFChars(csvContent, csvStr);

    // Procesar la imagen con Momentos Zernike y clasificar ----------------------------------------
    Mat processedImage = preprocessImage(img);
    string predictedLabel = classifyImageZERNIKE(img, dataset);

    // Devolver la categoría predicha
    return env->NewStringUTF(predictedLabel.c_str());
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_epautec_atlas_descriptoresapp_SecondActivity_processImageNative(
        JNIEnv *env,
        jobject thiz,
        jbyteArray imageData) {
    // Obtener datos de la imagen
    jsize dataSize = env->GetArrayLength(imageData);
    if (dataSize == 0) {
        LOGI("Error: imageData está vacío");
        return env->NewByteArray(0);
    }
    LOGI("Tamaño de imageData: %d", dataSize);

    jbyte *imageBytes = env->GetByteArrayElements(imageData, nullptr);
    std::vector<uchar> buffer(imageBytes, imageBytes + dataSize);
    env->ReleaseByteArrayElements(imageData, imageBytes, JNI_ABORT);

    // Decodificar la imagen a cv::Mat
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        LOGI("Error: No se pudo decodificar la imagen");
        return env->NewByteArray(0);
    }
    LOGI("Imagen decodificada correctamente. Tamaño: %d x %d", image.cols, image.rows);

    // (Aquí puedes aplicar procesamiento con OpenCV si es necesario)
    //cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);  // Ejemplo de procesamiento
    Mat processedImage = preprocessImage(image);

    // Codificar la imagen procesada a PNG
    std::vector<uchar> processedBuffer;
    bool success = cv::imencode(".png", processedImage, processedBuffer);
    if (!success || processedBuffer.empty()) {
        LOGI("Error: No se pudo codificar la imagen procesada");
        return env->NewByteArray(0);
    }
    LOGI("Imagen codificada correctamente. Tamaño del buffer: %lu", processedBuffer.size());

    // Convertir std::vector<uchar> a jbyteArray para devolver a Java
    jbyteArray processedData = env->NewByteArray(processedBuffer.size());
    env->SetByteArrayRegion(processedData, 0, processedBuffer.size(), (jbyte*)processedBuffer.data());

    return processedData;
}

