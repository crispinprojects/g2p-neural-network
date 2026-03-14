/*
 * mainWindow.cpp
 * Copyright 2026 Alan Crispin <crispinalan@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GNU Lesser General Public License v2.1
 */

#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QMessageBox>
#include <QTextStream>
#include <cmath>
#include <random>

#include <algorithm> // for std::shuffle
#include <random>    // for std::mt19937

/**
 * @brief MainWindow Constructor. Initializes the UI, Neural Network, and Maps.
 */
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    nn = new NeuralNetwork();
    setupCharacterMaps();
    setupPhonemeMap();

    // 1. Link the Progress Signal to the Progress Bar
    connect(this, &MainWindow::progressUpdated, ui->progressBar, &QProgressBar::setValue);

    // 2. Link the Thread Watcher to the "Cleanup" function
    connect(&m_watcher, &QFutureWatcher<void>::finished, this, [this]() {
        m_isTraining = false;
        ui->btnTrain->setEnabled(true);
        ui->btnStop->setEnabled(false);
        ui->labelResult->setText("Status: Ready / Training Saved");
        qDebug() << "Background thread finished safely.";
    });

    // Auto-load existing weights
    QString weightPath = QCoreApplication::applicationDirPath() + "/g2p_weights.dat";
    if (nn->loadWeights(weightPath.toStdString())) {
        qDebug() << "Loaded pre-trained weights.";
    }
}

MainWindow::~MainWindow() {
    m_stopTraining = true; // Signal thread to stop if we close the app
    m_watcher.waitForFinished(); // Wait for it to die before deleting nn
    delete ui;
    delete nn;
}

/**
 * @brief Maps 'a'-'z' to integers 1-26. 0 is reserved for padding (space).
 */
void MainWindow::setupCharacterMaps() {
    QString chars = "abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < chars.length(); i++) {
        m_charToIndex[chars[i]] = i + 1;
        m_indexToChar[i + 1] = chars[i];
    }
    m_charToIndex[' '] = 0;
    m_indexToChar[0] = ' ';
}

/**
 * @brief Initializes the list of valid CMU phonemes and their vector indices.
 */
void MainWindow::setupPhonemeMap() {

 m_phonemeList = {
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "_"
  };

    for (int i = 0; i < m_phonemeList.size(); i++) {
        m_phonemeToIndex[m_phonemeList[i]] = i;
    }
}

/**
 * @brief encodes a phoneme string into a vector for NN training targets.
 */
std::vector<double> MainWindow::phonemeToVector(const QStringList& phonemes) {
    std::vector<double> result(NUM_PHONEMES, 0.0);
    for (const QString& phoneme : phonemes) {
        if (m_phonemeToIndex.contains(phoneme)) {
            int index = m_phonemeToIndex[phoneme];
            if (index < NUM_PHONEMES) result[index] = 1.0;
        }
    }
    return result;
}

/**
 * @brief SLIDING WINDOW: Extracts a character and its neighbors as an input vector.
 * @param word The full word being processed.
 * @param targetIdx The index of the character currently in focus.
 * @return A vector of normalized values representing the context window.
 */
/**
 * @brief SLIDING WINDOW which represents characters as unique categories.
 * word encoding (ABC): [1,0,0...], [0,1,0...], [0,0,1...] (3 x 26 = 78 numbers)
 *
 */
std::vector<double> MainWindow::charContextToVector(const QString& word, int targetIdx) {
    // FIX: Must be NUM_INPUTS (189), not WINDOW_SIZE (7)
    std::vector<double> vec(NUM_INPUTS, 0.0);
    int halfWindow = WINDOW_SIZE / 2;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int wordPos = targetIdx - halfWindow + i;
        int charIdx = 0; // Default to space

        if (wordPos >= 0 && wordPos < word.length()) {
            QChar c = word[wordPos].toLower();
            charIdx = m_charToIndex.value(c, 0);
        }

        // This calculation goes up to index 188.
        // If vec was size 7, this is where the crash happens.
        int vectorOffset = (i * NUM_CHARS) + charIdx;
        if (vectorOffset < (int)vec.size()) {
            vec[vectorOffset] = 1.0;
        }
    }
    return vec;
}

/**
 * @brief Predicts phonemes for a user-entered word using the sliding window.
 */
void MainWindow::on_btnPredict_clicked()
{
    QString inputWord = ui->lineText->text().toLower();
    if (inputWord.isEmpty()) return;

    QStringList phonemeList = transcribeWord(inputWord);
    qDebug() << "Word =" << inputWord << "Predicted phonemes (word sounds) =" << phonemeList;
    ui->labelResult->setText(phonemeList.join(" "));
}

/**
 * @brief Training loop. Iterates through the dataset for multiple epochs.
 */
void MainWindow::on_btnTrain_clicked() {
    if (m_isTraining) return; // Prevent double-clicks/malloc crashes

    m_isTraining = true;
    m_stopTraining = false;
    ui->btnTrain->setEnabled(false);
    ui->btnStop->setEnabled(true);
    ui->labelResult->setText("Status: Training...");

    // Start the background process
    QFuture<void> future = QtConcurrent::run(&MainWindow::runTrainingProcess, this);
    m_watcher.setFuture(future);

    qDebug() << "Training started in background thread.";
}

/**
 * @brief Stop training loop
 */
void MainWindow::on_btnStop_clicked() {
    m_stopTraining = true;
    ui->btnStop->setEnabled(false);
    ui->labelResult->setText("Status: Stopping...");
}

void MainWindow::runTrainingProcess() {
    // 1. Prepare Data
    m_trainingData.clear();
    loadAndTrainFullData();
    if (m_trainingData.empty()) return;

    std::random_device rd;
    std::mt19937 g(rd());
    int totalEpochs = 500;
    //int totalEpochs = 1000;

    // 2. Training Loop
    for (int epoch = 0; epoch < totalEpochs; epoch++) {
        if (m_stopTraining) break; // Exit requested by user

        double epochError = 0;
        int count = 0;
        std::shuffle(m_trainingData.begin(), m_trainingData.end(), g);

        for (auto& pair : m_trainingData) {
            if (m_stopTraining) break;
            QString word = pair.first;
            QStringList phonemes = pair.second;

            for (int i = 0; i < word.length() && i < phonemes.size(); i++) {
                std::vector<double> input = charContextToVector(word, i);
                std::vector<double> target = phonemeToVector({phonemes[i]});
                nn->train_step(input, target);
                count++;
            }
        }

        // 3. Update Progress (Thread Safe via Signal)
        if (epoch % 5 == 0) {
            int percent = (epoch * 100) / totalEpochs;
            emit progressUpdated(percent);
            qDebug() << "Epoch:" << epoch << "Progress:" << percent << " pecent";
            inspectCharacterImportance();
        }
    }

    // 4. Save results
    QString weightPath = QCoreApplication::applicationDirPath() + "/g2p_weights.dat";
    nn->saveWeights(weightPath.toStdString());
    emit progressUpdated(100);
}

/**
 * @brief Primary prediction loop.
 * For each character in the word, it predicts the sound based on neighbors.
 */
QStringList MainWindow::transcribeWord(const QString& word)
{
    if (word.isEmpty()) return QStringList();
    QStringList fullPhonemeSequence;

    for (int i = 0; i < word.length(); i++) {
        std::vector<double> inputVector = charContextToVector(word, i);
        nn->predict(inputVector);

        QString bestPhoneme = getBestPhoneme(nn->output);

        // Filter out the NULL padding symbol (_) and low-confidence results
        if (bestPhoneme != "_" && !bestPhoneme.isEmpty()) {
            fullPhonemeSequence << bestPhoneme;
        }
    }
    return fullPhonemeSequence;
}

/**
 * @brief Naive Aligner: Ensures training targets match character positions.
 * It "stretches" the phonemes so that 'light' (5 chars) matches 5 phoneme slots.
 */
QStringList MainWindow::alignPhonemes(const QString& word, const QStringList& phonemes) {
    QStringList aligned;
    int wLen = word.length();
    int pLen = phonemes.size();

    for (int i = 0; i < wLen; i++) {
        int pIdx = (i * pLen) / wLen;
        if (i > 0 && pIdx == ((i-1) * pLen) / wLen) {
            aligned << "_"; // Pad with NULL sound if we have more letters than sounds
        } else {
            aligned << phonemes[pIdx];
        }
    }
    return aligned;
}

/**
 * @brief Finds the index of the highest value in the output layer (The "Winner").
 */
QString MainWindow::getBestPhoneme(const std::vector<double>& output) {
    struct PhonemeScore { int index; double score; };
    std::vector<PhonemeScore> scores;

    for (int i = 0; i < (int)output.size(); i++) {
        scores.push_back({i, output[i]});
    }

    // Sort: highest score first
    std::sort(scores.begin(), scores.end(), [](const PhonemeScore& a, const PhonemeScore& b) {
        return a.score > b.score;
    });

    // Logging the runners-up for debugging
    if (scores[0].score < 0.2) {
        qDebug() << "Low confidence! Top 3: "
                 << m_phonemeList[scores[0].index] << "(" << scores[0].score << ") "
                 << m_phonemeList[scores[1].index] << "(" << scores[1].score << ") "
                 << m_phonemeList[scores[2].index] << "(" << scores[2].score << ")";
        return "?";
    }

    double confidence = scores[0].score;

    // We can pass this back to the UI or print it
    qDebug() << "Top Prediction:" << m_phonemeList[scores[0].index]
             << "Confidence:" << QString::number(confidence * 100, 'f', 1) << "%";

    if (confidence < 0.2) return "?";

    return m_phonemeList[scores[0].index];


}

/**
 * @brief Loads and filters CMUdict entries. Skips symbols/numbers.
 * cmudict dataset must be lcoated in working directory
 */
void MainWindow::loadAndTrainFullData() {
    m_trainingData.clear();
    QFile file(QCoreApplication::applicationDirPath() + "/cmudict0.7b");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return;

    QTextStream in(&file);
    QRegularExpression re("^[a-z]+$");

    // Mandatory words which must be included in training
    QStringList mandatoryWords = {"cat", "dog", "wednesday", "hello", "world", "eleventh", "march", "time"};
    std::vector<std::pair<QString, QString>> pool;

    while (!in.atEnd()) {
        QString line = in.readLine();
        if (line.startsWith(";;;")) continue;
        QStringList parts = line.split("  ", Qt::SkipEmptyParts);
        if (parts.size() < 2) continue;
        QString word = parts[0].toLower();
        if (!re.match(word).hasMatch()) continue;
        pool.push_back({word, parts[1]});
    }
    file.close();

    // 1. Add Mandatory Words first
    for (const QString& target : mandatoryWords) {
        for (auto& entry : pool) {
            if (entry.first == target) {
                QStringList aligned = alignPhonemes(entry.first, entry.second.split(" ", Qt::SkipEmptyParts));
                m_trainingData.push_back({entry.first, aligned});
                break;
            }
        }
    }

    // 2. Add random words until we hit 100
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(pool.begin(), pool.end(), g);

    for (auto& entry : pool) {

        if (m_trainingData.size() >= 10000) break;
        // Don't add duplicates of mandatory words
        if (mandatoryWords.contains(entry.first)) continue;

        QStringList aligned = alignPhonemes(entry.first, entry.second.split(" ", Qt::SkipEmptyParts));
        m_trainingData.push_back({entry.first, aligned});
    }

     qDebug() << "Loaded" << m_trainingData.size() << "manadory and diverse random words.";

}

void MainWindow::on_actionExit_triggered()
{
    QApplication::quit();
}

void MainWindow::inspectCharacterImportance()
{
    QMap<QChar, double> characterPower;

    // Iterate through each character in our map
    for (auto it = m_charToIndex.begin(); it != m_charToIndex.end(); ++it) {
        QChar c = it.key();
        int charIdx = it.value();
        double totalWeightMagnitude = 0.0;

        // Sum weights for this character across all 7 window positions
        for (int w = 0; w < WINDOW_SIZE; w++) {
            int inputIdx = (w * NUM_CHARS) + charIdx;

            // Look at all connections from this input to the hidden layer
            for (int h = 0; h < NUM_HIDDEN; h++) {
                totalWeightMagnitude += std::abs(nn->weights_ih[inputIdx][h]);
            }
        }
        characterPower[c] = totalWeightMagnitude;
    }

    // Print the "Most Important" characters learned so far
    qDebug() << "--- Character Knowledge Map ---";
    for (auto c : characterPower.keys()) {
        if (characterPower[c] > 0.1) { // Only show characters the NN has "noticed"
            qDebug() << QString("Char [%1]: Influence Factor %2").arg(c).arg(characterPower[c]);
        }
    }
}





