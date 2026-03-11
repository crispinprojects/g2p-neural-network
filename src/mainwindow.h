/*
 *
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
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QMap>
#include <QStringList>
#include <vector>
#include <QtConcurrent>
#include <QFutureWatcher>
#include "neuralnetwork.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    // This MUST be in the signals section for the "Meta Call" to work
    void progressUpdated(int value);

private slots:
    void on_btnTrain_clicked();
    void on_btnStop_clicked();
    void on_btnPredict_clicked();
    void on_actionExit_triggered();

private:
    Ui::MainWindow *ui;
    NeuralNetwork* nn;

    // State flags
    bool m_isTraining = false;
    bool m_stopTraining = false;
    QFutureWatcher<void> m_watcher;

    // Data structures
    QMap<QChar, int> m_charToIndex;
    QMap<int, QChar> m_indexToChar;
    QStringList m_phonemeList;
    std::vector<std::pair<QString, QStringList>> m_trainingData;
    QMap<QString, int> m_phonemeToIndex;  // Map phonemes to indices

    // Functions
    void setupCharacterMaps();
    void setupPhonemeMap();
    void runTrainingProcess(); // The background worker
    void loadAndTrainFullData();
    QStringList alignPhonemes(const QString& word, const QStringList& phonemes);
    std::vector<double> charContextToVector(const QString& word, int targetIdx);
    std::vector<double> phonemeToVector(const QStringList& phonemes);
    QString getBestPhoneme(const std::vector<double>& output);
    QStringList transcribeWord(const QString& word);
    void inspectCharacterImportance();

};

#endif
