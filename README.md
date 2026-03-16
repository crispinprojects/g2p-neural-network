# G2P Neural Network

A custom **Grapheme-to-Phoneme (G2P)** engine written in C++ and Qt. This project uses a neural network to learn how to "pronounce" English words by converting strings of characters (graphemes) into phonetic sounds (phonemes).

The work in this repository explores if it could be possible to replace the G2P engine used in the [Talk  Calendar](https://github.com/crispinprojects/talkcalendar) project with a neural network. Currently Talk Calendar G2P engine uses decision trees with many lines of if/else logic. The idea is to see if the Talk Calendar G2P decision trees (hand-crafted rules for English pronunciation) could be replaced with a trained neural network loading a single file of trained weights and biases that represents the "essence" of English pronunciation. Training uses the Carnegie Mellon University Pronouncing Dictionary (CMUdict).

A screenshot of the G2P Neural Network applcation is shown below.

![](g2p-neuralnet.png)


##  The Approach: Sliding Window + FFNN

Instead of trying to feed an entire word of varying length into a fixed-size network, this project uses a **Sliding Window** technique combined with a **Feed-Forward Neural Network (FFNN)**.

### 1. The Sliding Window

Human speech is contextual. The letter 'c' sounds like /k/ in "cat" but /s/ in "city." To capture this, the network looks at a "window" of 7 characters at a time:

* **3 Letters of Left Context**
* **1 Target Letter** (the one we are predicting for)
* **3 Letters of Right Context**

As the "window" slides across the word, the network predicts the most likely sound for the target letter based on its neighbours.

The sliding window approach was used in early speech synthesis systems such as the [NETtalk]( https://everything.explained.today/NETtalk_(artificial_neural_network)/) project. 

### 2. The Neural Architecture (FFNN)

A simple word encoding scheme is used with letters represented as shown below.

```
A =[1,0,0...]
B =[0,1,0...]
C= [0,0,1...]
```

* **Input Layer:** 189 neurons (7 positions × 27 possible characters).
* **Hidden Layer:** 200 neurons with Sigmoid activation.
* **Output Layer:** 45 neurons (representing the 45 standard English phonemes).
* **Learning:** Backpropagation with **Momentum** to help the network "roll" over local mathematical dips and find the global best solution.

### 3. Training  & Optimization

#### The Training Pipeline

**Diverse Sampling:** The system shuffles the CMUdict to ensure a balanced representation of the entire alphabet (A-Z) in every batch.

**Thread Management:** Utilising QtConcurrent, the  mathematical calculations are decoupled from the Main UI thread, ensuring the application remains responsive during long training runs.

**Persistence:** Weights and biases are serialized to g2p_weights.dat using binary I/O for instant loading in future sessions.
    
#### Optimization: Momentum

To improve training stability and speed, this project implements Momentum (β=0.9) alongside standard Backpropagation.

**Accelerated Convergence:** In areas where the gradient (slope) is consistent, momentum builds up "velocity," allowing the network to reach lower error rates in fewer epochs.

**Escaping Local Minima:** Standard gradient descent can get "stuck" in suboptimal mathematical valleys. Momentum provides numerical inertia, helping the weights escape local minima to find the best global solution.

**Reduced Oscillation:** Language data is inherently noisy. Momentum acts as a filter, smoothing out erratic weight updates and focusing the learning on primary phonetic patterns.

---

## Neural Network Comparison Table

| Type | Description | Best Use Case |
| --- | --- | --- |
| **FFNN** (Feed-Forward) | Data moves in one direction. Simple and fast. Used here with a Sliding Window. | Pattern recognition, basic classification, and "NETtalk" style G2P. |
| **RNN** (Recurrent) | Has "internal loops" that act as a short-term memory of previous inputs. | Time-series data, speech recognition, and processing long sentences. |
| **LSTM** (Long Short-Term) | A smarter RNN that can remember information for long periods of time. | Complex translation and high-end text-to-speech engines. |
| **Transformer** | Uses "Attention" to look at all parts of a sequence simultaneously rather than in order. | State-of-the-art AI like ChatGPT, Google Translate, and modern G2P. |

---

##  The Dataset: CMUdict

In the context of AI and linguistics, a corpus is simply a large, structured collection of text or recorded speech used for statistical analysis or training models. This project utilizes the **Carnegie Mellon University Pronouncing Dictionary (CMUdict)** which is a pronunciation corpus. The dictionary dataset (cmudict-0.7b) can be downloaded from [here](http://www.speech.cs.cmu.edu/cgi-bin/cmudict).

* **What it is:** A free, machine-readable dictionary containing over 134,000 words and their transcriptions.
* **Why it is used:** It provides the "Ground Truth." In AI "Ground Truth"" refers to information that is known to be real or true, obtained through direct observation and measurement, rather than inference. By showing the neural network thousands of examples from this file, in theory it should eventually learn the statistical rules of English pronunciation.
* **Phoneme Set:** It uses the **Arpabet**, which represents sounds using 2-letter codes (e.g., `AE1` for the "a" in "cat"). ARPABET is a set of phonetic transcription codes developed by Advanced Research Projects Agency as a part of their Speech Understanding Research project in the 1970s. 

---

##  Results

### Training with 10,000 Words

Results for training with 10,000 words from the cmudict file and epochs set to 500 are shown below.  Rather that just load 10,000 diverse random words I added some mandatory words so that I could investigate the predictions for these words and other similar words.
```
QStringList mandatoryWords = {"cat", "dog", "wednesday", "hello", "world", "eleventh", "march", "time"};
```
I added all the [phoneme symbols](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols) used by the CMU dictionary pronunciation corpus.

Results were mixed. All phonemes for the word dog were predicted successfully.
```
Top Prediction: "D" Confidence: "100.0" %
Top Prediction: "AA1" Confidence: "58.4" %
Top Prediction: "G" Confidence: "97.8" %
Word = "dog" Predicted phonemes (word sounds) = QList("D", "AA1", "G")
```
I was expecting D AO1 G but the result is very close.

The phoneme predictions for the word cat are shown below.

```
Low confidence! Top 3:  "CH" ( 0.00606208 )  "AH2" ( 0.00279973 )  "HH" ( 0.00180787 )
Top Prediction: "AE1" Confidence: "97.6" %
Low confidence! Top 3:  "EY0" ( 0.00713005 )  "AW1" ( 0.00332639 )  "AW0" ( 0.00303057 )
Word = "cat" Predicted phonemes (word sounds) = QList("?", "AE1", "?")
```
The neural network does not have enough confidence to predict the first and last phonemes as the phonemes for cat which should be K AE1 T. The "?" indicates an unknown phoneme.

The prediction for the word hello is also missing phonemes.

```
Top Prediction: "HH" Confidence: "99.7" %
Low confidence! Top 3:  "EH1" ( 0.151192 )  "EH0" ( 0.0279087 )  "EH2" ( 0.0198296 )
Top Prediction: "EH1" Confidence: "87.6" %
Low confidence! Top 3:  "AH0" ( 0.0065182 )  "EH0" ( 0.00462847 )  "EH2" ( 0.00452076 )
Low confidence! Top 3:  "AO2" ( 0.0126723 )  "AH0" ( 0.00462215 )  "AO0" ( 0.0037352 )
Word = "hello" Predicted phonemes (word sounds) = QList("HH", "?", "EH1", "?", "?")
```
The prediction for the word birthday was:
```
Top Prediction: "B" Confidence: "99.8" %
Low confidence! Top 3:  "B" ( 0.0129889 )  "AA0" ( 0.00142518 )  "EH2" ( 0.000996788 )
Top Prediction: "ER1" Confidence: "94.5" %
Low confidence! Top 3:  "AE1" ( 0.0101408 )  "AA1" ( 0.00552619 )  "AH1" ( 0.0017942 )
Low confidence! Top 3:  "EH1" ( 0.0308226 )  "EH2" ( 0.00508963 )  "AW2" ( 0.00319251 )
Top Prediction: "D" Confidence: "96.6" %
Top Prediction: "D" Confidence: "54.0" %
Top Prediction: "EY2" Confidence: "51.1" %
Word = "birthday" Predicted phonemes (word sounds) = QList("B", "?", "ER1", "?", "?", "D", "D", "EY2")
```
The "day" part of the word birthday is predicted but there are missing phonemes.


### Training With More Words

The [NetTalk](https://en.wikipedia.org/wiki/NETtalk_(artificial_neural_network)) artificial neural network for speech synthesis developed in the 1980s used 20,000 words in its dataset (a different pronunciation corpus to CMUdict). Training my neural network with 20,000 word would take many hours on my home computer with a moderate processor. Although 20,000 words would give the network a much larger "corpus" to learn from, allowing it to see and understand more character combinations it would not see all patterns and I expect training with  all 134,000 words in the CMUdict pronunciation corpus is required which is not practical on my home computer due to processor limitations.  A more powerful system is needed and parallel (batch) training would most likely be required. The other thing that could be tried would be a reduced phoneme set.

Currently, I use QtConcurrent::run() to offload the entire training process to one background thread. This makes the GUI "concurrent" so that is does not freeze. Training is not split across multiple cores and so is not done in parallel. To use more cores parallel training (batch training) code would be need to be implemented which would require calculating weights and gradients simultaneously on different cores and then average them. The problem with this is that Neural Networks usually need to update weights and gradients sequentially (each step depends on the last) meaning that standard backpropagation is naturally a single-threaded task. For a project of this scale, sticking to one thread is safer and prevents complex memory "race conditions" which are an inherent problem when implementing parallel processing.

### Results Summary

Training with 10,000 diverse random words is not enough to successfully transcribe many common words like "birthday" as the the neural network needs see more word patterns to predict the phonemes. Training runs can take many hours on my home computer system and so increasing the number of words to the full CMUdict word count (134,000 words) is not practical.  Also I am not certain if a trained neural network would capture all English word pronunciations

The decision tree G2P algorithm currently used with [Talk Calendar](https://github.com/crispinprojects/talkcalendar) is a better method than the sliding window neural network approach described here. With words like "birthday" the decision tree uses rules specifically written to transcribe it. Even when the decision tree was trained with all 134,000 words in the CMUdict pronunciation corpus I still had to modify by hand the decision tree rules to get better word pronunciations. 

##  How to Compile and Run

### Prerequisites

* **Qt Framework 6.x** (or 5.15+)
* **C++17 Compiler** (GCC, Clang, or MSVC)
* **CMake** (recommended) or **qmake**

### Build Instructions

The project has been developed using Debian Trixie.

1. **Clone/Open:** Open the project file (`.pro` or `CMakeLists.txt`) in **Qt Creator**.
2. **Clean & Rebuild:** Select `Build > Clean All`, then `Build > Run CMake`, followed by `Build > Build Project`.
3. **Data Placement:** Ensure the `cmudict0.7b` file is located in the **Application Output folder** (where the binary is generated).
4. **Run:** Hit the "Run" button in Qt Creator.

### Training the Network

1. Ensure the `cmudict0.7b` file is located in the application output folder.
2. Click **"Train"**. The application will launch a background thread to prevent the UI from freezing.
3. Monitor the **Progress Bar**. Once complete, weights are automatically saved to `g2p_weights.dat`.

---

## Future Improvements

In terms of the current sliding neural network window approach swapping Sigmoid for ReLU (Rectified Linear Unit) to combat vanishing gradient could be used. However, some  initial testing of using ReLU activation proved not to help much. Adding a second hidden layer to capture more complex linguistic nuances might help improve predictions. Shuffling the 134,000 words in the CMUdict to ensure the network learns more of the corpus could be helpful but training with all 134,000 words is likely to be necessary which is not practical with my current hardware.

In state-of-the-art AI more advanced types of neural networks such as RNNs and Transformers are used for translations and G2P. 

RNNs (Recurrent Neural Networks) are significantly more complex to code from scratch than the feed forward neural network used in this project because you have to manage "hidden states" that persist over time. There is C++ example of a simple RNN network on GitHub [here](https://github.com/mIwav/simple-rnn).

Transformers (like Gemini or GPT) use sequential memory for context. With transformers instead of reading character-by-character and trying to remember the past, a transformer looks at every character in the sequence simultaneously and uses "attention weights" to decide which other characters are most relevant to the one it is currently processing. Programming a transformer neural network in C++ is challenging due to the complexity of the architecture and attention mechanisms. There is more information [here](https://www.youtube.com/watch?v=Nw_PJdmydZY). For the hobby project Talk Calendar using transformers is not necessary to achieve G2P.

---
 

## Weblinks

[CMUdict Corpus](https://github.com/Alexir/CMUdict/blob/master/cmudict-0.7b)
A working version of cmudict, maintained by Alexander Rudnicky 

[CMU Phoneme Symbols](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols)

[NetTalk](https://en.wikipedia.org/wiki/NETtalk_(artificial_neural_network))
NETtalk artificial neural network developed in the 1980s that learns to pronounce written English text. The program was trained on English words and their corresponding pronunciations. It was able to accurately generate pronunciations for unseen words. 

[Nettalk Corpus](https://archive.ics.uci.edu/dataset/150/connectionist+bench+nettalk+corpus)




