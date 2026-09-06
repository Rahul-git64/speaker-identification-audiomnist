# speaker-identification-audiomnist
60-class speaker identification on AudioMNIST using MFCC features — benchmarking SVM, KNN, Random Forest, CNN, and LSTM (best: 99.85% accuracy).
# Speaker Identification from Audio (AudioMNIST)

A 60-class speaker identification system built on the AudioMNIST dataset, using MFCC features and comparing classical ML and deep learning approaches.

## Dataset
[AudioMNIST](https://github.com/soerenab/AudioMNIST) — 30,000 spoken digit recordings from 60 speakers. Labels used here are **speaker identity**, not the spoken digit.

## Approach
- Extracted MFCC features from raw audio
- Train/test split: 24,000 / 6,000 samples
- Compared five models: SVM (RBF), KNN, Random Forest, a CNN (on MFCC spectrograms), and an LSTM (on MFCC sequences)

## Results

| Model | Accuracy |
|---|---|
| **SVM (RBF)** | **99.85%** |
| KNN | 99.35% |
| CNN | 99.18% |
| Random Forest | 99.15% |
| LSTM | 96.07% |

SVM with an RBF kernel performed best overall. The LSTM underperformed relative to the other approaches on this task — likely because the fixed-length MFCC sequence didn't benefit as much from recurrent modeling as the classical/CNN approaches did from static feature representations.

## How to Run
```
pip install -r requirements.txt
```
Open `speaker_identification.ipynb` and run all cells. Requires the AudioMNIST dataset downloaded locally (see link above).

## Tools
Python, Librosa, Scikit-learn, TensorFlow/Keras, NumPy, Matplotlib, Seaborn
