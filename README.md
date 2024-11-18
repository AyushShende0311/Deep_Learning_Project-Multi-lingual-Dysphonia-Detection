# Multi-lingual Dysphonia Detection and Severity Estimation Using Deep Learning Architectures

### Project team: Hung - Ayush
1. **Vinh Hung Nguyen** - K2KY36
2. **Ayush Nitin Shende** - WLIW35

## Description
This project aims to develop a deep learning model to diagnose dysphonic speech in multilingual scenarios and estimate the severity of the condition.

### Data
We are using audio data in multiple languages (Hungarian, English, German, Dutch), categorized by gender and speaker status (healthy or patient). The data is sourced directly to Google Colab from Google Drive.
This dataset is substantial and has been sourced from various domains, primarily from the medical testing domain. We have obtained official consent to use the data for this project and we received the data from BME university's database via the project instructor.


### Dataset Characteristics:
- The dataset includes a variety of languages, with an imbalance in the number and length of files across languages.
- German files are shorter and more abundant, while Hungarian, Dutch, English, Portuguese have fewer but longer sequences.
- To address this imbalance, we chunked the longer audio files (Hungarian, Dutch, English, Portuguese) to match the shorter length of German files.

## Data Processing
1. **Chunking the Audio**  
   Audio files in Hungarian, English, Dutch, and Portuguese were chunked to match the length of the German files using the `Chunking_Script.ipynb` notebook, which contains the `chunk_audio_in_folder` function.

2. **Speaker Embeddings Extraction**  
   We used a pretrained TDNN xvector model (via [SpeechBrain](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)) to extract speaker embeddings for each audio file. This process is detailed in the `EmbeddingModel.ipynb` notebook, and outputs a CSV file (`embeddings_final.csv`) with the following information:
   - `File_name`: Name of the chunked audio file.
   - `Embedding`: Speaker embedding.
   - `Category`: Indicator of healthy or dysphonic voice.
   - `Gender`: Speaker's gender.
   - `Language`: Language of the audio file.

This CSV file forms the main dataset for training our machine learning model, which will detect specific patterns in the speech and identify the presence of dysphonia.

## Model
   For this project, we use Siamese Neural Network (SNN) as our model. For running the model, to use SNN, pairs of embedding vectors are made which can be either similar or dissimilar (see `TrainingSN.ipynb` notebook - `DysphoniaPairsDataset`), after which we define the SNN architecture and train the model. The results are printed out in the same notebook.
   <p align="center"><img width="50%" src="1_siamese-network.png" /></p>
    
## Reference
   - https://builtin.com/machine-learning/siamese-network
   - https://github.com/sudharsan13296/Hands-On-Meta-Learning-With-Python/blob/master/02.%20Face%20and%20Audio%20Recognition%20using%20Siamese%20Networks/2.5%20Audio%20Recognition%20using%20Siamese%20Network.ipynb
   - https://en.wikipedia.org/wiki/Siamese_neural_network
