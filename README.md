# NLP Process
This repository contains the web app that helps user understand the NLP process. The app is built using HTML, CSS, and JavaScript. It provides a user-friendly interface to visualize the steps involved in Natural Language Processing (NLP) tasks.
## NLP Tools
* Tokenization
* Stemming
* Embeddings
* Named Entity Recognition
* POS Tagging

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/raichu03/nlp_process.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the project directory:
   ```bash
    cd app/
    ```
4. Run the app:
   ```bash
   uvicorn main:app
   ```
5. Open your web browser and go to `http://localhost:8000` to access the app.

This will give you the access to the user interface of the NLP process. You can input text and see the results of various NLP tasks.


## Seq2Seq model training
The code to train the seq2seq model is in the file `enc-dec-lstm.py` and to train the model you need to run the python file and the model will be trained and saved for future inference. The model is trained on the `CNN/Dailymain` dataset. The dataset is available at [CNN/Dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail). You can work with various hyperparameters to get the best results. The model is trained using the `pytorch` library. 