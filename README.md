<p align="center">
  <a href="" rel="noopener">
 <img width=500px height=200px src="https://symbl.ai/wp-content/uploads/2022/09/2022_0913_Blog_AudioClassification-980x552.png.webp" alt="Project logo"></a>
</p>

<h3 align="center">Audio Classifier To Telephony and Microphony with Log Data</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Audio Classifier Microservice With FastApi and AI Tools
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Aditional Data](#additional)

## 🧐 About <a name = "about"></a>

This project focuses on classifying and detecting two classes of sounds using a Convolutional Neural Network (CNN) model. The model is deployed using the FastAPI framework, providing a simple and efficient way to interact with the model through an API.

The workflow is structured as follows:

- **Sound Preprocessing and Feature Extraction**: The input sound files are preprocessed to ensure theyare in the correct format and duration for themodel to analyze effectively. For feature extraction, we use the Librosa library to extract various audio features such as MFCC, ChromaFeature, Spectral Contrast, Zero-Crossing Rate,Root Mean Square Energy, and Mel-Spectrogram.
- **Model Inference**: The extracted features are thenpassed to the CNN model, which classifies theminto one of the two predefined classes.
- **Result Handling**: The classification results are returned to the client via the FastAPI endpoint,providing real-time predictions.

All processes are encapsulated within a FastAPI service, making it easy to deploy and scale the application as needed. The model can be accessed through a REST API, allowing for easy integration with other systems or applications.

### Prerequisites

Clone the repository:

```
git clone https://github.com/rezasouri/Audio_Classification.git
cd Audio_Classification
```
### Using Docker

First Navigate to `FastAPI` directory

```
cd FastAPI
```


Load Docker Image File:

```
docker load -i fastapi-audio-classifier.tar
```

Run Docker File:

```
docker run -p 8000:8000 fastapi-audio-classifier
```

### Running Locally Without Docker

If you prefer not to use Docker, you can run the APIs locally:

1. Navigate to the `FastAPIAPI` directory.
2. Run each service using Python in `Terminal` with `Uvicorn`. For example:


``` bash
uvicorn script_name:app --port PORT_NUMBER --host HOST --reload
```

- `script_name`: Replace this with the name of your Python file that contains the FastAPI app instance (e.g., `main.py`).
- `app`: This should be the name of the FastAPI instance in your script.
- `PORT_NUMBER`: Specify the port number on which the server should listen (e.g., `8000`).
- `HOST`: Define the host address (e.g., `127.0.0.1` for local development or `0.0.0.0` to allow external access).
- `--reload`: This optional flag enables auto-reload on code changes, which is very useful during development.



## 🎈 Usage <a name="usage"></a>

Once the services are running, you can send a video stream to the first microservice's endpoint to begin the face detection and recognition process. Use the following endpoint to post video data:
```
curl -X POST http://localhost:8000/predict/ \
     -F "file=@/path/to/your/audio.file" \
     -H "Content-Type: multipart/form-data"
```

Or you can go to 

```
http:\\localhost:8000/docs
```

and send post request (select video and send it)

![Alt text](/assets/image.png "Optional title")

### In Docker
You can use this command for see logs data

```
docker logs [container id]
```


## ⛏️ Built Using <a name = "built_using"></a>

- [FastApi](https://fastapi.tiangolo.com/) - Web Framework
- [Docker](https://www.docker.com/) - Containerization Platform


## ✍️ Authors <a name = "authors"></a>

- [@rezasouri](https://github.com/rezasouri) - Idea & Initial work



## 📦 Additional Data <a name = "additional"></a>

..
