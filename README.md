# Midterm Project

<p style="text-align: center">♟</p>

## Table of Contents

- [Introduction](#introduction)
- [Required Tools](#required-tools)
- [Usage](#usage)
- [Features](#features)
- [Contact](#contact)

## Introduction

This little project is a test from the Machine Learning course curated by [DataTalks](https://datatalks.club/). The objective is create a model using Linear Regression methodologies to predict age based on clinical history. The benefits of it's usage could be understanding how different clinical markers (e.g., blood tests, genetic information) relate to biological aging could provide insights into the aging process itself. Certain diseases or conditions might have age-related progression patterns. A model could help identify these patterns and potentially inform treatment strategies. Analyzing large datasets of clinical histories could reveal trends in aging patterns within specific populations, aiding in public health planning.

## Required Tools

- Docker
- Poetry

## Usage

To get started with the project, follow these steps:

1. **Fork the Repository**: Create your own copy of the repository to make changes.
2. **Build and Launch the Application**: Navigate to the project directory in your terminal and run the following command to build and launch the API:

```
docker-compose up --build
```

3. **Visit the url**: Access the Swagger interface by navigating to http://localhost:5000 in your web browser. Follow the prompts to use the predict route.

4. **Modify with the parameters**: You can change the placeholder values of each key an see the predicted age, take in account that some features are more related to age than other features. I recommend you to modify bone density (g/cm²), vision sharpness, hearing ability (db), cognitive function, cholesterol level (mg/dl), blood glucose level (mg/dl), diastolic, systolic and pulse pressure.

5. **Install dependencies**: Navigte to the project directory and install dependencies using the following command:

```
poetry install
```

6. **Activate the virtual environment**: Only for Windows:

```
.\.venv\Scripts\activate.bat
```

macOs and Linux:

```
poetry shell
```

7. **Use the scripts**: You can use the train or predict script. Take in account that you can delete or modify keys from the dummy dictionary to understand [relationships between the features](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/059925ac72451f12cabd69d0b12ed2c6b90840c0/notebook.ipynb#L1629). Also you can train using your own model in the train script, remember to run these scripts with the following command:

```
python predict.py
```

## Features

- [x] [Notebook used for research](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/main/notebook.ipynb)
- [x] [Data preparation](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/059925ac72451f12cabd69d0b12ed2c6b90840c0/notebook.ipynb#L109)
- [x] [Feature engineering](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/059925ac72451f12cabd69d0b12ed2c6b90840c0/notebook.ipynb#L1373C10-L1373C29)
- [x] [Multiple models](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/059925ac72451f12cabd69d0b12ed2c6b90840c0/notebook.ipynb#L3120)
- [x] [Hyperparameter optimization](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/059925ac72451f12cabd69d0b12ed2c6b90840c0/notebook.ipynb#L5971)
- [x] [Script replication](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/main/train.py)
- [x] [Model exported](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/main/model_xgb_eta%3D0.1_score%3D1.206.bin)
- [x] [Flask API](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/main/app.py)
- [x] [Dependency management with Poetry](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/main/pyproject.toml)
- [x] [Containerization with Docker](https://github.com/SchneiderSix/Midterm-Project-Zoomcamp/blob/main/docker-compose.yml)
- [x] ~API Deployed on Koyeb~(not anymore)

## Contact

Ask me anything, regards

[Juan Matias Rossi](https://www.linkedin.com/in/jmrossi6/)
