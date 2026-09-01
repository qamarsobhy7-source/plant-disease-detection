# Plant Disease Detection System

A computer vision application designed to process crop imagery and identify leaf health anomalies automatically.

## Overview
This repository contains a trained model pipeline that analyzes plant images to detect signs of disease. The system uses visual feature extraction to categorize leaf conditions, supporting early detection in agricultural environments.

## Technologies
* Python
* TensorFlow and Keras
* OpenCV and NumPy

## Repository Layout
* `plant_disease_model.h5`: The primary weights and architecture file containing the trained model ready for evaluation.
* `README.md`: Project documentation.

## Implementation Details
* Processes and standardizes input images for model compatibility.
* Utilizes a convolutional network structure designed to recognize spatial features and patterns in leaf datasets.
* Exports the optimized output parameters directly into a reusable `.h5` file.
* 
