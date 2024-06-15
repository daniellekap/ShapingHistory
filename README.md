# Shaping History: Advanced Machine Learning: Techniques for the Analysis and Dating of Cuneiform
Tablets
This GitHub repository focuses on the intersection of Assyriology and machine learning, particularly through the digitization and analysis of ancient cuneiform tablets. Utilizing advanced techniques like Convolutional Neural Networks (CNNs) and Variational Auto-Encoders (VAEs), the project aims to enhance the precision of dating these tablets by examining their physical shapes. The codebase includes notebooks and Python files detailing methods for preprocessing tablet images, exploratory data analysis, and the development of machine learning models to identify patterns in tablet shapes, offering tools for Assyriologists to aid in their research.

Publication: https://arxiv.org/abs/2406.04039 

## Table of Contents

- [Introduction](#introduction)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

Cuneiform writing from ancient Mesopotamia offers invaluable insights into ancient civilizations. However, many of these clay tablets remain undated due to the scarcity of experts and established guidelines. Our project aims to automate and standardize the dating process using machine learning, specifically deep learning techniques based on generative modeling.

## Setup & Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/daniellekap/AnalysisBySynthesis.git
    cd AnalysisBySynthesis
    ```
    
2. **Install Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preparation**: 
   - Retrieve and preprocess data with notebooks 0 and 1.
   - Remove colorbars from images using notebook 1.1, resulting in 512x512 greyscale images in 'output/images_preprocessed'.

2. **Data Analysis**: 
   - Explore the data with notebooks 2 and 2.1, including viewing sample images and analyzing the height-width ratio of tablets.

3. **Model Training and Evaluation**:
   - DinoV2 - vits14: Extract features for the greyscale images using notebook 3, and for the masked images using notebook 4. Run an XGB model over the features of the greyscale images and evaluate using notebook 3.1, and for the masked images using notebook 4.1
   - CNN models: Train on greyscale images with notebook 6 and on masked images with notebook 7. Evaluate these models with notebooks 6.1 and 7.1 respectively.
   - ResNet50 models: Train on greyscale images with notebook 8 and masked images with notebook 9. Evaluate with notebooks 8.1 and 9.1. To train and predict using the "era" as a dependent variable instead of "period", use notebooks 4 and 4.1.
   - VAE model: Explore with notebook 10, analyze bottleneck vectors in notebook 10.1, and perform clustering and classification in notebook 11. Further analysis in notebooks 11.1 and 12. Notebook 13 allows exploration of shape evolution between periods.

5. **Code Structure**: 
   - Data structure and models are detailed in `era_data.py`, `era_model.py`, and `VAE_model_tablets_class.py`.
   - Visualization and colorbar removal functionalities are in `visualization_funcs.py` and `colorbar.py`.

   

## Results

The different model results can be found in notebooks 6.1 (CNN-greyscale), 7.1 (CNN-masked), 8.1 (ResNet50 Greyscale), 9.1 (ResNet50 - Masked) and 11 (VAE - masked)
VAE model performance metrics, visualizations, and comparisons with traditional methods can be found in notebook no. 11.1

## Acknowledgements

- Dr. Michael Fire of Ben Gurion University of the Negev and Dr. Shai Gording of Ariel University, for their invaluable guidance and insights.
-  Ben Gurion University of the Negev, for allowing this research.
-  Morris Alper, for his help in kicking this project off
