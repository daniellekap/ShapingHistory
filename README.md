# Machine Learning-based Dating of Cuneiform Tablets


This GitHub repository focuses on the intersection of Assyriology and machine learning, particularly through the digitization and analysis of ancient cuneiform tablets. Utilizing advanced techniques like Convolutional Neural Networks (CNNs) and Variational Auto-Encoders (VAEs), the project aims to enhance the precision of dating these tablets by examining their physical shapes. The codebase includes notebooks and Python files detailing methods for preprocessing tablet images, exploratory data analysis, and the development of machine learning models to identify patterns in tablet shapes, offering tools for Assyriologists to aid in their research.

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
   - CNN models: Train on greyscale images with notebook 4 and on masked images with notebook 5. Evaluate these models with notebooks 4.1 and 5.1 respectively.
   - ResNet50 models: Train on greyscale images with notebook 6 and on masked images with notebook 7. Evaluate with notebooks 6.1 and 7.1.
   - VAE model: Explore with notebook 8, analyze bottleneck vectors in notebook 9, and perform clustering and classification in notebook 10. Further analysis in notebooks 10.1 and 11. Notebook 12 allows exploration of shape evolution between periods.

4. **Code Structure**: 
   - Data structure and models are detailed in `era_data.py`, `era_model.py`, and `VAE_model_tablets_class.py`.
   - Visualization and colorbar removal functionalities are in `visualization_funcs.py` and `colorbar.py`.

   

## Results

The different model results can be found in notebooks 4.1 (CNN-greyscale), 5.1 (CNN-masked), 6.1 (ResNet50 Greyscale), 7.1 (ResNet50 - Masked) and 10 (VAE - masked)
VAE model performance metrics, visualizations, and comparisons with traditional methods can be found in notebook no. 10

## Acknowledgements

- Dr. Michael Fire of Ben Gurion University of the Negev and Dr. Shai Gording of Ariel University, for their invaluable guidance and insights.
-  Ben Gurion University of the Negev, for allowing this research.
-  Morris Alper, for his help in kicking this project off
- All open-source tools and libraries used in this project.
