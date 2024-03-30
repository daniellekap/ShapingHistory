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
    Retrieve the data using notebooks 0 and 1. Remove the colorbars that the images have using notebook 1.1. The images would end up in the folder 'output/imaghes_preprocessed', and their size will be 512*512 (greyscale).
2. **Analyze the data**:
    Notebooks 2 and 2.1 allow exploring the data, looking at sample images, and explring the height-width ratio of the tablets, extracted from the images themselved.
4. **Run the Prediction Models**:
    4.1 To run a simple CNN network, use notebook no. 4 for greyscale images, and notebook no. 5 for the masked images. The masking process happens while the images are loaded for the model. Use notebooks 4.1 and 5.1 correspondingly to evaluate the models.
    4.2 To run a ResNet50 network, use notebook no. 6 for greyscale images, and notebook no. 7 for the masked images. The masking process happens while the images are loaded for the model. Use notebooks 6.1 and 7.1 correspondingly to evaluate the models.
    4.3 Running the VAE model can be done using notebook no. 8, and extracting the bottleneck vectors of the VAE can be done in notebook no. 9. Analyzing and evaluating the VAE bottleneck vectors using unsupervised clustering and XGBoost (for period classification) can be done in notebook no. 10. Exracting dendrograms from hierarchical clustering per group pf periods/genres, can be done in notebook 10.1. Exploring the VAE bottleneck vector distribution and using a widget for altering these vectors to see the effect on the imgage can be done in notebook 11. Notebook 12 allows to traverse and interpulate between the mean tablet of teo different periods, can be done in notebook no. 12.
5. The data structure is according to the code at `era_data.py`
6. The ResNet50 model and the simple CNN model are under `era_model.py`
7. The VAE model is under "VAE_model_tablets_class.py"
8. Visualization functions are under "visualization_funcs.py"
9. The colorbar removal code is under "colorbar.py"
   

## Results

The different model results can be founf in notebooks 4.1 (CNN-greyscale), 5.1 (CNN-masked), 6.1 (ResNet50 Greyscale), 7.1 (ResNet50 - Masked) and 10 (VAE - masked)
VAE model performance metrics, visualizations, and comparisons with traditional methods can be found in notebook no. 10

## Acknowledgements

- Dr. Michael Fire of Ben Gurion University of the Negev and Dr. Shai Gording of Ariel University, for their invaluable guidance and insights.
-  Ben Gurion University of the Negev, for allowing this research.
-  Morris Alper, for his help in kicking this project off
- All open-source tools and libraries used in this project.
