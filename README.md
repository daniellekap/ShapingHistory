# Machine Learning-based Dating of Cuneiform Tablets

This repository contains the research and code for our thesis project on the automatic dating of cuneiform documents inscribed on clay tablets using state-of-the-art deep learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

Cuneiform writing, originating from ancient Mesopotamia, offers invaluable insights into ancient civilizations. However, many of these clay tablets remain undated due to the scarcity of experts and established guidelines. Our project aims to automate and standardize the dating process using machine learning, specifically deep learning techniques based on generative modeling.

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
    Place your data in the `output/images` directory. Ensure it's in the format as described in `data/README.md`.

2. **Run the Model**:
    Running the VAE model can be done using notebook no. 9
    Running a ResNet50 for masked cuneiform tablet classification can be done through notebook no. 7

## Results

VAE model performance metrics, visualizations, and comparisons with traditional methods can be found in notebook no. 10

## Acknowledgements

- Prof. [Advisor's Name], for their invaluable guidance and insights.
- [Institution or University Name], for supporting this research.
- All open-source tools and libraries used in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
