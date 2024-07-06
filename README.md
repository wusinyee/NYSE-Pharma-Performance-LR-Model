# NYSE-Pharma-Performance-LR-Model
Linear Regression Model for Predicting Pharmaceutical Sector Performance in New York Stock Exchange

## Project Overview
This project develops a linear regression model to predict pharmaceutical sector performance using economic, market, and industry-specific indicators.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Data](#data)
5. [Model](#model)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Installation
```bash
git clone https://github.com/yourusername/pharma-performance-prediction.git
cd pharma-performance-prediction
pip install -r requirements.txt
```

## Project Structure
pharma-performance-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 1.0-data-preprocessing.ipynb
│   ├── 2.0-eda.ipynb
│   └── 3.0-model-development.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── LICENSE
├── README.md
└── requirements.txt

## Ussage
1. Run data preprocessing: python src/data/preprocess.py
2. Perform EDA: jupyter notebook notebooks/2.0-eda.ipynb
3. Train the model: python src/models/train_model.py
4. Make predictions: python src/models/predict_model.py

## Data
- Data sources: NYSE, FDA, U.S. Bureau of Economic Analysis
- Features: stock prices, economic indicators, FDA approvals
- Target variable: Pharmaceutical sector daily returns

## Model
- Algorithm: Linear Regression
- Key features: [List top 5 features]
- Performance metrics: R-squared, MAE, RMSE

## Model
[Brief summary of model performance and key insights]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
[Mandy Wu] - [wuqianyi1021@gmail.com]

Project Link: https://github.com/yourusername/pharma-performance-prediction
