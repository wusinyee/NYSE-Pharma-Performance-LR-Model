# NYSE-Pharma-Performance-LR-Model
Linear Regression Model for Predicting Pharmaceutical Sector Performance in New York Stock Exchange

## Project Overview
This project develops a linear regression model to predict pharmaceutical sector performance using economic, market, and industry-specific indicators.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Outline](#outline)
4. [Usage](#usage)
5. [Data](#data)
6. [Model](#model)
7. [Results](#results)
8. [License](#license)
9. [Contact](#contact)

## Installation
```bash
git clone https://github.com/yourusername/pharma-performance-prediction.git
cd pharma-performance-prediction
pip install -r requirements.txt
```

## Project Structure
```markdown
pharma-performance-prediction/
│
├── data/
│ ├── raw/
│ │ └── .gitkeep
│ └── processed/
│ └── .gitkeep
│
├── notebooks/
│ ├── 1.0-data-preprocessing.ipynb
│ ├── 2.0-exploratory-data-analysis.ipynb
│ └── 3.0-model-development.ipynb
│
├── src/
│ ├── data/
│ │ ├── init.py
│ │ └── preprocess.py
│ ├── features/
│ │ ├── init.py
│ │ └── build_features.py
│ ├── models/
│ │ ├── init.py
│ │ ├── train_model.py
│ │ └── predict_model.py
│ └── visualization/
│ ├── init.py
│ └── visualize.py
│
├── tests/
│ ├── init.py
│ ├── test_data.py
│ ├── test_features.py
│ └── test_models.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```
This structure follows best practices for organizing a data science project:

- `data/`: Stores raw and processed data files.
- `notebooks/`: Contains Jupyter notebooks for exploration and analysis.
- `src/`: Houses the main source code of the project.
- `tests/`: Includes unit tests for different components.
- Root directory files for project setup and documentation.

## Outline 
**Pharmaceutical Sector Performance Prediction Project Outline**

1. Data Collection and Preparation
   a. Stock Data Collection
      - NYSE historical dataset from Kaggle
      - S&P 500 index data
      - API-fetched pharmaceutical company data
   b. Economic Data Collection
   c. Healthcare Data Collection
   d. Market Sentiment Data Collection
   e. Data Preprocessing
   f. Data Integration
   g. Data Quality Checks
   h. Feature Engineering
   i. Data Documentation

2. Exploratory Data Analysis (EDA)
   a. Analyze variable distributions
   b. Investigate correlations
   c. Examine time series characteristics
   d. Visualize key relationships

3. Feature Selection
   a. Statistical methods (correlation, VIF, mutual information)
   b. Domain knowledge application

4. Model Development
   a. Data splitting (train, validation, test)
   b. Baseline model implementation
   c. Advanced model development
      - Linear models (Ridge, Lasso)
      - Tree-based models (Random Forest, Gradient Boosting)
      - Support Vector Regression
      - Neural Networks
   d. Cross-validation

5. Model Optimization
   a. Hyperparameter tuning
   b. Ensemble methods exploration

6. Model Evaluation and Selection
   a. Performance metric comparison
   b. Model interpretability assessment
   c. Final model selection

7. Model Interpretation
   a. Feature importance analysis
   b. SHAP value analysis

8. Model Validation
   a. Test set evaluation
   b. Backtesting
   c. Sensitivity analysis

9. Deployment Planning
   a. Deployment system design
   b. Infrastructure setup
   c. Prediction pipeline development

10. Documentation and Reporting
    a. Technical documentation
    b. Final report preparation
    c. Visualization creation

11. Stakeholder Presentation
    a. Presentation preparation
    b. Key findings and results communication

12. Model Deployment
    a. Implementation of deployment system
    b. Testing and quality assurance

13. Monitoring and Maintenance
    a. Performance monitoring setup
    b. Retraining schedule establishment
    c. Version control implementation

14. Compliance and Ethics
    a. Regulatory compliance review
    b. Fairness and bias assessment
    c. Ethical use guidelines development

15. Knowledge Transfer
    a. User guide creation
    b. Training session conduction
    c. Support system setup

16. Impact Assessment
    a. Model impact measurement
    b. Efficiency gains quantification
    c. Stakeholder feedback collection

17. Iterative Improvement
    a. Regular performance reviews
    b. Continuous improvement implementation

18. Scaling and Expansion
    a. Scalability assessment
    b. Expansion roadmap development

19. Project Closure
    a. Comprehensive project review
    b. Lessons learned documentation
    c. Formal project closure

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
