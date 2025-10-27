# ML-based Network Intrusion Detection System

A comprehensive machine learning-based network intrusion detection system that analyzes network traffic patterns to identify and classify various types of cyber attacks. This project includes complete data preprocessing, exploratory data analysis (EDA), and experiments with multiple ML/DL models for intrusion detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

##  Overview

Network intrusion detection systems (NIDS) are critical components of cybersecurity infrastructure. This project implements a machine learning approach to detect and classify network intrusions, including:

- **DoS/DDoS Attacks**: Denial of Service and Distributed Denial of Service
- **Brute Force Attacks**: FTP-Patator, SSH-Patator
- **Web Attacks**: XSS, SQL Injection
- **Botnet Traffic**
- **Port Scanning**
- **Heartbleed Attacks**

The system uses real-world network traffic datasets and employs various machine learning algorithms to achieve high detection accuracy while minimizing false positives.

## Features

- **Comprehensive Data Pipeline**: End-to-end data cleaning and preprocessing
- **Exploratory Data Analysis**: In-depth analysis of network traffic patterns and attack characteristics
- **Multiple ML Models**: Experiments with various machine learning and deep learning algorithms
- **Real-time Detection**: Docker-based simulation environment for testing IDS capabilities
- **Attack Simulation**: Realistic DoS attack simulation using containerized environments
- **Feature Engineering**: Advanced feature selection and dimensionality reduction techniques
- **Performance Metrics**: Detailed evaluation using accuracy, precision, recall, and F1-score

## Dataset

This project primarily uses the **CIC-IDS2017** dataset, developed by the Canadian Institute for Cybersecurity (CIC). The dataset contains:

- **Size**: Over 2.8 million network flow instances
- **Duration**: 5 days of network traffic (July 3-7, 2017)
- **Features**: 80+ network flow characteristics including:
  - Flow duration and inter-arrival times
  - Packet lengths (forward/backward)
  - Header lengths
  - Flags count
  - Packet rates
  - Protocol information
- **Attack Types**: Multiple attack categories representing real-world threats
- **Class Distribution**: Highly imbalanced with benign and malicious traffic

## 📁 Project Structure

```
ML_based_Network_Intrusion_Detection/
├── data/                          # Dataset files
├── notebooks/                     # Jupyter notebooks
│   ├── data_cleaning.ipynb       # Data preprocessing and cleaning
│   ├── eda.ipynb                 # Exploratory data analysis
│   └── model_experiments.ipynb   # ML model training and evaluation
├── network/                       # Network simulation files
│   └── dos-attack/               # DoS attack simulation
│       └── ids.sh                # IDS demonstration script
├── models/                        # Trained model files
├── src/                          # Source code
│   ├── preprocessing.py          # Data preprocessing utilities
│   ├── feature_engineering.py    # Feature selection and extraction
│   ├── train.py                  # Model training scripts
│   └── evaluate.py               # Model evaluation utilities
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker Desktop
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/rohit-2304/ML_based_Network_Intrusion_Detection.git
cd ML_based_Network_Intrusion_Detection
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install Docker Desktop

Download and install Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

##  Usage

### Running the IDS Demo

This demonstration simulates a DoS attack and shows real-time intrusion detection using trained ML models.

1. **Navigate to the network simulation directory**:
   ```bash
   cd network/dos-attack
   ```

2. **Grant execution permissions**:
   ```bash
   chmod +x ids.sh
   ```

3. **Run the IDS demonstration**:
   ```bash
   ./ids.sh
   ```

4. **View the output**: The script will display real-time detection results in the terminal, showing:
   - Network traffic analysis
   - Attack detection alerts
   - Classification results
   - Performance metrics

### Data Cleaning and Preprocessing

Explore the data cleaning notebook to understand the preprocessing pipeline:

```bash
jupyter notebook notebooks/data_cleaning.ipynb
```

Key preprocessing steps include:
- Handling missing values and duplicates
- Feature normalization and standardization
- Encoding categorical variables
- Handling class imbalance (using techniques like SMOTE or RandomOverSampling)
- Memory optimization

### Exploratory Data Analysis

Analyze traffic patterns and attack characteristics:

```bash
jupyter notebook notebooks/eda.ipynb
```

The EDA includes:
- Traffic distribution analysis
- Attack type visualization
- Feature correlation analysis
- Statistical summaries
- Time-series patterns

### Model Training and Experimentation

Train and evaluate different ML models:

```bash
jupyter notebook notebooks/model_experiments.ipynb
```

## Models Implemented

This project experiments with various machine learning and deep learning algorithms:

### Traditional Machine Learning
- **Logistic Regression**: Baseline linear classifier
- **Decision Trees**: Rule-based classification
- **Random Forest**: Ensemble learning approach
- **Support Vector Machines (SVM)**: Both linear and non-linear kernels
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Fast gradient boosting

### Deep Learning
- **Multi-Layer Perceptron (MLP)**: Feedforward neural networks
- **Recurrent Neural Networks (RNN/LSTM)**: For sequential pattern detection
- **Autoencoders**: For anomaly detection

### Feature Selection Methods
- Information Gain
- Principal Component Analysis (PCA)
- XGBoost Feature Importance


### Key Findings
- Ensemble methods (Random Forest, XGBoost) achieve the highest accuracy
- Feature selection significantly reduces training time without sacrificing performance

##  Acknowledgments

- **Canadian Institute for Cybersecurity (CIC)** for the CIC-IDS2017 dataset
- Research papers and open-source projects in network intrusion detection
- The machine learning and cybersecurity communities

## 📧 Contact

Rohit - [@rohit-2304](https://github.com/rohit-2304)

Project Link: [https://github.com/rohit-2304/ML_based_Network_Intrusion_Detection](https://github.com/rohit-2304/ML_based_Network_Intrusion_Detection)

## 📚 References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*.
2. Machine Learning-Based Network Intrusion Detection: A Survey
3. Deep Learning Approaches for Network Intrusion Detection Systems

---

**Note**: This is an educational project for research and learning purposes. Always ensure you have proper authorization before testing intrusion detection systems on any network.
