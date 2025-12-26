# Stock Price Prediction Project

This repository contains a complete pipeline for preparing historical stock market data and constructing datasets for machine learning–based stock price prediction.  
The workflow is designed to run on **NERSC Perlmutter compute nodes**, but it can also be adapted to other Linux-based HPC systems.

---

## Computing Environment

- **System:** NERSC Perlmutter (compute nodes)
- **OS:** Linux
- **Python:** 3.9+
- **Execution mode:** Batch or interactive compute nodes (not login nodes)

> ⚠️ It is strongly recommended to run all data processing scripts on **compute nodes**, not login nodes.

---

## Dataset

This project uses the **Stock Market Dataset** from Kaggle.

- **Dataset:** Stock Market Dataset
- **Source:** https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

The dataset is **not included in this repository** and must be downloaded locally.

---

## Step 1: Download the Dataset

### Create the data directory
```bash
mkdir -p data
cd data
curl -L -o ~/Downloads/stock-market-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset
mv ~/Downloads/stock-market-dataset.zip .
unzip stock-market-dataset.zip
```

## Step 2: Install Python Dependencies

It is recommended to use a virtual environment on Perlmutter compute nodes.
```bash
module load python/3.12
pip install -r requirements.txt
```

## Step 3: Run Data Pipeline

```bash
python prepareData.py
python generateFeatures.py
python generateLabels.py
python prepareDataset.py
```

- `prepareData.py`: Loads and cleans raw CSV stock data, producing structured intermediate outputs.
- `generateFeatures.py`, `generateLabels.py`, `prepareDataset.py`: Generate technical features and labels, then merge and filter them into a final dataset for CNN/LSTM time-series prediction.