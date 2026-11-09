# GAN-Enhanced Explainable Diabetes Prediction

## Setup (Mac)
1. Download and unzip this folder onto your Desktop.
2. In Terminal:
```bash
cd ~/Desktop/Diabetes_Prediction_Project
conda create -n diabetes python=3.12 -y
conda activate diabetes
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run in order
```bash
python download_dataset.py
python check_dataset.py
python explore_data.py
python preprocess.py
python split_data.py
python gan_augmentation.py
python ml_models.py
python shap_explanation.py
python lime_explanation.py
python app.py
```
Open http://127.0.0.1:5000 for the Flask app.

## Method
The original cleaned dataset is split once (80/20, stratified, random_state=42). The GAN sees only positive-class records from the training partition. Models train on augmented training data and are evaluated on the untouched original test partition. Results will vary slightly with package versions and GAN training.

This is an educational demonstration, not a medical diagnostic tool. GAN-generated records are synthetic and require careful validation; they are not real patients or clinical evidence.
