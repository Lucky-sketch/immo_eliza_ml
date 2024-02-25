# Immo Eliza ML: Forecasting Real Estate Values in Belgium

Welcome to the Real Estate Price Prediction project by Immo Eliza! 🏡 In this machine learning project, we use the XGBoost algorithm to predict real estate prices in Belgium. Follow the guide below to explore the project and start predicting property prices with confidence!


## Quick Start

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Lucky-sketch/immo_eliza_ml.git

2. **Set Up Your Virtual Environment:**

   ```bash
   python3 -m venv venv

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

4. **Run the Training Script:**

   ```bash
   python3 train.py

4. **Predict for unseen data:**

   ```bash
   python predict.py -i data/properties.csv -o output/predictions.csv

## Project Anatomy
The project is structured as follows:

**data:** Contains the cleaned dataset, "properties.csv."    
**models:** The trained XGBoost model and related artifacts.       
**train.py, predict.py:** Contains the source code for data preprocessing and model training.
**.gitignore:** Specifies ignored files and folders for version control.  
**README.md:**  This comprehensive guide you're currently reading.     
**requirements.txt:** Enumerates project dependencies for straightforward setup.

## Data Preprocessing
           
**Categorical Features:** Applied OneHotEncoder for one-hot encoding.                      
**Feature List:**               
- **Numerical Features:**         
  - ["nbr_frontages", 'nbr_bedrooms', "latitude", "longitude", "total_area_sqm",'surface_land_sqm','terrace_sqm','garden_sqm']
- **Categorical Features:**                 
  - ["province", 'heating_type', 'state_building', "property_type", "epc", 'locality', 'subproperty_type','region', "fl_terrace", 'fl_garden', 'fl_swimming_pool']


## Model Training

**Model Training Data:** Split the model training data into training (80%) and testing (20%) sets through random sampling.      
**XGBoost Model:** Trained with the best hyperparameters obtained through RandomizedSearchCV.                                
**Evaluation**: R² scores calculated for both training and testing sets.                                  

## Model Artifacts

**features:** Information about numerical and categorical features.             
**enc**: The OneHotEncoder object for categorical feature encoding.               
**model**: The trained XGBoost model.

## 🤝 Contributing
Contributions are welcome! Please feel free to open issues, suggest improvements, or submit pull requests. Let's collaborate to enhance this project together!

![giphy (2)](https://github.com/Lucky-sketch/immo_eliza_ml/assets/53155116/f1c89a5c-d941-4399-ad59-1c5946a7115e)

