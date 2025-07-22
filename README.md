# Ocular-Disease-Detection
Eye Disease Detection to detect eye/retina problems
Eye Disease Detection using ODIR-5K Dataset

Project Overview
This project is focused on building a machine learning pipeline to detect various eye diseases from retinal fundus images using the ODIR-5K dataset. The pipeline involves data preprocessing, feature extraction, training multiple machine learning models, and evaluating their performance to predict the presence of eye diseases.
The goal of the project is to implement and compare different machine learning algorithms and identify the one with the best accuracy for disease detection.

________________________________________

Dataset
The ODIR-5K (Ocular Disease Intelligent Recognition) dataset is a publicly available dataset containing 5,000 retinal fundus images. The dataset consists of:
•	Left-Fundus Images: Images of the left eye.
•	Right-Fundus Images: Images of the right eye.
•	Labels: Diagnostic labels indicating the presence of different ocular diseases.
Dataset Structure
The dataset is divided into:
•	Training Images: Used for training the model.
•	Testing Images: Used for evaluating the model's performance.
Columns in data.xlsx:
•	Left-Fundus: Name of the left eye image file.
•	Right-Fundus: Name of the right eye image file.
•	Diagnosis: Labels for the presence of diseases like N/M/C/D/G/H/AMD/DR.
________________________________________





Project Structure
├── ODIR-5K/
│   ├── Training Images/
│   │   ├── 0_left.jpg
│   │   ├── 0_right.jpg
│   ├── Testing Images/
│   │   ├── 1001_left.jpg
│   │   ├── 1001_right.jpg
│   ├── data.xlsx
├── cnn_model.py
├── feature_extraction.py
├── README.md
________________________________________
Implementation
1. Data Preprocessing
•	Load and preprocess the images from the dataset.
•	Normalize the pixel values for better convergence during training.
2. Feature Extraction
•	Use pre-trained deep learning models like VGG16 to extract features from the retinal images.
•	Apply Principal Component Analysis (PCA) to reduce the dimensionality of the extracted features.
3. Model Training
•	Train a Convolutional Neural Network (CNN) for binary/multiclass classification.
•	Experiment with various machine learning algorithms such as:
o	Logistic Regression
o	Support Vector Machine (SVM)
o	Random Forest
o	K-Nearest Neighbors (KNN)
o	XGBoost
o	Decision Tree
o	Naive Bayes
o	Multi-Layer Perceptron (MLP)
4. Model Evaluation
•	Evaluate the models using metrics like:
o	Accuracy
o	Precision
o	Recall
o	F1-Score
o	Confusion Matrix
________________________________________
Dependencies
To run this project, you need the following libraries:
•	Python 3.x
•	TensorFlow
•	Keras
•	NumPy
•	Pandas
•	Scikit-Learn
•	Matplotlib
•	Seaborn
Install the dependencies using:
bash
Copy code
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
________________________________________
How to Run the Project
Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/your-repository-name/odir5k-eye-disease-detection.git
cd odir5k-eye-disease-detection
Step 2: Organize the Dataset
Place the ODIR-5K dataset in the following structure:
kotlin
Copy code
ODIR-5K/
├── Training Images/
├── Testing Images/
└── data.xlsx
Step 3: Train the Model
Run the following command to train the CNN model:
bash
Copy code
python cnn_model.py
Step 4: Evaluate the Model
Run the evaluation script to test the model on new images:
bash
Copy code
python evaluate_model.py
________________________________________
Results
Model	Accuracy	Precision	Recall	F1-Score
CNN	85%	84%	86%	85%
Logistic Regression	78%	76%	79%	77%
Random Forest	81%	80%	82%	81%
________________________________________
Future Work
•	Fine-tune the CNN model for better accuracy.
•	Explore ensemble learning techniques for improved performance.
•	Integrate more pre-trained models like ResNet and Inception.
•	Deploy the model as a web application for real-time disease detection.
________________________________________
Acknowledgments
We acknowledge the contributors of the ODIR-5K dataset for providing valuable resources for ocular disease detection.


DATASET LINK-

https://drive.google.com/drive/folders/1r21IeeVamsls-r-qeaPGysk2f4Kw8Y_v?usp=sharing
