# Mental-health-survey-Prediction-using-ANN

🧠 Mental Health Survey Using ANN
This project is focused on analyzing mental health survey data and predicting the likelihood of depression using an Artificial Neural Network (ANN). The application also features a Streamlit-powered interactive web interface where users can input personal data and receive real-time predictions about their mental health status.


mental-health-ann/
│
├── data/
│   └── train.csv                  # Raw survey dataset
│
├── Mental health survey USING ANN.ipynb  # Main Jupyter notebook for EDA and model training
│
├── model.pkl                     # Trained ANN model (pickle format)
│
├── dip_app.py                    # Streamlit web app code
│
├── requirements.txt              # List of required Python packages
│
└── README.md                     # Project documentation

🔍 Problem Statement
With rising awareness about mental health issues, early detection of depression can help individuals seek timely support. This project aims to:

Clean and preprocess real survey data

Analyze key features affecting mental health

Build a neural network model for depression prediction

Deploy a user-friendly prediction app

🧪 Technologies Used
Python (Pandas, NumPy, Matplotlib, Seaborn)

TensorFlow / Keras for ANN modeling

Scikit-learn for preprocessing and metrics

Streamlit for web app interface

Pickle for model serialization

📊 Dataset Overview
The dataset includes responses from mental health surveys, with columns such as:

Gender, Age, City

Working Professional or Student, Profession, Work Pressure, Job Satisfaction

Dietary Habits, Degree, Work/Study Hours

Financial Stress, Family History of Mental Illness

Depression, Have you ever had suicidal thoughts?, Sleep Duration

🧹 Data Preprocessing Steps
Removed high-missing or irrelevant columns (CGPA, Study Satisfaction, Academic Pressure)

Cleaned categorical features by filtering valid entries (City, Profession, Sleep Duration, Dietary Habits)

Encoded categorical variables using label encoding or manual mapping

Handled missing values via row dropping or filtering

Split the dataset into training and testing sets

🧠 ANN Model
Input Layer: 14 features

Hidden Layers: 2–3 dense layers with ReLU activation

Output Layer: Sigmoid activation for binary classification

Loss: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall

✅ Model Evaluation
Achieved high accuracy and good precision-recall balance

Used validation loss/accuracy to tune the network

Saved final model using pickle

🌐 Streamlit App
The Streamlit app allows users to:

Enter personal and survey-related information

View real-time prediction: "Likely Depressed" or "Not Likely Depressed"

See the model's confidence level

To run the app:

bash
Copy
Edit
streamlit run dip_app.py
🔧 Installation
Clone this repository:

bash
Copy
Edit
git clone https://github.com/yourusername/mental-health-ann.git
cd mental-health-ann
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
📝 Requirements (requirements.txt)
txt
Copy
Edit
pandas
numpy
scikit-learn
seaborn
matplotlib
tensorflow
streamlit
📌 Future Improvements
Add more advanced feature engineering

Collect and validate data from more diverse sources

Deploy the Streamlit app to cloud platforms (Streamlit Cloud / Heroku)

Include user feedback for model improvement

🙋‍♂️ Author
Premkumar
 | Data Science Enthusiast




