# Customer Churn Classification using ANN & Streamlit

An end-to-end Machine Learning project that predicts customer churn using an Artificial Neural Network (ANN). This repository contains the complete pipeline, from exploratory data analysis and model training to deployment through an interactive Streamlit web application.

## 🚀 Project Overview

Customer churn prediction is crucial for businesses to identify users who are likely to stop using a service. This project leverages the `Churn_Modelling` dataset to train a deep learning classifier. The trained model is served via a Streamlit interface, allowing users to input customer parameters and receive real-time churn probability predictions.

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow / Keras
* **Web Framework:** Streamlit
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Environment:** Python 3.x, Jupyter Notebook

## 📁 Repository Structure

\`\`\`text
├── Churn_Modelling.csv          # Raw dataset used for training
├── app.py                       # Streamlit application for the web UI
├── experiments.ipynb            # EDA, data preprocessing, and ANN training
├── prediction.ipynb             # Notebook for testing model inference offline
├── model.h5                     # Saved trained Artificial Neural Network model
├── scaler.pkl                   # Saved StandardScaler object for feature scaling
├── label_encoder_gender.pkl     # Saved LabelEncoder for the 'Gender' column
├── one_hot_encoder_geo.pkl      # Saved OneHotEncoder for the 'Geography' column
├── requirements.txt             # Required Python dependencies
├── LICENSE                      # GPL-2.0 License
└── README.md                    # Project documentation
\`\`\`

## ⚙️ Installation

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/Shourya2003ML/ANN-Classification-Churn-Streamlit.git
   cd ANN-Classification-Churn-Streamlit
   \`\`\`

2. **Create a virtual environment (Optional but recommended):**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   \`\`\`

3. **Install the dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## 💻 Usage

To launch the Streamlit web application, run the following command in your terminal:

\`\`\`bash
streamlit run app.py
\`\`\`

This will start a local server. Open the provided localhost URL in your web browser. You can interact with the UI, input customer details (such as Credit Score, Geography, Age, Balance, etc.), and click "Predict" to see the likelihood of the customer churning.

## 🧠 Model Pipeline

1. **Preprocessing:** Categorical variables are transformed using Label Encoding (`Gender`) and One-Hot Encoding (`Geography`). Continuous numerical variables are scaled using Scikit-Learn's `StandardScaler` to optimize the convergence of the neural network.
2. **Architecture:** The ANN consists of dense hidden layers with ReLU activation functions, and an output layer utilizing a Sigmoid activation function for binary classification (Churn vs. No Churn).
3. **Inference Pipeline:** `app.py` loads the saved weights (`model.h5`) alongside the exact preprocessing pickle files to ensure that any new user input from the web app is scaled and encoded identically to the original training data.

## 📄 License

This project is licensed under the GPL-2.0 License. See the [LICENSE](LICENSE) file for more details.
