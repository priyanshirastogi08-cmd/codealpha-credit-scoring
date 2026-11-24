# CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model

## 📌 Project Overview
This repository contains the solution for **Task 1** of the CodeAlpha Machine Learning Internship. The objective of this project is to build a robust binary classification model to predict credit risk (**Good** vs. **Bad** credit) based on applicant financial and demographic attributes, helping financial institutions make informed lending decisions.

---

## 📂 Dataset Details
* **Dataset Used:** German Credit Dataset 
* **Instances (Rows):** 1,000
* **Attributes (Columns):** 21 features (including account status, duration, credit amount, employment history, and credit history).
* **Target Variable:** `target` (Binary encoded as `1` for Good Credit Risk and `0` for Bad Credit Risk).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Development Environment:** VS Code
* **Core Libraries:**
  * `pandas` & `numpy` (Data manipulation and preprocessing)
  * `scikit-learn` (Model training, scaling, evaluation metrics, and cross-validation)
  * `matplotlib` & `seaborn` (Data visualization and performance plotting)

---

## ⚙️ Methodology & Workflow
1. **Data Loading & Cleaning:** Checked dataset shape, handled missing values, and stripped accidental whitespaces from column headers.
2. **Feature Engineering & Encoding:** 
   * Mapped target string labels (`'good'`, `'bad'`) to binary numeric values (`1`, `0`).
   * Applied **One-Hot Encoding** (`pd.get_dummies`) to convert categorical text features into numeric formats.
3. **Train-Test Split:** Split the dataset into 80% training and 20% testing sets with stratification (`stratify=y`) to maintain class balance.
4. **Feature Scaling:** Utilized `StandardScaler` to normalize numerical features, optimizing convergence for Logistic Regression.
5. **Model Training & Comparison:** Trained and evaluated three distinct classification models:
   * **Logistic Regression**
   * **Decision Tree Classifier**
   * **Random Forest Classifier**

---

## 📊 Model Evaluation Metrics
Each model was evaluated using industry-standard machine learning metrics:
* **Accuracy Score**
* **Precision, Recall, and F1-Score** (via Classification Report)
* **ROC-AUC Score & ROC Curves**
* **Confusion Matrices** (visualized via Seaborn heatmaps)

---

## 🚀 How to Run the Project Locally

1. Clone or download this repository into your local machine.
2. Open the project folder in **VS Code**.
3. Install the required dependencies via your terminal:
   ```bash
   pip install -r requirements.txt

   # CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model

## 📌 Project Overview
This repository contains the solution for **Task 1** of the CodeAlpha Machine Learning Internship. The objective of this project is to build a robust binary classification model to predict credit risk (**Good** vs. **Bad** credit) based on applicant financial and demographic attributes, helping financial institutions make informed lending decisions.

---

## 📂 Dataset Details
* **Dataset Used:** German Credit Risk Dataset (Statlog / UCI Repository)
* **Instances (Rows):** 1,000
* **Attributes (Columns):** 21 features (including account status, duration, credit amount, employment history, and credit history).
* **Target Variable:** `target` (Binary encoded as `1` for Good Credit Risk and `0` for Bad Credit Risk).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Development Environment:** VS Code
* **Core Libraries:**
  * `pandas` & `numpy` (Data manipulation and preprocessing)
  * `scikit-learn` (Model training, scaling, evaluation metrics, and cross-validation)
  * `matplotlib` & `seaborn` (Data visualization and performance plotting)

---

## ⚙️ Methodology & Workflow
1. **Data Loading & Cleaning:** Checked dataset shape, handled missing values, and stripped accidental whitespaces from column headers.
2. **Feature Engineering & Encoding:** 
   * Mapped target string labels (`'good'`, `'bad'`) to binary numeric values (`1`, `0`).
   * Applied **One-Hot Encoding** (`pd.get_dummies`) to convert categorical text features into numeric formats.
3. **Train-Test Split:** Split the dataset into 80% training and 20% testing sets with stratification (`stratify=y`) to maintain class balance.
4. **Feature Scaling:** Utilized `StandardScaler` to normalize numerical features, optimizing convergence for Logistic Regression.
5. **Model Training & Comparison:** Trained and evaluated three distinct classification models:
   * **Logistic Regression**
   * **Decision Tree Classifier**
   * **Random Forest Classifier**

---

## 📊 Model Evaluation Metrics
Each model was evaluated using industry-standard machine learning metrics:
* **Accuracy Score**
* **Precision, Recall, and F1-Score** (via Classification Report)
* **ROC-AUC Score & ROC Curves**
* **Confusion Matrices** (visualized via Seaborn heatmaps)

---

## 🚀 How to Run the Project Locally

1. Clone or download this repository into your local machine.
2. Open the project folder in **VS Code**.
3. Install the required dependencies via your terminal:
   ```bash
   pip install -r requirements.txt

   

## 👤 Author
* **Priyanshi Rastogi**  
* *B.Tech Computer Science and Engineering, IIIT Pune*  
* *CodeAlpha Machine Learning Intern*
