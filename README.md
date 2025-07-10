
# 📘 Model Evaluation and Hyperparameter Tuning

## 🔍 Overview
This project demonstrates the process of evaluating multiple machine learning models and optimizing their performance using hyperparameter tuning techniques such as **GridSearchCV** and **RandomizedSearchCV**. The dataset used is the **Breast Cancer** dataset from `sklearn`.

---

## ⚙️ Steps Covered

1. **Data Loading & Preprocessing**
   - Load Breast Cancer dataset
   - Perform train-test split
   - Apply standard scaling

2. **Model Training**
   - Train baseline models: 
     - Logistic Regression  
     - Random Forest  
     - Support Vector Machine (SVM)  
   - Evaluate using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score

3. **Hyperparameter Tuning**
   - `GridSearchCV` for SVM  
   - `RandomizedSearchCV` for Random Forest  
   - Best parameters selected based on F1-Score

4. **Model Evaluation**
   - Re-evaluate tuned models on test set  
   - Final model selection based on best overall performance

---

## 📁 File Structure

```
Week6_Assignment.py    # Main script with all steps
Week_6.txt
README.md
```

---

## 🛠️ Requirements

```bash
pip install numpy pandas scikit-learn
```

---

## 🚀 How to Run

```bash
python model_evaluation_and_tuning_assignment.py
```

---

## 📈 Output
- Evaluation metrics for all baseline and tuned models
- Final selection based on the best F1-Score
