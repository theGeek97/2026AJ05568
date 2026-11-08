# Obesity Level Classification: 5 ML Models + Streamlit App

## a. Problem Statement

Estimate a person's obesity level (7 classes, from Insufficient Weight to Obesity Type III) from physical attributes and eating/lifestyle habits. The goal is to implement and compare five classification algorithms (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) on the same dataset using Accuracy, AUC, Precision, Recall, F1 and MCC, and to deploy them in an interactive Streamlit app.

## b. Dataset Description  \[1 mark]

* **Source:** UCI ML Repository, *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* (id 544); also on Kaggle.
* **Instances:** 2,111 (about 77% synthetic via SMOTE, 23% collected from users in Mexico, Peru and Colombia). After dropping exact duplicates, \~2,087 remain.
* **Features:** 16 (≥ 12 required): Gender, Age, Height, Weight, family\_history\_with\_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS.
* **Target:** `NObeyesdad`, 7 classes: Insufficient\_Weight, Normal\_Weight, Overweight\_Level\_I, Overweight\_Level\_II, Obesity\_Type\_I, Obesity\_Type\_II, Obesity\_Type\_III.
* **Preprocessing:** drop duplicates; StandardScaler on numeric columns; One-Hot encoding on categorical columns; stratified 80/20 train/test split (random\_state=42). Preprocessing is bundled inside each saved sklearn Pipeline, so the app accepts raw test CSV columns.
* **Metric averaging (multi-class):** Precision/Recall/F1 are macro-averaged; AUC is one-vs-rest macro-averaged.

## c. GitHub Repository Link  \[1 mark]

<PASTE YOUR GITHUB REPO URL HERE>

Live app: <PASTE YOUR STREAMLIT COMMUNITY CLOUD URL HERE>

```
project-folder/
|-- app.py
|-- utils.py
|-- train\_models.py
|-- requirements.txt
|-- README.md
|-- test\_data.csv
|-- data/ObesityDataSet\_raw\_and\_data\_sinthetic.csv
|-- model/   (5 x .joblib pipelines, label\_encoder.joblib, meta.json, metrics.json)
```

## d. Models Used  \[5 marks]

### Comparison Table

| ML Model Name 	| Accuracy | AUC 	| Precision 	| Recall 	| F1 	| MCC |

|---|---|---|---|---|---|---|

| Logistic Regression 	| 0.8971 | 0.9858 | 0.8924 | 0.8946 | 0.8928 | 0.8800 |

| Decision Tree 	| 0.9330 | 0.9598 | 0.9322 | 0.9306 | 0.9305 | 0.9220 |

| KNN | 0.8278 		| 0.9647 | 0.8145 | 0.8197 | 0.8062 | 0.8017 |

| Naive Bayes 		| 0.5144 | 0.8376 | 0.5142 | 0.5030 | 0.4521 | 0.4566 |

| Support Vector Machine | 0.9115 | 0.9937 | 0.9076 | 0.9081 | 0.9075 | 0.8967 |

### Observations  \[3 marks]

|ML Model Name|Observation about model performance|
|-|-|
|Logistic Regression|Accuracy of 0.8971 and MCC of 0.8800 make it a strong linear baseline. Its AUC of 0.9858 is the second highest, so it ranks the classes well even where it misclassifies. It trails the Decision Tree and SVM because the class boundaries (BMI bands) are not perfectly linear.|
|Decision Tree|The best on Accuracy (0.9330), Precision (0.9322), Recall (0.9306), F1 (0.9305) and MCC (0.9220). Splitting on Weight and Height matches how the obesity classes are defined. Its AUC (0.9598) is lower than that of Logistic Regression and SVM, because a tree gives hard, coarse probabilities. It may also be slightly overfitted|
|KNN|Accuracy of 0.8278 and MCC of 0.8017 are mid-range, noticeably below the top three. Its AUC (0.9647) is better than its accuracy suggests. Distance-based voting is hurt by the many one-hot columns and by the overlapping neighbouring classes.|
|Naive Bayes|The weakest by a wide margin: Accuracy 0.5144, F1 0.4521, MCC 0.4566, AUC 0.8376. It assumes the features are independent and Gaussian, which fails here. Height and Weight are strongly related, and the one-hot columns are binary, not Gaussian.|
|Support Vector Machine|Accuracy 0.9115 and MCC 0.8967, second only to the Decision Tree. It has the highest AUC (0.9937), so it separates the classes best overall. The RBF kernel handles the curved boundaries well.|
|**Overall, Winner for your dataset?**|Decision Tree is the winner. It has the highest Accuracy, Precision, Recall, F1 and MCC. I judged by MCC (0.9220) because it is a reliable single score for 7 classes. The SVM is a close second and has the best AUC (0.9937), so it is the better choice if well-calibrated class probabilities matter more than raw accuracy.<br /><br />Notes before you submit|



