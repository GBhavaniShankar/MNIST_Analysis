
# Handwritten Digit Prediction

**Authors:**  
- M. Rahul (142201022)  
- G. Sai Rohith (142201019)  
- G. Bhavani Shankar (112201026)

---

## Introduction to the MNIST Dataset

- **MNIST** (Modified National Institute of Standards and Technology) is a widely used dataset in image recognition, particularly in machine learning and deep learning.
- **Dataset Details:**
  - **Images:** 70,000 handwritten digit images (0–9), each 28x28 pixels in grayscale.
  - **Training Set:** 60,000 images.
  - **Test Set:** 10,000 images.
- **Objective:** Correctly classify each handwritten digit.
- **Challenges:** 
  - Computational complexity
  - Variability in handwriting styles
  - Image noise
  - Efficient feature extraction and classification

- **Our Goal:** Achieve competitive accuracies similar to recent research by optimizing models using hyperparameter tuning, ensemble learning, and dimensionality reduction.

---

## Models Accuracy (Initial Results)

| Model         | Accuracy  |
|---------------|-----------|
| SVM           | 95.88%    |
| Random Forest | 93.85%    |
| KNN           | 86.58%    |
| Decision Tree | 90.37%    |
| Naïve Bayes   | 90.53%    |

---

## Initial Approach

- **Techniques Used:**
  - Hyperparameter tuning and ensemble methods (Bagging).
  - No feature selection or extraction in this stage.
- **Multi-Class Strategies:**  
  - One-vs-One (OvO)  
  - One-vs-Rest (OvR)
  
The following results were obtained before applying feature selection/extraction.

---

## Model Performance with Hyperparameter Tuning & Ensemble Methods

### Without Feature Extraction

| Model          | Best Parameters                                                         | Accuracy | Accuracy (OvO) | Accuracy (OvR) | Accuracy (Bagging) |
|----------------|-------------------------------------------------------------------------|----------|----------------|----------------|--------------------|
| **SVM**      | `kernel: rbf`, `gamma: 0.01`, `degree: 2`, `C: 1`                         | 96.69%   | 96.67%         | 96.64%         | 97%                |
| **Random Forest** | `n_estimators: 150`, `max_features: sqrt`, `max_depth: 40`             | 96.10%   | 95.81%         | 95.82%         | 96%                |
| **KNN**         | `weights: uniform`, `n_neighbors: 1`, `metric: euclidean`               | 95.87%   | 95.87%         | 95.87%         | 96%                |
| **Logistic**    | `solver: saga`, `penalty: l2`, `C: 0.1`                                | 91.90%   | 93.47%         | 91.31%         | 92%                |
| **Decision Tree** | `min_samples_split: 2`, `min_samples_leaf: 1`, `max_depth: None`       | 84.55%   | 90.31%         | 80.17%         | 93%                |
| **Naïve Bayes** | None                                                                  | 57.32%   | 57.32%         | 21.86%         | 57%                |

---

## Data Sampling, Feature Selection, and Feature Extraction

- **Data Sampling:**  
  The dataset was reduced to 20,000 images to lower computational complexity while preserving diversity.
- **Feature Selection:**  
  Correlation-based Feature Selection was applied (threshold = 0.02), reducing features from 784 to approximately 520.
- **Feature Extraction:**  
  Two methods were used:
  - **LDA (Linear Discriminant Analysis)**
  - **PCA (Principal Component Analysis)**
- **Post Extraction:**  
  Hyperparameter tuning and ensemble methods were re-applied after feature extraction.

### Optimal Components after Feature Extraction

| Model             | LDA Optimal Components | PCA Optimal Components |
|-------------------|------------------------|------------------------|
| **SVM**         | 9                      | 30                     |
| **Random Forest** | 9                      | 40                     |
| **KNN**           | 9                      | 50                     |
| **Decision Tree** | 9                      | 30                     |
| **Logistic**      | 9                      | 150                    |
| **Naïve Bayes**   | 9                      | 50                     |

---

## Results After Feature Extraction Using LDA

| Model          | LDA Components | Best Parameters                                                   | Accuracy | Accuracy (OvO) | Accuracy (OvR) | Accuracy (Bagging) |
|----------------|----------------|-------------------------------------------------------------------|----------|----------------|----------------|--------------------|
| **SVM**      | 9              | `C=1`, `kernel=rbf`, `gamma=auto`, `degree=4`                        | 92.75%   | 92.33%         | 91.76%         | 92.68%             |
| **Random Forest** | 9          | `max_features=log2`, `n_estimators=150`                              | 91.51%   | 91.31%         | 91.54%         | 91.46%             |
| **KNN**         | 9          | `metric=euclidean`, `n_neighbors=9`, `weights=distance`              | 91.99%   | 91.88%         | 91.73%         | 92%                |
| **Logistic**    | 9          | `C=1`, `max_iter=1000`, `penalty=l1`, `solver=saga`                  | 88.85%   | 89.34%         | 88.05%         | 88.89%             |
| **Decision Tree** | 9         | `max_depth=30`, `min_samples_leaf=4`, `min_samples_split=5`            | 86.77%   | 88.63%         | 82.63%         | 89.87%             |
| **Naïve Bayes** | 9           | None                                                              | 88.01%   | 88.01%         | 88.18%         | 88%                |

---

## Results After Feature Extraction Using PCA

| Model                | PCA Components | Best Parameters                                                           | Accuracy | Accuracy (OvO) | Accuracy (OvR) | Accuracy (Bagging) |
|----------------------|----------------|---------------------------------------------------------------------------|----------|----------------|----------------|--------------------|
| **SVM**          | 30             | `kernel: rbf`, `gamma: 0.1`, `degree: 3`, `C: 1`                            | 97.43%   | 97.43%         | 97.59%         | 97.18%             |
| **Random Forest**    | 40             | `n_estimators: 150`, `max_features: sqrt`, `max_depth: 40`                  | 94.41%   | 94.04%         | 94.56%         | 94.09%             |
| **K-NN**             | 50             | `weights: distance`, `n_neighbors: 3`, `metric: minkowski`                  | 96.59%   | 96.63%         | 96.59%         | 96.69%             |
| **Logistic Regression** | 500         | `solver: saga`, `penalty: l2`, `C: 0.1`                                   | 92.01%   | 93.32%         | 91.27%         | 91.68%             |
| **Decision Tree**    | 40             | `min_samples_split: 2`, `min_samples_leaf: 4`, `max_depth: None`            | 81.20%   | 86.59%         | 81.08%         | 90.11%             |
| **Naïve Bayes**      | 50             | None                                                                      | 88.01%   | 88.01%         | 88.17%         | 87.93%             |

---

## Final Results

| Model           | Best Method                                        | Highest Accuracy |
|-----------------|----------------------------------------------------|------------------|
| **SVM**       | PCA at 30 components, OvR                          | 97.59%           |
| **KNN**       | PCA at 50 components, Bagging                      | 96.69%           |
| **Random Forest** | Normal Hyperparameter Tuning                    | 96.1%            |
| **Logistic Regression** | Normal Hyperparameter Tuning             | 93.47%           |
| **Decision Tree** | Normal Hyperparameter Tuning, Bagging             | 93%              |
| **Naïve Bayes**   | LDA at 9 components, OvR                          | 88.18%           |

---

## Clustering Analysis

- **Visualization:**  
  Clusters were visualized in 2D using T-SNE.
- **K-Means Hyperparameter Tuning:**  
  - **Algorithm:** elkan  
  - **Initialization:** k-means++  
  - **n_clusters:** 8  
  - **n_init:** 10  
- **K-Means on 2 Principal Components:**  
  - **Results:**  
    - ARI: 0.396, NMI: 0.506  
    - ARI: 0.232, NMI: 0.355  
- **Cluster Centroids:**  
  - *Note:* Cluster *i* does not correspond to digit *i*.
- **Partitional vs Hierarchical Clustering:**  
  - **K-Means Purity:** 0.5801  
  - **Agglomerative Clustering Purity:** 0.6916  
  - *(Using 1000 datapoints for each digit)*
- **Silhouette Analysis for Optimal k:**  
  - **Actual Number of Clusters:** 10  
  - **Optimal Number of Clusters:** 8
- **Observations:**  
  - Certain digits (e.g., "1" and "7") show similar shapes or overlapping features.
  - Some "4"s may resemble "9"s due to handwriting similarities.

---

## Thank You

Thank you for reviewing the project overview and detailed performance analysis. If you have any questions or need further information, please feel free to ask.

---

