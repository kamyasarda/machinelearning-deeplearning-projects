# machinelearning-deeplearning-projects
Machine Learning & Deep Learning Projects Portfolio
A collection of machine learning and deep learning projects demonstrating predictive modeling, classification, clustering, and neural network implementations using Python.

Projects Overview

Career Path Prediction Using Multiple Intelligence Theory
File: careerdata.ipynb
Applied machine learning to predict suitable career professions based on Howard Gardner's Multiple Intelligence scores.

Algorithms: Random Forest Classifier, K-Means Clustering (k=3), Hierarchical Clustering (Ward's method)
Features: Eight intelligence dimensions—Linguistic, Musical, Bodily-Kinesthetic, Logical-Mathematical, Spatial-Visualization, Interpersonal, Intrapersonal, Naturalist
Analysis Methods: Elbow method for optimal cluster identification, dendrogram analysis, feature importance ranking
Findings: Identified distinct career personality clusters and determined which intelligence traits most strongly predict specific professional paths

______________________________

Agricultural Loan Repayment Prediction
File: jayalaxmiagro23jan.ipynb
Applied logistic regression to predict agricultural loan repayment status using demographic and financial data from 1,218 farmers in Karnataka, India.
Algorithm: Logistic Regression
Accuracy: 67.2%, AUC: 0.656
Key Predictors: Income per acre, crop insurance, sericulture training, pest impact
Findings: Income stability is the strongest driver of repayment behavior; model exhibits class imbalance bias toward predicting successful repayment

______________________________

Bollywood Box Office Success Prediction
File: careerdata.ipynb
Comparative analysis of K-Nearest Neighbors and Decision Tree classifiers for predicting commercial success of 149 Bollywood films.

Algorithms: KNN (k=5) and Decision Tree (max_depth=5)
Performance: KNN achieved 63.33% accuracy (AUC: 0.581) vs Decision Tree 50% (AUC: 0.486)
Key Predictors: YouTube views and likes account for 78.5% of feature importance
Findings: Digital engagement metrics dominate traditional factors (budget, star power) in predicting box office performance
______________________________
Hospital Package Price Prediction & Patient Clustering
File: hospital.ipynb
Combined predictive modeling and unsupervised learning to estimate treatment costs and identify patient segments using 248 records from Mission Hospital Durgapur.

Algorithms: Random Forest Regression for price prediction, Hierarchical Clustering (Ward's linkage) for segmentation
Regression Performance: R² = 0.728, MAE = ₹39,563
Clustering Results: Three patient segments identified—Routine care (53%), Moderate cases (34%), Critical care (13%)
Findings: Hospital stay duration and ICU time are primary cost determinants

______________________________

Netflix Customer Churn Prediction & Segmentation
File: netflix.ipynb
Developed churn prediction model and customer segmentation analysis using 4,990 Netflix subscriber records.

Algorithms: Random Forest Classifier for churn prediction, K-Means Clustering (k=3) for segmentation
Classification Performance: 97.6% accuracy, Precision: 0.99, Recall: 0.97
Customer Segments: Basic plan users (33%), At-risk users (34%), Premium users (33%)
Findings: Behavioral engagement patterns (daily watch time: 40% importance, login frequency) predict churn significantly better than demographic variables
______________________________
Deep Learning Image Classification
File: deeplearning.ipynb
Implementation of convolutional neural networks for image recognition tasks.

Framework: TensorFlow/Keras
Architecture: Convolutional Neural Networks (CNN)
Date Fruit Classification using Artificial Neural Networks
File: date-fruit-classification-using-ann.ipynb
Multi-class classification of date fruit varieties using artificial neural networks.

Algorithm: Artificial Neural Network (ANN)
Framework: TensorFlow/Keras
Application: Agricultural product quality assessment through image classification

______________________________
Technologies Used

Languages: Python
Libraries:
scikit-learn (ML models)
pandas, numpy (Data manipulation)
matplotlib, seaborn (Visualization)
TensorFlow, Keras (Deep Learning)


Algorithms:
Supervised Learning: Logistic Regression, Random Forest, KNN, Decision Trees
Unsupervised Learning: K-Means, Hierarchical Clustering
Deep Learning: ANN, CNN


Tools: Jupyter Notebook, VS Code, Google Colab
