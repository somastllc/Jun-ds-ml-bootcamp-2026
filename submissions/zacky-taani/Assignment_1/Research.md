# Data Science and Machine Learning Research File

## 1. Data Science and Machine Learning

### 1.1 Data Science

Data Science is a multidisciplinary field concerned with extracting useful information, knowledge, and evidence from data. It combines statistics, computing, data analysis, domain knowledge, and machine learning.

Data Science involves several activities, including:

* Defining problems
* Collecting data
* Cleaning and preparing data
* Exploring and analyzing data
* Building models
* Evaluating results
* Communicating findings
* Deploying and monitoring solutions

Therefore, Data Science is broader than simply analyzing data or building machine-learning models.

### 1.2 Machine Learning

Machine Learning (ML) is a branch of computing and artificial intelligence in which algorithms learn patterns from data or experience and use those patterns to perform tasks such as prediction or classification.

Instead of explicitly programming every rule, a machine-learning system can learn relationships from examples.

For example, an ML model can learn from historical house prices and their characteristics and then use those learned relationships to estimate the price of a new house.

### 1.3 Relationship Between Data Science and Machine Learning

Data Science and Machine Learning are closely related, but they are not the same thing.

**Machine Learning is one important set of techniques used within the broader field of Data Science.**

A Data Science project may use Machine Learning, but it does not necessarily have to. For example, a data scientist could analyze sales data using statistics and visualization without building an ML model.

The relationship can be represented as:

```text
Data Science
│
├── Problem Definition
├── Data Collection
├── Data Cleaning
├── Data Analysis
├── Data Visualization
├── Statistics
├── Machine Learning
├── Evaluation
└── Deployment
```

### 1.4 Real-Life Example: Healthcare

A healthcare organization could use Data Science and Machine Learning to assist with skin-lesion classification.

A Data Science project might involve:

1. Defining the medical problem.
2. Collecting clinical skin images.
3. Preparing and organizing the data.
4. Exploring the data.
5. Training a Machine Learning model.
6. Evaluating the model.
7. Considering how the system could be used in practice.
8. Monitoring its performance after deployment.

Machine Learning is therefore part of the larger Data Science workflow.

A real example is the study by Esteva et al. (2017), which investigated whether a deep neural network could classify skin cancer from clinical images. The researchers trained a convolutional neural network using **129,450 clinical images** representing **2,032 diseases**. They evaluated the system on two specific skin-cancer classification tasks and compared its performance with **21 board-certified dermatologists**.

The researchers reported that the CNN achieved performance comparable to the tested dermatologists on those particular tasks.

However, this result should be interpreted within the conditions of the study. It does not mean that the system can independently diagnose every skin disease or replace medical professionals.

---

# 2. Data Science Lifecycle

The Data Science lifecycle describes the major stages involved in turning a real-world problem into a data-driven solution.

There is no single lifecycle that is used for every Data Science project. Different frameworks exist, including **CRISP-DM (Cross-Industry Standard Process for Data Mining)**. However, many frameworks contain similar activities.

A commonly used lifecycle can be represented as:

```text
Problem Definition
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Modeling
        ↓
Evaluation
        ↓
Deployment
        ↓
Monitoring
        ↺
```

## 2.1 Problem / Business Understanding

The first stage is to understand the problem that needs to be solved.

The team should determine:

* What problem needs to be solved?
* Who will use the results?
* What data is available?
* What does success mean?
* What limitations or requirements exist?

For example:

> A hospital wants to identify patients who may require additional medical screening.

A technically advanced model is not useful if it does not address the actual problem.

---

## 2.2 Data Understanding

After defining the problem, the team needs to understand the available data.

This can include:

* Identifying data sources
* Collecting relevant data
* Examining variables
* Checking data quality
* Identifying missing values
* Detecting unusual observations
* Understanding data distributions
* Checking whether the data represents the intended population

---

## 2.3 Data Preparation

Raw data usually needs to be prepared before it can be used.

Data preparation can include:

* Removing or correcting errors
* Handling missing values
* Removing duplicates
* Combining datasets
* Transforming variables
* Encoding categorical variables
* Creating useful features
* Selecting relevant variables

This stage is important because poor-quality data can lead to unreliable results even when a sophisticated model is used.

---

## 2.4 Modeling

This is the stage where **Machine Learning typically fits most directly**.

The team selects an appropriate algorithm and trains it using prepared data.

Possible algorithms include:

* Linear Regression
* Logistic Regression
* Decision Trees
* Random Forests
* Support Vector Machines
* Neural Networks
* Clustering Algorithms

Machine Learning is particularly useful when the project requires a system to learn patterns from historical data and use those patterns to make predictions or classifications.

---

## 2.5 Evaluation

After developing a model, its performance needs to be evaluated.

Depending on the problem, evaluation measures may include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Evaluation should answer more than:

> "Is the model accurate?"

It should determine whether the model performs adequately for the actual problem and whether it generalizes to data that it has not seen before.

---

## 2.6 Deployment

If the solution satisfies the project's requirements, it can be deployed for practical use.

Deployment could involve:

* A prediction API
* A dashboard
* A recommendation system
* A fraud-detection system
* A medical decision-support system
* A business application

Deployment is not necessarily the end of the lifecycle. Real-world systems may require continuous monitoring, maintenance, and retraining.

---

## 2.7 Where Does Machine Learning Fit?

Machine Learning fits most directly into the **Modeling** stage.

However, Machine Learning also interacts with other stages.

For example:

```text
Data Preparation
       ↓
   ML Modeling
       ↓
    Evaluation
       ↓
Problem Found?
   ↙       ↘
 Yes        No
 ↓           ↓
Return     Deploy
to earlier
stage
```

If the model performs poorly, the team may discover problems with the data, features, or original problem definition. Therefore, the lifecycle is usually **iterative rather than strictly linear**.

---

# 3. Supervised Learning vs. Unsupervised Learning

The main difference between supervised and unsupervised learning is whether the training data contains a known target or label.

| Feature         | Supervised Learning        | Unsupervised Learning                |
| --------------- | -------------------------- | ------------------------------------ |
| Training data   | Labeled                    | Unlabeled                            |
| Target variable | Available                  | Usually absent                       |
| Main purpose    | Predict a target           | Discover hidden structure            |
| Common tasks    | Classification, Regression | Clustering, Dimensionality Reduction |
| Example         | Predict loan outcome       | Group similar customers              |

## 3.1 Supervised Learning

In supervised learning, the algorithm learns from examples where the desired output is already known.

For example, a bank may have historical loan applications:

|  Income |    Debt | Previous Default | Loan Result |
| ------: | ------: | ---------------- | ----------- |
| $40,000 |  $5,000 | No               | Approved    |
| $25,000 | $20,000 | Yes              | Rejected    |
| $70,000 | $10,000 | No               | Approved    |

The model learns relationships between the input features and the known outcome.

The goal could then be to predict the outcome for a new application.

### Common supervised-learning tasks

#### Classification

Classification predicts a category.

```text
Email
  ↓
Machine Learning Model
  ↓
Spam / Not Spam
```

#### Regression

Regression predicts a numerical value.

```text
House Features
      ↓
Machine Learning Model
      ↓
Predicted House Price
```

---

## 3.2 Unsupervised Learning

Unsupervised learning works with data without a predefined target label.

For example, an online retailer might have information about customers:

```text
Customer
├── Age
├── Purchase Frequency
├── Average Spending
└── Product Categories
```

The algorithm can analyze similarities between customers and identify groups.

For example:

```text
Customer Data
      ↓
Unsupervised Learning
      ↓
Similarity Analysis
      ↓
Group 1
Group 2
Group 3
```

The groups are discovered by the algorithm rather than being provided as predefined labels.

### Common unsupervised-learning tasks

* Clustering
* Dimensionality reduction
* Association analysis
* Anomaly detection

Therefore:

> **Supervised learning learns to predict known targets, while unsupervised learning attempts to discover structure or patterns without a predefined target.**

---

# 4. Overfitting

## 4.1 What Is Overfitting?

**Overfitting** occurs when a machine-learning model learns the training data too closely, including patterns that do not generalize to new data.

An overfitted model can have:

```text
Training Performance → Very High
New Data Performance → Poor
```

The main problem is poor **generalization**.

A model should learn meaningful patterns rather than memorize the training examples.

---

## 4.2 Causes of Overfitting

### 1. Excessive Model Complexity

A model with too much flexibility can learn very detailed patterns in the training data, including random noise.

For example, a very deep decision tree can continue creating branches until it closely fits individual training examples.

### 2. Too Little Training Data

A complex model trained on a small dataset may memorize the available examples instead of learning general patterns.

### 3. Noisy Data

Training data may contain:

* Measurement errors
* Incorrect labels
* Random variations
* Unusual observations

A flexible model may learn these accidental patterns.

### 4. Unrepresentative Training Data

If the training dataset does not adequately represent the data encountered in the real world, strong training performance may not translate into strong real-world performance.

### 5. Excessive Tuning

Repeatedly adjusting a model based on the same evaluation data can cause the model-development process to adapt to that dataset.

---

## 4.3 How Can Overfitting Be Prevented?

### 1. Use More Representative Data

Increasing the quantity and diversity of relevant training data can help the model learn more general patterns.

### 2. Reduce Model Complexity

Examples include:

* Limiting decision-tree depth
* Removing unnecessary features
* Using a simpler model

### 3. Regularization

Regularization adds a penalty for excessive model complexity.

For example, L1 and L2 regularization can discourage overly complex model parameters.

### 4. Cross-Validation

Cross-validation repeatedly trains and evaluates models using different portions of the available training data.

This can provide a more reliable basis for selecting models and hyperparameters.

### 5. Early Stopping

For models trained iteratively, such as neural networks, training can be stopped when validation performance stops improving.

### 6. Proper Dataset Separation

Keeping training, validation, and test data separate helps prevent the evaluation process from becoming overly influenced by the training process.

The overall goal is:

> **Good generalization to new data rather than simply achieving extremely high training accuracy.**

---

# 5. Training Data and Test Data

## 5.1 Training Data

**Training data** is the portion of a dataset used to train a machine-learning model.

The algorithm uses these examples to learn patterns and adjust its internal parameters.

```text
Training Data
      ↓
Learning Algorithm
      ↓
Trained Model
```

---

## 5.2 Test Data

**Test data** is a separate portion of the dataset that is not used to fit the final model.

After training, the model makes predictions on the test examples.

```text
Unseen Test Data
       ↓
Trained Model
       ↓
Predictions
       ↓
Performance Evaluation
```

The test set therefore provides evidence about how the model may perform on unseen data.

---

## 5.3 Why Is the Dataset Split Necessary?

Suppose a student memorizes the exact questions that will appear on an examination.

If the same questions are used to evaluate the student, the student could achieve a very high score without actually demonstrating general understanding.

Machine Learning has a similar issue.

A model can memorize aspects of its training data and achieve excellent training performance while performing poorly on new examples.

Therefore, separate data is needed to evaluate **generalization**.

---

## 5.4 Training, Validation, and Test Sets

A typical structure is:

```text
Complete Dataset
│
├── Training Set
│       └── Used to train the model
│
├── Validation Set
│       └── Used to tune/select the model
│
└── Test Set
        └── Used for final evaluation
```

For example, one project might use:

| Dataset    | Example Proportion | Purpose               |
| ---------- | -----------------: | --------------------- |
| Training   |                70% | Train the model       |
| Validation |                15% | Tune/select the model |
| Test       |                15% | Final evaluation      |

These percentages are examples, not universal requirements.

The appropriate split depends on factors such as:

* Dataset size
* Dataset structure
* Problem type
* Amount of available data
* Whether the data is time-dependent

---

## 5.5 Data Leakage

An important issue is **data leakage**.

Data leakage occurs when information from outside the training data improperly influences the model-training process.

For example, preprocessing the entire dataset before splitting it can allow information from the test set to influence the training process.

A safer approach is:

```text
Complete Dataset
       ↓
    Split Data
    ↙       ↘
Training    Test
    ↓
Fit preprocessing
    ↓
Train model
    ↓
Evaluate on test
```

The test set should remain separate until the final evaluation.

---

# 6. Healthcare Case Study

## 6.1 Selected Research Paper

**Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017).**

> *Dermatologist-level classification of skin cancer with deep neural networks.*

Published in **Nature, 542, 115–118**.

---

## 6.2 Research Problem

The researchers investigated whether a deep neural network could classify skin lesions for two binary classification tasks:

1. **Keratinocyte carcinomas vs. benign seborrheic keratoses**
2. **Malignant melanomas vs. benign nevi**

The objective was to determine whether deep learning could perform high-level visual classification of skin lesions.

---

## 6.3 Data Used

The researchers trained the system using:

* **129,450 clinical images**
* **2,032 different diseases**

The images were used to train a convolutional neural network (CNN).

The researchers then evaluated the system using biopsy-proven clinical images.

---

## 6.4 Machine Learning Method

The study used a **convolutional neural network (CNN)**, a type of deep neural network particularly useful for analyzing visual information.

The simplified workflow was:

```text
Clinical Skin Images
        ↓
Data Preparation
        ↓
CNN Training
        ↓
Learned Visual Patterns
        ↓
New Skin Image
        ↓
Classification
        ↓
Evaluation
```

The model learned visual patterns from the training images and used them to classify new images.

---

## 6.5 Evaluation

The researchers compared the CNN's performance with **21 board-certified dermatologists**.

The evaluation focused on the two specific binary classification tasks described above.

The researchers reported that the CNN achieved performance comparable to the tested dermatologists on those tasks.

---

## 6.6 Findings

The study provided evidence that deep learning can perform sophisticated image-classification tasks in dermatology.

However, the findings should be interpreted within the conditions of the experiment.

The study does **not** demonstrate that the model can:

* Diagnose every skin disease
* Work perfectly with every patient
* Replace dermatologists
* Operate without clinical oversight

Instead, the research demonstrates the potential of deep neural networks for specific medical image-classification tasks.

---

## 6.7 Data Science Lifecycle Stages Covered

The study covers several stages of the Data Science lifecycle:

| Lifecycle Stage      | Application in the Study                               |
| -------------------- | ------------------------------------------------------ |
| Problem Definition   | Classification of skin lesions                         |
| Data Understanding   | Clinical images and disease labels                     |
| Data Preparation     | Organization and preparation of image data             |
| Modeling             | Training a deep CNN                                    |
| Evaluation           | Testing the model and comparing it with dermatologists |
| Potential Deployment | Discussion of possible practical/mobile applications   |

The strongest emphasis is on **Modeling and Evaluation** because the main contribution of the study is the development and experimental assessment of a deep-learning model.

---

# Conclusion

Data Science is the broader discipline concerned with extracting useful knowledge from data, while Machine Learning provides computational methods that allow systems to learn patterns from data.

Machine Learning is therefore an important component of many Data Science projects, but Data Science includes considerably more than machine-learning modeling.

The Data Science lifecycle provides a structured approach for moving from a real-world problem through data preparation, modeling, evaluation, and deployment.

Supervised learning uses labeled data to learn a target, whereas unsupervised learning works without a predefined target and attempts to discover patterns or structure.

Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data. Techniques such as regularization, cross-validation, appropriate model complexity, and proper dataset separation can help reduce this problem.

Separating training, validation, and test data is important because it provides a better estimate of how a model will generalize to new examples.

Finally, the healthcare case study by Esteva et al. demonstrates how Data Science and Machine Learning can work together in a real research project. The researchers defined a medical classification problem, used a large image dataset, trained a deep neural network, and evaluated its performance against expert dermatologists.

---

# References

1. Provost, F., & Fawcett, T. (2013). *Data Science and its Relationship to Big Data and Data-Driven Decision Making*. Big Data, 1(1), 51–59.
   https://doi.org/10.1089/big.2013.1508

2. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.

3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
   https://hastie.su.domains/ElemStatLearn/

4. Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017). *Dermatologist-level classification of skin cancer with deep neural networks*. Nature, 542, 115–118.
   https://doi.org/10.1038/nature21056

5. Google for Developers. *Machine Learning Crash Course: Overfitting and Generalization*.
   https://developers.google.com/machine-learning/crash-course/overfitting/overfitting

6. Google for Developers. *Machine Learning Crash Course: Dividing Datasets*.
   https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets

7. scikit-learn Developers. *Common pitfalls and recommended practices: Data leakage*.
   https://scikit-learn.org/stable/common_pitfalls.html.
