# Assignment 2 – Data Foundations for Machine Learning

## Student Study Habits and Academic Performance

### 1. Dataset & Collection Method

I created a **self-made synthetic dataset** containing **50 student records**. The dataset represents study habits and academic performance. It was created manually for this assignment and is not taken from Kaggle, UCI, or another dataset repository.

The dataset contains **50 rows and 8 columns**.

| Column | Type | Description |
|---|---|---|
| `Student_ID` | Integer | Record identifier |
| `Age` | Integer | Student age |
| `Study_Hours_Per_Day` | Numerical | Daily study hours |
| `Attendance_Percent` | Numerical | Class attendance |
| `Sleep_Hours_Per_Day` | Numerical | Daily sleep hours |
| `Practice_Sessions_Per_Week` | Integer | Weekly practice sessions |
| `Study_Level` | Categorical | Low, Medium, or High |
| `Exam_Result` | Label | Pass or Fail |

### 2. Features and Label

The **features (X)** are the variables used as inputs:

`Age`, `Study_Hours_Per_Day`, `Attendance_Percent`, `Sleep_Hours_Per_Day`, `Practice_Sessions_Per_Week`, and `Study_Level`.

The **label (y)** is:

`Exam_Result`

The label has two possible values: **Pass** and **Fail**.

`Student_ID` is only an identifier and should normally be removed before machine-learning training.

### 3. Dataset Sample

| ID | Age | Study Hours | Attendance % | Sleep Hours | Practice/Week | Level | Result |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 18 | 2.0 | 6.0 | 5.0 | 3 | Low | Pass |
| 2 | 19 | 3.5 | 7.0 | 6.0 | 4 | Medium | Pass |
| 3 | 20 | 1.5 | 5.0 | 4.0 | 2 | Low | Fail |
| 4 | 21 | 4.0 | 8.0 | 7.0 | 5 | High | Pass |
| 5 | 18 | 2.5 | 6.5 | 5.5 | 3 | Medium | Pass |
| 6 | 22 | 5.0 | 8.5 | 8.0 | 6 | High | Pass |



### 4. Data Quality Issues

Before using the dataset for machine learning, several issues should be checked:

- **Missing values** – real data may contain unanswered fields.
- **Duplicates** – the same student could accidentally be recorded twice.
- **Categorical values** – `Study_Level` must be encoded into a suitable numerical format.
- **Different scales** – attendance and study hours have different numerical ranges, so scaling may be required.
- **Identifier column** – `Student_ID` should not be used as a predictive feature.
- **Invalid values** – values such as negative study hours or attendance above 100% must be detected.



### 5. Learning Type

This is a **supervised learning** problem because the dataset has a known label, `Exam_Result`.

The main machine-learning task is **binary classification**:

```text
Student Study Data → ML Model → Pass / Fail
```

If the `Exam_Result` label were removed and the goal became finding groups of students with similar study habits, the problem would instead be **unsupervised learning (clustering)**.

### 6. Machine-Learning Use Case

The dataset could be used to build a model that predicts whether a student may **Pass or Fail** based on study habits and attendance.

Possible classification algorithms include:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors


### 7. Data Science Lifecycle

This project fits mainly into the early stages of the Data Science lifecycle:

**Problem Definition → Data Collection → Data Cleaning → Exploration → Feature Engineering → Model Training → Evaluation → Deployment**

For this assignment, the main focus is **data collection, dataset structure, features, labels, and data quality**. Preprocessing will be applied in Lesson 3.



### Conclusion

This dataset demonstrates how raw data can be organized into **features and a target label** for machine learning. With 50 student records, it can be used as a basic supervised classification dataset. However, because the dataset is synthetic and small, it should be treated as an academic practice dataset rather than evidence about real students. The next step is to clean, encode, and scale the data before training a machine-learning model.

---