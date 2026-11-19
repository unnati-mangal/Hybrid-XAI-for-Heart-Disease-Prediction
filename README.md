# CardioXAI
### Hybrid Explainable AI for Heart Disease Prediction Using Clinical and Lifestyle Data

**Minor Project | B.Tech (AI & ML) | Semester III**

CardioXAI is a web-based cardiovascular risk assessment application that combines Machine Learning (ML) and Explainable Artificial Intelligence (XAI) to estimate an individual's 10-year risk of developing coronary heart disease (CHD).

---

## 1. Project Title

**Hybrid Explainable AI for Heart Disease Prediction Using Clinical and Lifestyle Data**

**Application Name:** CardioXAI

**Project Domain:** Machine Learning, Explainable Artificial Intelligence, Healthcare Technology, Web Development

## 2. Project Description / Overview

CardioXAI is designed to provide an accessible and interpretable approach to cardiovascular risk assessment using clinical and lifestyle information.

The application uses a **Logistic Regression model trained on the Framingham Heart Study dataset** to estimate an individual's probability of developing coronary heart disease within the next ten years.

During model development, Logistic Regression, Random Forest, Gradient Boosting, and LightGBM were evaluated. Logistic Regression was selected based on its comparative predictive performance and interpretability.

Unlike a prediction-only system, CardioXAI integrates two Explainable Artificial Intelligence techniques:

- **SHAP (SHapley Additive exPlanations):** Identifies the contribution of individual features to a prediction.
- **LIME (Local Interpretable Model-agnostic Explanations):** Provides local explanations of the model's behavior for individual assessments.

The application also includes user authentication, a personalized dashboard, prediction history, and downloadable assessment reports.

**Note:** CardioXAI is developed for educational and cardiovascular risk-awareness purposes. It is not a medical diagnostic tool.

## 3. Problem Statement

Cardiovascular diseases are a major health concern, and identifying individuals who may be at risk is important for preventive healthcare.

Although traditional cardiovascular risk calculators and machine learning-based systems can estimate disease risk, some approaches provide limited explanations of how individual clinical and lifestyle factors influence the generated predictions.

This creates challenges such as:

- **Limited Explainability:** Users may receive a risk score without understanding the factors contributing to the prediction.
- **Difficulty in Interpretation:** Numerical risk estimates and technical information may be difficult for non-technical users to understand.
- **Fragmented Functionality:** Risk prediction, explanation, assessment history, and report generation may not be available within a single platform.

CardioXAI addresses these challenges by integrating machine learning-based risk estimation, SHAP and LIME explanations, prediction history, and downloadable reports into one web application.

## 4. Project Objectives

The main objectives of CardioXAI are:

1. Develop a machine learning-based system to estimate 10-year coronary heart disease risk.
2. Integrate SHAP and LIME to explain the factors influencing individual predictions.
3. Create an interactive and user-friendly web application.
4. Implement user registration, authentication, and a personalized dashboard.
5. Allow users to enter clinical and lifestyle parameters for risk assessment.
6. Maintain prediction history for reviewing previous assessments.
7. Generate downloadable reports containing assessment information.
8. Integrate frontend, backend, machine learning, and explainability components into a functional application.

## 5. Technologies and Tools Used

| Category | Technology | Purpose |
|---|---|---|
| Frontend | React.js | User interface development |
| Build Tool | Vite | Frontend development and build |
| Backend | Python, Django | Application logic and model integration |
| Machine Learning | Scikit-learn | Model training and prediction |
| ML Algorithm | Logistic Regression | 10-year CHD risk estimation |
| Data Processing | Pandas, NumPy | Data preprocessing and numerical operations |
| Explainable AI | SHAP, LIME | Individual prediction explanations |
| Database | Relational database | User and assessment record management |
| Dataset | Framingham Heart Study | Training and evaluating the prediction model |

### Machine Learning Model Evaluation

Four algorithms were compared using out-of-fold evaluation.

| Model | OOF PR-AUC | OOF Brier Score | OOF ROC-AUC |
|---|---:|---:|---:|
| **Logistic Regression** | **0.3503** | **0.1159** | **0.7300** |
| Random Forest (Tuned) | 0.3299 | 0.1172 | 0.7236 |
| Gradient Boosting (Tuned) | 0.3261 | 0.1178 | 0.7164 |
| LightGBM (Tuned) | 0.3328 | 0.1174 | 0.7166 |

Logistic Regression was selected as the final prediction model.

Its held-out test ROC-AUC was **0.6993**, with a PR-AUC of **0.2903**.

## 6. Key Features and Functionalities

### 6.1 Landing Page

Provides an introduction to CardioXAI, its purpose, and navigation to authentication and other application features.

### 6.2 User Authentication

Supports user registration and login, allowing individuals to access personalized features and their own assessment records.

### 6.3 User Dashboard

Provides a central interface for accessing risk prediction, assessment history, and related functionalities.

### 6.4 Heart Disease Risk Prediction

Allows users to enter clinical and lifestyle parameters such as:

- Age and sex
- Cigarettes smoked per day
- Blood pressure medication status
- History of stroke and hypertension
- Diabetes status
- Total cholesterol
- Systolic and diastolic blood pressure
- Body Mass Index (BMI)
- Heart rate and glucose level

The submitted information is processed using the trained Logistic Regression model to estimate 10-year CHD risk.

### 6.5 Explainable AI Results

Integrates SHAP and LIME to explain how different input parameters contribute to the model's prediction.

### 6.6 Prediction History

Maintains previous assessment records, allowing authenticated users to review their earlier risk predictions.

### 6.7 Downloadable Reports

Allows users to download assessment reports containing relevant clinical information and prediction results.

## 7. System Requirements

### Hardware Requirements

| Component | Recommended |
|---|---|
| Processor | Intel Core i5 or equivalent |
| RAM | 8 GB or higher |
| Storage | 5 GB available space |
| Internet | Stable connection for network-dependent features |

### Software Requirements

- Python 3.x
- Node.js and npm
- Django and project-specific Python dependencies
- React.js and Vite dependencies
- Git
- A modern web browser
- A code editor such as Visual Studio Code
- Access to the configured application database

## 8. Installation Instructions

The following commands assume a repository containing separate frontend and backend directories. Adjust directory names to match the actual project structure.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

### Step 2: Set Up the Backend

Navigate to the Django backend directory.

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up the Frontend

Open another terminal and navigate to the React frontend directory.

```bash
cd frontend
npm install
```

### Step 4: Configure Environment Variables

Create or configure the environment files required by the application.

Set the necessary backend configuration, database connection settings, frontend API endpoint, and authentication settings according to the project's actual configuration.

**Important:** Do not commit passwords, API keys, database credentials, or secret environment files to GitHub.

## 9. Steps to Run the Project

### Step 1: Start the Backend Server

Navigate to the backend directory, activate the virtual environment, and run:

```bash
python manage.py runserver
```

By default, Django's development server is accessible at:

`http://127.0.0.1:8000/`

### Step 2: Start the Frontend Server

Open another terminal, navigate to the frontend directory, and run:

```bash
npm run dev
```

Vite typically provides a local development URL such as:

`http://localhost:5173/`

Use the URL displayed in the terminal.

### Step 3: Access the Application

1. Open CardioXAI in a web browser.
2. Register or log in to an account.
3. Navigate to the prediction module.
4. Enter the required clinical and lifestyle parameters.
5. Submit the form to generate a 10-year CHD risk estimate.
6. View the prediction and SHAP/LIME explanations.
7. Access previous assessments through prediction history.
8. Download the assessment report if required.

## 10. Database Configuration

CardioXAI uses a relational database design consisting of two main tables: `AUTH_USERS` and `ASSESSMENTS`.

### AUTH_USERS Table

Stores user account information.

| Field | Data Type | Description |
|---|---|---|
| id | UUID | Primary Key |
| email | TEXT | Registered email |
| encrypted_password | TEXT | Stored password credential |
| created_at | TIMESTAMPTZ | Account creation timestamp |
| updated_at | TIMESTAMPTZ | Last update timestamp |

### ASSESSMENTS Table

Stores clinical and lifestyle inputs together with prediction results.

| Field | Description |
|---|---|
| id | Unique assessment identifier |
| user_id | Foreign Key referencing AUTH_USERS |
| created_at | Assessment timestamp |
| age, sex | Demographic information |
| cigs_per_day | Smoking information |
| bp_meds, prevalent_stroke, prevalent_hyp, diabetes | Clinical indicators |
| tot_chol, sys_bp, dia_bp | Cholesterol and blood pressure |
| bmi, heart_rate, glucose | Additional clinical measurements |
| prediction_probability | Estimated 10-year CHD probability |
| prediction_label | Low Risk or Elevated Risk |
| model_version | Prediction model version |
| threshold_used | Classification threshold |
| is_stub | Indicates a temporary assessment awaiting an ML score |

### Database Relationship

```text
AUTH_USERS (1)
     |
     | owns
     |
     v
ASSESSMENTS (N)
```

One user can have multiple assessments, while each assessment belongs to one user.

The `user_id` foreign key references `AUTH_USERS(id)` with `ON DELETE CASCADE`.

### Configuration Notes

- Configure the database connection using the project's actual environment settings.
- Ensure the required tables and access permissions are configured.
- Keep database credentials outside the source code.
- Restrict access to assessment records so users can access only their own data.

## 11. Screenshots of the Working Project

Insert actual screenshots from the CardioXAI application in the following sections.

### 11.1 Landing Page

[Insert Landing Page Screenshot]

### 11.2 Authentication Page

[Insert Login / Registration Screenshot]

### 11.3 User Dashboard

[Insert Dashboard Screenshot]

### 11.4 Heart Disease Prediction Form

[Insert Prediction Form Screenshot]

### 11.5 Prediction Results and Explainability

[Insert Risk Result and SHAP/LIME Screenshot]

### 11.6 Prediction History

[Insert Prediction History Screenshot]

### 11.7 Downloadable Assessment Report

[Insert Generated Report Screenshot]

For GitHub, store screenshots in an `assets/screenshots/` directory and reference them using relative Markdown image paths.

## 12. Team Members' Names and Enrollment Numbers

| Team Member | Enrollment Number |
|---|---|
| Unnati Mangal | BETN1AI25053 |
| Anshika Khandelwal | BETN1AI25001 |

**Programme:** B.Tech (AI & ML), Semester III

**Department:** SOCIS

**Institution:** ITM University, Gwalior

**Faculty Guide:** Dr. Pallavi Khatri

---

### Disclaimer

CardioXAI is an academic project intended for educational purposes and cardiovascular risk awareness. The generated predictions are statistical estimates based on the trained model and should not be interpreted as a medical diagnosis, treatment recommendation, or replacement for professional healthcare advice.

![LANDING PAGE](image-1.png)
![LANDING PAGE 2](image.png)
![AUTHENTICATION PAGE](image-2.png)
![USER DASHBOARD](image-3.png)
![SYMPTOMS INPUT PAGE](image-4.png)
![PAST ASSESSMENT PAGE](image-5.png)
![USER RESULT REPORT](image-6.png)
![USER RESULT REPORT](image-7.png)