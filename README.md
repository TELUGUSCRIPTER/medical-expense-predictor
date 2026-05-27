# 🏥 AI-Based Medical Expense & Health Insurance Prediction System

A machine learning-powered web application that predicts medical insurance charges based on user health parameters. Built with Python, Flask, and Scikit-learn.

---

## 📸 Screenshots

> *Application Form – Dark glassmorphism UI with input fields*

> *Prediction Result – Shows estimated cost in USD and INR*

---

## 🚀 Features

- **AI Prediction**: Uses Linear Regression to predict insurance costs
- **Dual Currency**: Shows results in both USD ($) and INR (₹)
- **Input Validation**: Client-side and server-side validation
- **Dataset Preview**: View sample data from the training dataset
- **Smoking Impact Alert**: Special warning for smokers
- **Modern UI**: Dark glassmorphism design with Tailwind CSS
- **Mobile Responsive**: Works on all screen sizes

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Backend Language |
| Flask | Web Framework |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Tailwind CSS | Frontend Styling |
| Pickle | Model Serialization |

---

## 📂 Project Structure

```
medical-expense-predictor/
├── dataset/
│   └── insurance.csv          # Training dataset
├── model/
│   ├── train_model.py         # ML training script
│   └── model.pkl              # Trained model (generated)
├── app/
│   ├── app.py                 # Flask application
│   └── templates/
│       └── index.html         # Frontend UI
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone / Download the Project
```bash
cd medical-expense-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the ML Model
```bash
python model/train_model.py
```
This will:
- Load the insurance.csv dataset
- Preprocess and encode categorical variables
- Train a Linear Regression model
- Save the model as `model/model.pkl`

### Step 4: Run the Flask App
```bash
python app/app.py
```

### Step 5: Open in Browser
```
http://127.0.0.1:5000
```

---

## 📊 How It Works

1. **Data Collection**: Uses the `insurance.csv` dataset with 700+ records
2. **Preprocessing**: Encodes categorical variables (sex, smoker, region) to numeric
3. **Training**: Splits data 80/20 and trains a Linear Regression model
4. **Prediction**: User inputs are processed and fed to the model
5. **Display**: Results shown in both USD and INR currencies

### Input Parameters

| Parameter | Type | Range |
|---|---|---|
| Age | Integer | 1 – 100 |
| Gender | Select | Male / Female |
| BMI | Float | 10 – 60 |
| Children | Integer | 0 – 10 |
| Smoker | Select | Yes / No |
| Region | Select | Southwest / Southeast / Northwest / Northeast |

---

## 📈 Model Performance

The Linear Regression model achieves:
- **R² Score**: ~75% (varies with random state)
- **Key Insight**: Smoking status is the strongest predictor of high insurance costs

---

## ⚠️ Disclaimer

This is an educational project. Predictions are based on historical data and should **not** be used for actual insurance pricing decisions.

---

## 👨‍💻 Author

College Project — AI-Based Medical Expense & Health Insurance Prediction System

---

## 📄 License

This project is for educational purposes only.

---

## 🚀 Deploying to Railway (quick)

This repository includes a `Procfile` so it can be deployed to Railway or any platform that supports Procfile-based startups.

What was added for deployment:
- `Procfile` — runs `gunicorn app.app:app` binding to `0.0.0.0:$PORT`
- `requirements.txt` — now includes `gunicorn`

Steps to deploy on Railway:

1. Push this repository to a Git provider (GitHub, GitLab).
2. Create a new Railway project and connect your repository.
3. Railway will install dependencies from `requirements.txt` and use the `Procfile` to start the web process.
4. Railway provides the `PORT` environment variable automatically — no changes required.

Notes:
- The repository contains trained models in `model/` (`rf_model.pkl`, `linear_model.pkl`, `comparison.json`). If you prefer to retrain on the server, run `python model/train_model.py` before starting the app (not necessary if models are present).

Local testing (Windows PowerShell):
```powershell
# install deps
pip install -r requirements.txt

# run locally (dev)
python app/app.py

# or run with gunicorn (requires gunicorn installed)
gunicorn app.app:app --bind 0.0.0.0:5000 --workers 2
```

If you want UI changes from the WhatsApp image you attached, please tell me what to change or paste the image notes — I couldn't open the binary image directly here but I can implement updates quickly once I know the desired changes.
