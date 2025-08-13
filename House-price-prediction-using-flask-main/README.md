# 🏠 House Price Prediction (Flask Web App)

A machine learning project that predicts house prices based on selected features using the **King County House Sales Dataset**.

## 📂 Project Structure
```
├── house.py          # Train models, select best, save as pickle
├── app.py            # Flask web app
├── templates/
│   └── index.html    # Web form with CSS styling
├── house_data.csv    # Dataset
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

## ⚙ Features Used
- Bedrooms
- Bathrooms
- Sqft Living
- Floors
- Waterfront
- View
- Condition
- Grade

## 🚀 How to Run

### 1️⃣ Clone the repo
```bash
git clone https://github.com/technical-beast-07/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model
```bash
python house.py
```

This will generate:
- `best_model.pkl`
- `scaler.pkl`

### 4️⃣ Run the web app
```bash
python app.py
```
Then visit **`http://127.0.0.1:5000`** in your browser.

## 📊 Sample Predictions
Example inputs for testing:
| Bedrooms | Bathrooms | Sqft Living | Floors | Waterfront | View | Condition | Grade |
|----------|-----------|-------------|--------|------------|------|-----------|-------|
| 3        | 2.0       | 1800        | 1.0    | 0          | 1    | 3         | 7     |
| 4        | 3.5       | 3500        | 2.0    | 1          | 4    | 4         | 10    |
| 2        | 1.0       | 900         | 1.0    | 0          | 0    | 3         | 6     |

## 📜 License
MIT License
