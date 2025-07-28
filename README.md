
# 🧠 Emotion Detection API – FastAPI + ML

A lightweight ML-powered API that predicts the **emotion** behind a piece of text using a trained logistic regression model. Emotions include:

```

😢 Sadness | 😄 Joy | ❤️ Love | 😡 Anger | 😱 Fear | 😲 Surprise

````

---

## 📌 What This Project Does

This project:

- Trains a model on labeled emotional text (e.g., joy, sadness, anger)
- Saves the trained model using `joblib`
- Serves predictions via a FastAPI REST API
- Accepts raw text input and returns the predicted emotion

---

## 🔧 Tech Stack

| Layer        | Tool                   |
|--------------|------------------------|
| Language     | Python                 |
| ML Library   | scikit-learn           |
| API Server   | FastAPI + Uvicorn      |
| Model Saving | joblib                 |
| Data Format  | CSV (train/val/test)   |

---

## 🚀 How It Works

### 🏗️ 1. Dataset Structure
Each CSV should look like this:

```csv
text,label
"I am feeling happy",1
"I want to cry",0
...
````

Where:

* `text` is the sentence
* `label` is a number (0 to 5), mapped to emotions:

  * 0: sadness
  * 1: joy
  * 2: love
  * 3: anger
  * 4: fear
  * 5: surprise

---

### 🧠 2. Train & Save Model

```bash
python train_save_model.py
```

* Trains on `dataset/training.csv` + `validation.csv`
* Saves `emotion_model.joblib`

---

### ⚙️ 3. Start FastAPI Server

```bash
uvicorn main:app --reload
```

* Visit: [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI
* Endpoint: `POST /predict`

---

### 📬 Example API Request

#### `POST /predict`

```json
{
  "text": "You're like semicolons in my code — I don’t always need you, but when I forget you, everything falls apart ❤️"
}
```

#### Response:
```json
{
  "text": "You're like semicolons in my code — I don’t always need you, but when I forget you, everything falls apart ❤️",
  "predicted_label": 2,
  "emotion": "love"
}
```
####  other examples
```
"I just segfaulted my heart thinking about you 🧠💔"
"I’m not crying, you’re leaking memory 😭"
"If we were variables, we’d have perfect type compatibility 💘"
"My love for you is like an infinite while loop 😳"
"I'm feeling more lost than a missing semicolon 😵‍💫"
```

---

## 🧪 Testing Your Model

Use `predict("your sentence here")` in `main.ipynb` or test via cURL/Postman/Swagger.

---

## 📁 Project Structure

```
emotion-detector/
├── dataset/
│   ├── training.csv
│   ├── validation.csv
│   └── test.csv
├── train_save_model.py
├── emotion_model.joblib
├── app.py ------------------------> fastapi 
├── main.py and main.ipynb -----------> our model training files
├── .gitignore
└── README.md
```

---

## 📌 Future Improvements

* ✅ Add `Gradio` web interface
* ⏳ Deploy on HuggingFace Spaces / Render / Vercel
* 🌍 Add language support for Nepali, Hindi, etc.
* 🧪 Confidence score or probabilities in response

---

## 📜 License

MIT – do whatever you want 🚀

---

## 🧔 Author

Made with ❤️ by **Pujan Neupane**

