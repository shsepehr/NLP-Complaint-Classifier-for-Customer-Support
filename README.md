# 🧠 NLP Complaint Classifier for Customer Support

This project is a Machine Learning & NLP system for classifying customer complaints into predefined categories using TF-IDF and Logistic Regression. It is designed for real-world customer support workflows to automate routing of support tickets.

## 🚀 Features
- Text cleaning & preprocessing (NLTK, stemming, stopword removal)
- TF-IDF Vectorization + Logistic Regression
- Predict complaint categories from raw text
- FastAPI REST endpoint for real-world integration
- Fully modular ML pipeline


## 📦 Installation
```bash
pip install -r requirements.txt
pip install -r requirements.txt
⚙️ Training the Model
bash
Copy code
python train.py
🔮 Predicting from Python
bash
Copy code
python predict.py
🌐 Running the API
bash
Copy code
uvicorn app.api:app --reload
Open API Docs:

arduino
Copy code
http://127.0.0.1:8000/docs
🏷 Example JSON Request
json
Copy code
{
  "text": "I was charged twice for my premium subscription"
}
📌 Output Example
json
Copy code
{
  "category": "Billing Issue"
}
📊 Categories to Detect
Category	Example Complaint
Billing Issue	"I was double charged"
Technical Problem	"Internet not working"
Cancellation Request	"I want to cancel my account"
Account Access	"I forgot my password"
Customer Service	"Support was rude to me"

🛠 Tech Stack
Python 3.10+

Scikit-Learn

NLTK

FastAPI

TF-IDF NLP Pipeline

📌 Note
Ensure Python <= 3.10 for maximum compatibility.

