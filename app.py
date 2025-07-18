from flask import Flask, request, render_template
import joblib
import openai
import os
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("career_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load career data
df = pd.read_csv("career_data.csv")
df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == 'object' else col)

# Preload unique streams
streams = sorted(df['stream'].dropna().unique())

# Group data by stream for dependent dropdowns
grouped_data = df.groupby('stream')

def get_options_for_stream(stream):
    stream = stream.lower().strip()
    if stream in grouped_data.groups:
        group = grouped_data.get_group(stream)
        return {
            'subjects': sorted(group['subject_liked'].dropna().unique()),
            'skills': sorted(group['skills'].dropna().unique()),
            'soft_skills': sorted(group['soft_skill'].dropna().unique()),
            'preferred_fields': sorted(group['preferred_field'].dropna().unique())
        }
    else:
        return {
            'subjects': [],
            'skills': [],
            'soft_skills': [],
            'preferred_fields': []
        }

@app.route('/')
def home():
    return render_template("index.html", streams=streams, suggestions={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fields = ["stream", "subject_liked", "skills", "soft_skill", "preferred_field"]
        inputs = [request.form.get(f, "").strip().lower() for f in fields]

        if not all(inputs):
            suggestions = get_options_for_stream(inputs[0]) if inputs[0] else {}
            return render_template("index.html", streams=streams, suggestions=suggestions, result="Please fill all fields.")

        encoded = []
        for field, val in zip(fields, inputs):
            if val in encoders[field].classes_:
                encoded.append(encoders[field].transform([val])[0])
            else:
                raise ValueError(f"Invalid input: {val}")

        prediction = model.predict([encoded])[0]
        result = target_encoder.inverse_transform([prediction])[0]

    except Exception as e:
        print("Error:", e)
        try:
            prompt = f"""Suggest suitable careers for:
- Stream: {inputs[0]}
- Subject Liked: {inputs[1]}
- Technical Skills: {inputs[2]}
- Soft Skills: {inputs[3]}
- Preferred Field: {inputs[4]}"""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            result = "Model couldn't predict. Here's an AI suggestion:\n\n" + response['choices'][0]['message']['content']
        except:
            result = "Sorry, we couldn't process your input."

    suggestions = get_options_for_stream(inputs[0])
    return render_template("index.html", streams=streams, suggestions=suggestions, result=result)

@app.route("/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip()
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        answer = response['choices'][0]['message']['content']
    except:
        answer = "Sorry, I couldn't answer that."
    return render_template("index.html", streams=streams, suggestions={}, chat_answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)