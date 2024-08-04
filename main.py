from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')

def classify_emotion(text):
    result = classifier(text)
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['journal_entry']
    result = classify_emotion(text)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
