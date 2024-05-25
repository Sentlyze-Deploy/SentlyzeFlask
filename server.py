from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Stopwords'i indir ve ayarla
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish') + stopwords.words('english'))

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum istek boyutunu 16MB olarak ayarla

def suppress_tf_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow loglarını bastır
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

suppress_tf_logs()

# Model ve tokenizer'ı yükle
model_path = '/home/site/wwwroot/GRU_sentiment_model.h5'
tokenizer_path = '/home/site/wwwroot/tokenizer_turkish.pickle'
model = None
tokenizer = None

try:
    model = load_model(model_path)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")

try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer başarıyla yüklendi.")
except Exception as e:
    print(f"Tokenizer yüklenirken hata oluştu: {e}")

def predict_sentiment_batch(texts, tokenizer, model, max_tokens=64):
    tokens = tokenizer.texts_to_sequences(texts)
    tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
    predictions = model.predict(tokens_pad)
    return [float(pred[0]) for pred in predictions]

def get_most_frequent_words(comments, n=3):
    combined_text = ' '.join(comments)
    words = re.findall(r'\b\w+\b', combined_text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    counter = Counter(filtered_words)
    most_common_words = counter.most_common(n)
    return [word for word, count in most_common_words]

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model veya tokenizer yüklenmedi'}), 500

    try:
        data = request.json
        if 'comments' not in data:
            return jsonify({'error': 'Yorumlar sağlanmadı'}), 400

        comments = data['comments']
        predictions = predict_sentiment_batch(comments, tokenizer, model)

        # Ortalama tahmini hesapla
        if predictions:
            average_prediction = sum(predictions) / len(predictions)
        else:
            average_prediction = 0

        most_frequent_words = get_most_frequent_words(comments)

        return jsonify({
            'predictions': predictions,
            'average_prediction': average_prediction,
            'most_frequent_words': most_frequent_words
        })
    except Exception as e:
        print(f"Tahmin sırasında hata oluştu: {e}")
        return jsonify({'error': 'Dahili Sunucu Hatası', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Azure App Service için
