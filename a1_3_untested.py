import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load previous question papers dataset
previous_question_papers = pd.read_csv('previous_question_papers.csv')

# Preprocess text data
def preprocess_text(text):
    text = text.lower().replace('[^\w\s]', '')  # Convert to lowercase and remove punctuation
    return text

# Apply preprocessing
previous_question_papers['question_cleaned'] = previous_question_papers['Question'].apply(preprocess_text)
previous_question_papers['subject_cleaned'] = previous_question_papers['Subject'].apply(preprocess_text)
previous_question_papers['exam_type_cleaned'] = previous_question_papers['Exam Type'].apply(preprocess_text)
previous_question_papers['combined_features'] = (
    previous_question_papers['exam_type_cleaned'] + ' ' +
    previous_question_papers['subject_cleaned'] + ' ' +
    previous_question_papers['Year'].astype(str) + ' ' +
    previous_question_papers['question_cleaned']
)

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
previous_papers_matrix = vectorizer.fit_transform(previous_question_papers['combined_features'])

# Train LDA model on the combined features
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(previous_papers_matrix)

# Prepare data for LSTM model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(previous_question_papers['combined_features'])
sequences = tokenizer.texts_to_sequences(previous_question_papers['combined_features'])
max_sequence_length = max([len(seq) for seq in sequences])
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
y = np.roll(sequences, -1, axis=0)  # Shift for next question prediction
y = y[:-1]
X = X[:-1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and compile Bidirectional LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=256, input_length=max_sequence_length))
lstm_model.add(Bidirectional(LSTM(128, return_sequences=True)))
lstm_model.add(Dropout(0.3))
lstm_model.add(Bidirectional(LSTM(64)))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')
]

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks)

# Evaluate the LSTM model
y_pred = lstm_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("LSTM Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_classes, y_pred_classes):.2f}")
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Function to predict the next likely topic based on LSTM model
def predict_next_topic_lstm(previous_papers, lstm_model, tokenizer, max_sequence_length):
    last_sequence = previous_papers[-1].reshape(1, max_sequence_length)
    predicted_sequence_idx = lstm_model.predict(last_sequence).argmax(axis=1)[0]
    predicted_topic = tokenizer.index_word[predicted_sequence_idx]
    return predicted_topic

# Function to generate a question from the predicted topic
def generate_question_from_predicted_topic(previous_papers, lda_model, vectorizer, lstm_model, tokenizer, max_sequence_length, t5_model, t5_tokenizer):
    predicted_topic_description = predict_next_topic_lstm(previous_papers, lstm_model, tokenizer, max_sequence_length)
    input_text = f"Generate an exam question on: {predicted_topic_description}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")
    outputs = t5_model.generate(input_ids)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Initialize the T5 model and tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Generate an exam paper with 10 questions
exam_paper = []
for _ in range(10):
    question = generate_question_from_predicted_topic(X, lda_model, vectorizer, lstm_model, tokenizer, max_sequence_length, t5_model, t5_tokenizer)
    exam_paper.append(question)

# Print the generated exam paper
for idx, question in enumerate(exam_paper, start=1):
    print(f"Question {idx}: {question}")

# Feedback Loop: Collect and process feedback
# This is a placeholder for collecting user feedback
def collect_feedback(generated_questions):
    # Simulate feedback collection
    feedback = ["Good", "Needs more details", "Accurate", "Incorrect"]
    return feedback

feedback = collect_feedback(exam_paper)
print("Feedback received:", feedback)

# Adjust model based on feedback (Placeholder for actual implementation)
def adjust_model_based_on_feedback(feedback, model):
    # Implement logic to adjust model based on feedback
    pass

adjust_model_based_on_feedback(feedback, lstm_model)
