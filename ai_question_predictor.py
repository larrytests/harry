import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load previous question papers and syllabus datasets
previous_question_papers = pd.read_csv('previous_question_papers.csv')
syllabus = pd.read_csv('syllabus.csv')

# Preprocess text data (you can expand this function as needed)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace('[^\w\s]', '')  # Remove punctuation
    return text

# Apply preprocessing to the datasets
previous_question_papers['question_cleaned'] = previous_question_papers['question'].apply(preprocess_text)
syllabus['content_cleaned'] = syllabus['content'].apply(preprocess_text)

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english')
previous_papers_matrix = vectorizer.fit_transform(previous_question_papers['question_cleaned'])
syllabus_matrix = vectorizer.transform(syllabus['content_cleaned'])

# Train LDA model on the previous papers
lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(previous_papers_matrix)

# Function to predict the next likely topic based on previous exam patterns
def predict_next_topic(previous_papers, syllabus_topics, lda_model, vectorizer):
    # Analyze patterns in previous exam papers to predict the next topic
    previous_topics_distributions = lda_model.transform(vectorizer.transform(previous_papers['question_cleaned']))
    
    # Average topic distribution across all previous questions
    avg_topic_distribution = np.mean(previous_topics_distributions, axis=0)
    
    # Compare this distribution with the syllabus topics to predict the next likely topic
    syllabus_topics_distributions = lda_model.transform(vectorizer.transform(syllabus_topics['content_cleaned']))
    
    # Compute similarity between average distribution and syllabus topics
    similarity_scores = cosine_similarity([avg_topic_distribution], syllabus_topics_distributions)
    
    # Select the syllabus topic with the highest similarity score as the predicted topic
    best_match_idx = similarity_scores.argmax()
    predicted_topic = syllabus_topics['content_cleaned'].iloc[best_match_idx]
    
    return predicted_topic

# Initialize the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Function to generate a question from the predicted topic
def generate_question_from_predicted_topic(previous_papers, syllabus, lda_model, vectorizer, model, tokenizer):
    # Predict the topic based on the analysis of previous papers
    predicted_topic_description = predict_next_topic(previous_papers, syllabus, lda_model, vectorizer)
    
    # Generate a question based on the predicted topic
    input_text = f"Generate an exam question on: {predicted_topic_description}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return question

# Generate an exam question based on the predicted topic
exam_paper = generate_question_from_predicted_topic(previous_question_papers, syllabus, lda_model, vectorizer, model, tokenizer)
print("Generated Question:", exam_paper)
