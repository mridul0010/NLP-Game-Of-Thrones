import joblib

def run_lda(tokens, num_topics):

    lda = joblib.load("models/lda_model.joblib")

    topics = lda.print_topics(num_topics=num_topics)

    return topics
