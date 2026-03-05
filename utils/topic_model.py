from gensim import corpora, models

def run_lda(tokens, num_topics):

    dictionary = corpora.Dictionary([tokens[:40000]])

    corpus = [dictionary.doc2bow(tokens[:40000])]

    lda = models.LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10
    )

    return lda.print_topics()