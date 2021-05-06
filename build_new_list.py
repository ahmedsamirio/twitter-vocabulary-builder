preprocess_tweets(conn, preprocess_pipeline, stopwords)

profanities_db = pd.read_csv('Arabic.csv', header=None)
profanities_set = set(profanities_db[0].to_list())

_ = label_users(conn, profanities_set)
negative_tweets_count, positive_tweets_count = label_tweets(conn, profanities_set)

obscene_n_grams, non_obscene_n_grams = make_n_grams(conn)

n_gram_LOR = calculate_LOR(obscene_n_grams, non_obscene_n_grams,
                           negative_tweets_count, positive_tweets_count)