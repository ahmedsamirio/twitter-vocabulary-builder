from preprocessing import *

import os, sqlite3


def create_db(db_name):
    """
    Creates an sqlite database or returns a database if one exists.

    Args:
    1. db_name (str)

    Return:
    1. Sqlite connect object
    """

    if db_name not in os.listdir():
    
        # Create SQL database then create tweets and users tables
        conn = sqlite3.connect(db_name)
        c = conn.cursor()

        c.execute('''CREATE TABLE tweets (text text, id real, user_id text)''')  # Create tweets table
        c.execute('''CREATE TABLE users (screen_name text, name text, id real,  description text, followers_count real,\
                    friends_count real, statuses count real, created_at text)''')  # Create users table
        conn.commit()
    else:
        conn = sqlite3.connect(db_name)  # Open SQL database
        c = conn.cursor()

    return conn, c



def preprocess_tweets(db, pipeline):
    """
    Preprocesses tweets for a given user in tweets sqlite database.
    
    Args:
    1. db (sqlite database): sqlite database connect object.
    2. pipeline (func): function for text preprocessing.
    
    Return: None
    """
    
    user_c = db.cursor()  # cursor for iteration over users table
    tweet_c = db.cursor()  # cursor for iterating over a user tweets in tweets table
    update_c = db.cursor()  # cursor for updating the tweets table with preprocessed text
    
    # Make new preprocessed_text column if it doesn't exist
    try: 
        update_c.execute('ALTER TABLE tweets ADD preprocessed_text text')

    except:
        pass
    
    for user_id, in user_c.execute('SELECT DISTINCT id FROM users'):
        for text, tweet_id in tweet_c.execute('SELECT text, tweet_id FROM tweets WHERE user_id=?',
                                                     (user_id)):
            preprocessed_tweet = pipeline(text)
            data = (preprocessed_tweet, tweet_id)
            update_c.execute('UPDATE tweets set preprocessed_text = ? where tweet_id = ?', data)
            
    # inactivate cursors
    user_c.close()
    tweet_c.close()  
    update_c.close()

    db.commit()



def label_users(db, words_set):
    """
    Labels users for using a word in their tweets from a provided list.

    Args:
    1. db (sqlite database): sqlite database connect object.
    2. words_list (list): a list of words to run users' tweets on.

    Return: None
    """

    user_c = db.cursor()  # cursor for iteration over users table
    tweet_c = db.cursor()  # cursor for iterating over a user tweets in tweets table
    update_c = db.cursor()  # cursor for updating the tweets table with preprocessed text

    # Make new label column if it doesn't exist
    try: 
        update_c.execute('ALTER TABLE users ADD label real')
        update_c.execute('ALTER TABLE users ADD words_used text')

    except:
        pass
    
    for user_id, in user_c.execute('SELECT DISTINCT id FROM users'):
        user_corpus = {}

        for text, in tweet_c.execute('SELECT preprocessed_text FROM tweets WHERE user_id=?', 
                                           (user_id,)):
            tokens = text.split()
            for token in tokens:

                if token in user_corpus.keys():
                    user_corpus[token] += 1

                else:
                    user_corpus[token] = 1

        
        if words_set.intersection(user_corpus):
            words_used = words_set.intersection(user_corpus)
            update_c.execute('UPDATE users SET label = ?, words_used = ? WHERE screen_name = ?',
                              (1, ','.join(words_used), screen_name))

        else:
            update_c.execute('UPDATE users SET label = ? WHERE screen_name = ?',
                              (0, screen_name))
              
    # inactivate cursors
    user_c.close()
    tweet_c.close()  
    update_c.close()
    
    db.commit()



def label_tweets(db, words_set):
    """
    Labels tweets for containing a word from a provided list.
    
    Args:
    1. db (sqlite database): sqlite database connect object.
    2. words_list (list): a list of words to run users' tweets on.
    
    Return: None
    """
    tweet_c = db.cursor()  # cursor for iterating over a user tweets in tweets table
    update_c = db.cursor()  # cursor for updating the tweets table with preprocessed text
    
    # Make new label column if it doesn't exist
    try: 
        update_c.execute('ALTER TABLE tweets ADD label real')
        update_c.execute('ALTER TABLE tweets ADD words_used text')

    except:
        pass
    

    for text, tweet_id in tweet_c.execute('SELECT preprocessed_text, tweet_id FROM tweets'):
        tokens = text.split()

        if words_set.intersection(tokens):
            words_used = words_set.intersection(tokens)
            update_c.execute('UPDATE tweets SET label = ?, words_used = ? WHERE tweet_id = ?',
                              (1, ','.join(words_used), tweet_id))
                            
        else:
            update_c.execute('UPDATE tweets SET label = ? WHERE tweet_id = ?',
                              (0, tweet_id))
                              
    # inactivate cursors
    tweet_c.close()  
    update_c.close()
    
    db.commit()
    


def tweets_counter(db):
    """
    Counts all tweets, positive and negative tweets.

    Args:
    1. db (sqlite database): sqlite database connect object.

    Returns:
    1. Tweets count
    2. Negative tweets count
    3. Positive tweets count
    """

    tweet_c = db.cursor()  # cursor for iterating over tweets in tweets table

    tweets_count = 0
    postitive_tweets_count = 0
    negative_tweets_count = 0

    for label, in tweet_c.execute('SELECT label FROM tweets'):
        tweets_count += 1

        if label == 1:
            positive_tweets_count += 1

        else:
            negative_tweets_count += 1

    return tweets_count, positive_tweets_count, negative_tweets_count


def make_n_grams(db):
    """
    Makes n-grams for positive and negative users.
    
    Args:
    1. db (sqlite database): Sqlite database containing users and tweets table.
    
    Returns: 
    1. A set of positive users n-grams.
    2. A set of negative users n-grams.
    """

    user_c = db.cursor()  # cursor for iteration over users table
    tweet_c = db.cursor()  # cursor for iterating over a user tweets in tweets table
    update_c = db.cursor()  # cursor for updating the tweets table with preprocessed text
    
    positive_n_grams = {}
    negative_n_grams = {}
    
    for user_id, label in user_c.execute('SELECT DISTINCT id, label FROM users'):
        for text, in tweet_c.execute('SELECT preprocessed_text FROM tweets WHERE user_id=?', 
                                           (user_id,)):
            text_n_grams = [item for n in range(1, 3) for item in nltk.ngrams(text.split(), n)]

            for n_gram in text_n_grams:

                if label == 1:
                    positive_n_grams[n_gram] = positive_n_grams.get(n_gram, 0) + 1

                else:
                    negative_n_grams[n_gram] = negative_n_grams.get(n_gram, 0) + 1
                
    # inactivate cursors
    user_c.close()
    tweet_c.close()  
    update_c.close()
    
    db.commit()
    
    return obscene_n_grams, non_obscene_n_grams


def calculate_LOR(positive_n_grams, negative_n_grams, positive_count, negative_count):
    """
    Calculate log odds ratio for a n-grams provided in two dictionaires.
    
    Args:
        1. positive_n_grams (dict): A dictionary containing counts of n_grams in positive users.
        2. negative_n_grams (dict): A dictionary containing counts of n_grams in negative users.
        3. positive_count (int)
        4. negative_count (int)
        
    Returns:
        1. A dictionary containing log odds ratio of all n_grams
    """
    
    all_n_grams = set(positive_n_grams.keys()).union(negative_n_grams.keys())
    
    positive_n_grams = {n_gram: count for n_gram, count in neg_n_grams.items() if count >= 10}
    negative_n_grams = {n_gram: count for n_gram, count in pos_n_grams.items() if count >= 10}

    n_gram_LOR = {}
    
    for n_gram in all_n_grams:
        numerator = positive_n_grams.get(n_gram, 0) * (negative_count - positive_n_grams.get(n_gram, 0))
        denominator = negative_n_grams.get(n_gram, 0) * (positive_count - negative_n_grams.get(n_gram, 0)) + 0.01
        n_gram_LOR[n_gram] = np.log(numerator/denominator)
        
    return n_gram_LOR        
    
