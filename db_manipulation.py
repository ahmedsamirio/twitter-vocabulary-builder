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
        conn = sqlite3.connect(db_name)  # Create SQL database
        c = conn.cursor()  # Create a database cursor
        c.execute('''CREATE TABLE tweets (text text, id real, user_id text)''')  # Create tweets table
        c.execute('''CREATE TABLE users (screen_name text, name text, id real,  description text, followers_count real,\
                    friends_count real, statuses count real, created_at text)''')  # Create users table
        conn.commit()
    else:
        conn = sqlite3.connect(db_name)  # Open SQL database
        c = conn.cursor()
    return conn, c


def inactivate_cursors(cursors):
    for cursor in cursors:
        cursor.close()


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
    update_c = db.cursor()  # cursor for updating the tweets table with processed text
    
    add_new_column(update_c, "tweets", "processed_text", "text")
    
    for user_id, in get_distinct_table_columns(user_c, "users", ["id"]):
        for text, tweet_id in get_user_tweets(tweet_c, user_id, processed=False):                                                     (user_id)):
            processed_tweet = pipeline(text)
            data = (processed_tweet, tweet_id)
            update_tweet_processed_text(update_c, data)
            
    inactivate_cursors([users_c, tweet_c, update_c])
    db.commit()


def update_tweet_processed_tweet(cursor, data):
    cursor.execute('UPDATE tweets set processed_text = ? where tweet_id = ?', data)



def get_distinct_table_columns(cursor, table, columns):
    cursor.execute('SELECT DISTINCT {} FROM {}'.format(', '.join(columns), table))


def get_table_columns(cursor, table, columns):
    cursor.execute('SELECT {} FROM {}'.format(', '.join(columns), table))


def get_user_tweets(cursor, user_id, processed=True):
    if processed:
        columns = ["processed_text"]
    else:
        columns = ["text", "tweet_id"]
    cursor.execute('SELECT {} FROM tweets WHERE user_id=?'.format(', '.join(columns)), (user_id,))


def make_user_corpus(user_id, cursor):
    user_corpus = {}
    for text, in cursor.execute('SELECT processed_text FROM tweets WHERE user_id=?', (user_id,)):
        tokens = text.split()
        for token in tokens:
            if token in user_corpus.keys():
                user_corpus[token] += 1
            else:
                user_corpus[token] = 1
    return user_corpus


def add_new_column(cursor, table, column, type):
    try: 
        cursor.execute('ALTER TABLE {0} ADD {1} {2}}'.format(table, column, type))
    except:
        pass


def update_table_label(word_set, words, cursor, table):
    # if the tokens contains words mentioned in the seed word set label the data with 1 else 0
    if words_set.intersection(words):
        words_used = words_set.intersection(words)
        cursor.execute('UPDATE {} SET label = ?, words_used = ? WHERE screen_name = ?'.format(table),
                          (1, ','.join(words_used), screen_name))
    else:
        cursor.execute('UPDATE {} SET label = ? WHERE screen_name = ?'.format(table),
                          (0, screen_name))


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
    update_c = db.cursor()  # cursor for updating the tweets table with processed text

    add_new_column(update_c, "users", "label", "real")
    add_new_column(update_c, "users", "words_used", "text")
    
    for user_id, in get_distinct_table_columns(user_c, "users", ["id"]):
        user_corpus = make_user_corpus(user_id, tweet_c)
        update_table_label(word_set, user_corpus, update_c, "users")
               
    inactivate_cursors([users_c, tweet_c, update_c])
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
    update_c = db.cursor()  # cursor for updating the tweets table with processed text
    
    add_new_column(update_c, "tweets", "label", "real")
    add_new_column(update_c, "tweets", "words_used", "text")

    for text, tweet_id in get_table_columns(tweet_c, "tweets", ["processed_text", "tweet_id"]):
        tokens = text.split()
        update_table_label(word_set, tokens, update_c, "tweets")
                              
    inactivate_cursors([tweet_c, update_c])
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

    for label, in get_table_columns(tweet_c, "tweets", ["label"]):
            tweets_count += 1
        if label == 1:
            positive_tweets_count += 1
        else:
            negative_tweets_count += 1

    return tweets_count, positive_tweets_count, negative_tweets_count


def make_n_grams(text):
    n_grams = []
    for n in range(1, 3):
        for item in nltk.ngrams(text.split(), n):
            n_grams.append(item)
    return n_grams


def get_n_grams(db):
    """
    Returns n-grams for positive and negative users.
    
    Args:
    1. db (sqlite database): Sqlite database containing users and tweets table.
    
    Returns: 
    1. A set of positive users n-grams.
    2. A set of negative users n-grams.
    """
    user_c = db.cursor()  # cursor for iteration over users table
    tweet_c = db.cursor()  # cursor for iterating over a user tweets in tweets table
    update_c = db.cursor()  # cursor for updating the tweets table with processed text
    
    positive_n_grams = {}
    negative_n_grams = {}
    
    for user_id, label in get_distinct_table_columns(user_c, "users", ["id", "label"]):
        for text, in get_user_tweets(tweet_c, user_id):
            text_n_grams = make_n_grams(text)
            for n_gram in text_n_grams:
                if label == 1:
                    positive_n_grams[n_gram] = positive_n_grams.get(n_gram, 0) + 1
                else:
                    negative_n_grams[n_gram] = negative_n_grams.get(n_gram, 0) + 1
                
    inactivate_cursors([users_c, tweet_c, update_c])
    db.commit()
    
    return positive_n_grams, negative_n_grams


def calculate_n_gram_LOR(n_gram, positive_n_grams, negative_ngrams, positive_count, negative_count):
    numerator = positive_n_grams.get(n_gram, 0) * (negative_count - positive_n_grams.get(n_gram, 0))
    denominator = negative_n_grams.get(n_gram, 0) * (positive_count - negative_n_grams.get(n_gram, 0)) + 0.01
    return np.log(numerator/denominator)


def filter_n_grams_dict(n_grams_dict, limit):
    return {n_gram: count for n_gram, count in n_gram_dict.items() if count >= limit}


def get_n_grams_LOR(positive_n_grams, negative_n_grams, positive_count, negative_count):
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
    
    positive_n_grams = filter_n_gram_dict(positive_n_grams, 10)
    negative_n_grams = filter_n_gram_dict(negative_n_grams, 10)

    n_gram_LOR = {} 
    for n_gram in all_n_grams:
        n_gram_LOR[n_gram] = calculate_n_gram_LOR(n_gram, positive_n_grams, negative_n_grams, positive_count, negative_count)
        
    return n_gram_LOR        
    
