import emoji
import nltk
import re


def remove_spaces(text):
    text = text.strip()
    text = text.split()

    return " ".join(text)


def arabic_stopwords():
    stopwords = []
    with open('arabic_stopwords.txt') as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    nltk_stopwords = nltk.corpus.stopwords.words('arabic')
    stopwords.extend(nltk_stopwords)
    stopwords = [remove_diacritics(normalize_arabic(stopword))
                 for stopword in stopwords]

    return set(stopwords)


def remove_stopwords(text, stopwords):
    text = [token for token in text.split() if token not in stopwords]

    return " ".join(text)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(r'([ ])([و])([\w+])', r'\1 \2 \3',
                  text)  # "وبعدين" --> "و بعدين"

    return text


def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):

    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations

    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def tweet_cleanser(text):
    cleaning_re = "(?:\@|https?\://)\S+"

    text = re.sub(cleaning_re, '', text)  # remove mentions and urls
    text = re.sub("^RT[\s]+", '', text)  # remove RT
    text = re.sub(emoji.get_emoji_regexp(), '', text)  # remove emojis

    return text


def preprocess_pipeline(text):
    stopwords = arabic_stopwords()

    text = tweet_cleanser(text)
    # remove stopwords before normalizing text
    text = remove_stopwords(text, stopwords)
    text = normalize_arabic(text)
    text = remove_punctuations(text)
    text = remove_diacritics(text)
    text = remove_repeating_char(text)
    # remove stopwords after normalizing text
    text = remove_stopwords(text, stopwords)
    text = remove_spaces(text)

    return text
