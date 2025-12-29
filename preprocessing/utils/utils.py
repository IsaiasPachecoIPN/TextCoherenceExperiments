import spacy

nlp = spacy.load('en_core_web_sm')

def get_reasons(story_id,reasons_data, sentences, score):
    
    print(f"Processing story {story_id}...")

    obj_sentences = {}

    for i in range(len(sentences)):
        obj_sentences[sentences[i].lower().strip()] = [0,0,0,0,0,0,0, score]
    
    for anotator in reasons_data:
        for sentence_id in reasons_data[anotator]:
            for reason in reasons_data[anotator][sentence_id]:
                _old_array = obj_sentences[sentences[int(sentence_id)-1].lower().strip()]
                _old_array[reason-1] = 1

    return obj_sentences

def preprocess_sentences(sentence, lower=False, remove_puntuation=False, remove_stopwords=False, lemmatize=False):

    english_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    sentence = sentence.strip()

    if lower:
        sentence = sentence.lower()

    if remove_puntuation:
        sentence = sentence.translate(str.maketrans('', '', english_punctuation))

    if remove_stopwords:
        f = open("../utils/stopwords.txt", "r").read().split("\n")
        stopwords = [elem.strip() for elem in f]
        sentence = ' '.join([word for word in sentence.split() if word not in stopwords])

    if lemmatize:
        doc = nlp(sentence)
        sentence = ' '.join([token.lemma_ for token in doc])

    if sentence == '':
        sentence = 'REMOVE'
    
    return sentence

def get_score( elem_score):
    if elem_score > 3:
        return 1
    elif elem_score < 3:
        return 0
    else:
        return -1