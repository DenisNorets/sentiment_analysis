import re
from string import punctuation
import numpy as np
SEQ_LEN = 35


def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = text.translate(str.maketrans('', '', f'{punctuation + "’»«"}'))
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return str(text).strip()


def remove_stopword(words, stopwords_list):
    custom_list = ["im", "like", "get"]
    clean_words = [word for word in words if (word not in stopwords_list and word not in custom_list) and len(word) > 1]
    return " ".join(clean_words)


def tokenize(texts_list, tokenizer):
    input_ids = []
    attention_masks = []

    for i, text in enumerate(texts_list):
        tokens = tokenizer.encode_plus(text, max_length=SEQ_LEN,
                                       truncation=True, padding='max_length',
                                       add_special_tokens=True, return_attention_mask=True,
                                       return_token_type_ids=False, return_tensors='tf')

        input_ids.append(np.asarray(tokens["input_ids"]).reshape(SEQ_LEN, ))
        attention_masks.append(np.asarray(tokens["attention_mask"]).reshape(SEQ_LEN, ))

    return np.asarray(input_ids), np.asarray(attention_masks)


def predict_on_sentence(text, model, stopwords_list, tokenizer, roberta_model):
    sentiments = ['approval', 'curiosity', 'neutral', 'nervousness', 'surprise', 'sadness', 'fear']
    text = clean_text(text)
    text = remove_stopword(text.split(), stopwords_list)
    train_ids, train_attention_masks = tokenize([text], tokenizer)
    embedding = roberta_model(train_ids, attention_mask=train_attention_masks)[0]

    predict = model.predict(embedding)
    indexes = predict[0].round()

    predicted_emojies = ''
    if sum(indexes) > 0:
        for index, prediction in enumerate(indexes):
            if int(prediction) == 1:
                predicted_emojies += f'{sentiments[index]}, '
        predicted_emojies = predicted_emojies[:-2]
    else:
        sentiment_index = np.argmax(predict)
        predicted_emojies += f'{sentiments[sentiment_index]} (lower than trashhold)'

    return predicted_emojies
