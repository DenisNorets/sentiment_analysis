import argparse
from nltk.corpus import stopwords
from transformers import RobertaTokenizer, TFAutoModel
from models import bilstm_model
from functions import predict_on_sentence


def main():
    parser = argparse.ArgumentParser(prog='Image Standardization End-to-End Test')
    parser.add_argument('--sentence', type=str, required=True)

    model = bilstm_model()
    model.load_weights('checkpoints/bilstm_model/bilstm_model.hdf5')

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = TFAutoModel.from_pretrained("roberta-base")

    args = parser.parse_args()
    sentence = args.sentence

    stopwords_list = stopwords.words('english')

    result = predict_on_sentence(sentence, model, stopwords_list, tokenizer, roberta)

    print(f"Your input text: {sentence} \nPredicted emojis: {result}")


if __name__ == "__main__":
    main()
