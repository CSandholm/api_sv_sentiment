import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

with open("model_config.json", "r") as f:
    config = json.load(f)

model = AutoModelForSequenceClassification.from_pretrained(config.get("model_path"))
tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_path"))


async def get_sentiment(sentence):
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    result = model(tokens)
    output_np = result.logits[0].detach().cpu().numpy()
    output = softmax(output_np)
    result = calc_sentiment(output)

    return result


def calc_sentiment(float_array):
    highest_index = float_array.max()
    highest = float_array[highest_index]
    lowest_index = float_array.min()
    lowest = float_array[lowest_index]

    #Index 0: Negative
    #Index 1: Neutral
    #Index 2: Positive
    if highest_index != 0:
        if highest_index != 1:
            sentiment = highest
        else:
            sentiment = 0.5 - float_array[0] + float_array[2]
    else:
        sentiment = 1 - float_array[highest_index]

    return sentiment

