import whisper
from transformers import pipeline

## Sentiment analysis
sentiment_analysis = pipeline(
    "sentiment-analysis",
    framework="pt",
    model="SamLowe/roberta-base-go_emotions"
)


## Analyze text
def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {
        result["label"]:result['score'] for result in results
    }
    return sentiment_results

## Get sentiment emoji
def get_sentiment_emoji(sentiment):
    # Define the mapping of sentiments to emojis
    emoji_mapping = {
    "disgust": "😞",
    "sadness": "😢",
    "neutral": "😐",
    "anger": "😡",
    "happy": "😊",
    "fear": "😨",
    "surprise": "😲",
   }
    return emoji_mapping.get(sentiment, "")


## Display sentiment results
def display_sentiment_results(sentiment_results,option):
    sentiment_text = ""
    for sentiment,score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text+=f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score}\n"
            
    return sentiment_text


## Take audio as input and generate language , transcription, sentiment analysis
def inference(audio,sentiment_option):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    _,probs = model.detect_language(mel)
    lang = max(probs,key=probs.get)
    
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model,mel,options)
    
    sentiment_results = analyze_sentiment(result.text)
    sentiment_output = display_sentiment_results(sentiment_results,sentiment_option)
    sentiment_percent = sentiment_results[sentiment_output.strip(' \n')]

    
    return lang.upper(),result.text,sentiment_output.strip(' \n'),sentiment_percent