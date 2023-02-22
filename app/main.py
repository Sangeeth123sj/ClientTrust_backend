import os
import openai
import cohere
from cohere.classify import Example
from dotenv import load_dotenv
from fastapi import FastAPI,Form
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# loading env variables
load_dotenv() 

# initializing Cohere API
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# initializing OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


# hello world of ClientSense
@app.get("/")
async def root():
    return {"message": "Hello ClientSense World"}

# endpoint to autocorrect the text obtained from converation 
# using transcription in vuejs on the frontend
@app.post("/autocorrect/")
async def autocorrect(text: str= Form(...)):
    # initial try with Open AI api since Cohere makes some mistakes sometimes by adding extra text
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Correct this to standard English:\n\n.{text}",
        temperature=0.0,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # using cohere generate to autocorrect transcribed text
    # co_response = co.generate(
    # model='command-xlarge-nightly',
    # prompt=f'Make this grammaticaly correct and add punctuations:\n {text}',
    # max_tokens=900,
    # temperature=0.0,
    # k=0,
    # p=0.75,
    # frequency_penalty=0,
    # presence_penalty=0,
    # stop_sequences=[],
    # return_likelihoods='NONE')
    # print('Prediction: {}'.format(co_response.generations[0].text))
    print("response:",response)
    return {"response": response}


# endpoint to summarize the sales conversation for further follow up
@app.post("/summarize/")
async def summarize(text: str= Form(...)):
    response = co.generate(
    model='xlarge',
    prompt='''Conversation: Hello, can I help you with anything today?
            Yes, I'm looking for a new phone. Do you have the latest model in stock?
            Yes, we have it available. Can I see it? Of course, here it is. What do you think of it?
            It's perfect, I'll take it.How much does it cost? It's priced at $800.
            Alright, I'll purchase it. Thank you for your assistance. You're welcome, have a great day.
            \n
            Summary: A customer is looking for a new phone and asks if the latest model is in stock. 
            The salesman provides the phone for the customer to see and the customer decides to purchase it for $800. 
            The salesman thanks the customer for the purchase.
            \n--\n
            Conversation: Hello, how can I assist you today?
            I'm in need of a new laptop. Do you have any recommendations?
            Yes, we have several options that might interest you.
            Can you show me some of them? Of course, here are a few.
            Which one do you like the best? I think this one would be great.
            How much is it? It's priced at $1,200. Okay, I'll take it.
            Thank you for your help. You're welcome, have a great day.
            \n
            Summary: A customer is in need of a new laptop and asks for recommendations.
            The salesman provides several options for the customer to choose from.
            The customer decides on one laptop and purchases it for $1,200.
            The salesman thanks the customer for the purchase.
            \n--\n
            Conversation:{text}
            \n
            Summary:
            ''',
    max_tokens=100,
    temperature=0.8,
    k=0,
    p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=["--"],
    return_likelihoods='NONE')

    summary = response.generations[0].text
    print("summary:",summary)
    return {"summary":summary}




# examples dataset for training Cohere to give sales conversation suggessions

suggession_examples = [
Example("What is the cost of this product?", "Provide pricing information and offer financing options"),
Example("How much does the product cost?", "Provide pricing information and offer financing options"),
Example("Can you show me a demo of the product?", "Provide product information and offer a demo,Emphasize unique selling points and offer a demo"),
Example("What is the demo of this product like?", "Provide product information and offer a demo,Emphasize unique selling points and offer a demo"),
Example("Is it possible to purchase this product online?", "Provide purchasing options and highlight convenience"),
Example("Can I buy the product online?", "Provide purchasing options and highlight convenience"),
Example("Do you provide financing options?", "Provide financing options and emphasize the affordability"),
Example("What financing options are available?", "Provide financing options and emphasize the affordability"),
Example("Can you explain the financing process?", "Explain the financing process and provide examples"),
Example("How does financing work?", "Explain the financing process and provide examples"),
Example("Is it possible to pay in installments?", "Explain the installment options and highlight the convenience"),
Example("What are the installment options?", "Explain the installment options and highlight the convenience"),
Example("What payment methods do you accept?", "Provide payment options and emphasize the security"),
Example("What are the payment options?", "Provide payment options and emphasize the security"),
Example("Are there discounts for bulk purchases?", "Explain bulk purchasing discounts and highlight the savings"),
Example("What discounts are available for bulk purchases?", "Explain bulk purchasing discounts and highlight the savings"),
Example("How does bulk purchasing work?", "Explain bulk purchasing process and highlight the benefits"),
Example("What is the process of bulk purchasing?", "Explain bulk purchasing process and highlight the benefits"),
Example("Can I get a discount for buying in bulk?", "Emphasize bulk purchasing discounts and provide pricing information"),
Example("Is there a bulk purchase discount?", "Emphasize bulk purchasing discounts and provide pricing information"),
Example("Do you offer any promotions?", "Explain current promotions and emphasize the savings"),
Example("What promotions are available?", "Explain current promotions and emphasize the savings"),
Example("What are the current promotions?", "Provide current promotion information and highlight the benefits"),
Example("Are there any current promotions?", "Provide current promotion information and highlight the benefits"),
Example("Is there a better price if I buy now?", "Explain promotional deals and emphasize the urgency"),
Example("Can I get a better price if I buy now?", "Explain promotional deals and emphasize the urgency"),
Example("Do you offer a warranty for the product?", "Explain warranty options and emphasize the peace of mind"),
Example("What kind of warranty is available for the product?", "Explain warranty options and emphasize the peace of mind"),
Example("What kind of after-sales support do you provide?", "Explain after-sales support options and emphasize the reliability"),
Example("What support is available after I buy the product?", "Explain after-sales support options and emphasize the reliability"),
Example("What happens if the product breaks down?", "Explain warranty options and emphasize the peace of mind"),
Example("Is there support if the product breaks down?", "Explain warranty options and emphasize the peace of mind"),
]

# endpoint to provide suggessions for the next steps in the sales conversation
@app.post("/suggessions/")
async def suggessions(text: str= Form(...)):
    print("suggesion input:", text)
    suggession_list = []
    response = co.classify(  
    model='large',  
    inputs=[text],  
    examples=suggession_examples)

    print("testing data-----------",response.classifications[0].labels.keys())
    print("labels-------------------",response.classifications)
    # obtaining suggession keys for forming a new list of dict 
    # consisting of suggessions and their corresponding confidences
    suggession_keys = list(response.classifications[0].labels.keys())
    for suggession in suggession_keys:
        # steps to convert the confidence in the response to a percentage form
        confidence = str(response.classifications[0].labels[suggession])
        confidence = float(confidence.split('=')[1].strip().rstrip(')'))
        confidence = round(confidence* 100,2)
        print("suggession:", suggession, "confidence:", confidence)
        suggession_list.append({"suggession:": suggession, "confidence:": confidence})
    # sorts the list to show higher confidence suggessions first
    suggession_list.sort(key=lambda x: x['confidence:'], reverse=True)
    # limiting the no. of suggessions to the best 6
    suggession_list = suggession_list[:5]
    # the best suggession 
    main_suggession = response.classifications[0].prediction
    print("main_suggession:",main_suggession)
    print("suggesion_list", suggession_list)
    return {"main_suggession":main_suggession,"suggession_list":suggession_list}



# examples to train Cohere model for customer sentiment analysis
sales_pitch_conversation_examples = [
Example("I'm interested in purchasing this product", "positive"),
Example("Can you tell me more about the product's features?", "neutral"),
Example("How does this product compare to similar products?", "neutral"),
Example("What discounts do you offer for bulk purchasing?", "positive"),
Example("Can you provide pricing information?", "neutral"),
Example("What financing options do you offer?", "positive"),
Example("How reliable is your after-sales support?", "positive"),
Example("Can you explain the bulk purchasing process?", "positive"),
Example("What promotions are currently available?", "positive"),
Example("Can you explain the warranty options?", "positive"),
Example("Can I return the product if it's not what I expect?", "negative"),
Example("What payment options do you offer?", "neutral"),
Example("Do you offer a demo of the product?", "positive"),
Example("What makes your product unique?", "positive"),
Example("Can you provide product information?", "neutral"),
Example("How convenient is the purchasing process?", "positive"),
Example("The order came 5 days early", "positive"),
Example("The item exceeded my expectations", "positive"),
Example("I ordered more for my friends", "positive"),
Example("I would buy this again", "positive"),
Example("I would recommend this to others", "positive"),
Example("The package was damaged", "negative"),
Example("The order is 5 days late", "negative"),
Example("The order was incorrect", "negative"),
Example("I want to return my item", "negative"),
Example("The item's material feels low quality", "negative"),
Example("I had a terrible experience", "negative"),
Example("The customer service was unhelpful", "negative"),
Example("The product was not as described", "negative"),
Example("I was dissatisfied with my purchase", "negative"),
Example("The product was okay", "neutral"),
Example("I received five items in total", "neutral"),
Example("I bought it from the website", "neutral"),
Example("I used the product this morning", "neutral"),
Example("The product arrived yesterday", "neutral"),
Example("The product didn't live up to my expectations", "negative"),
Example("I was disappointed with the product's quality", "negative"),
Example("The product didn't work as advertised", "negative"),
Example("I wouldn't recommend this product", "negative"),
Example("I had trouble setting up the product", "negative"),
Example("The instructions were unclear", "negative"),
Example("I had to contact customer support multiple times", "negative"),
Example("The customer service was slow and unresponsive", "negative"),
Example("The product was faulty and had to be returned", "negative"),
Example("I wouldn't purchase this product again", "negative"),
Example("The product was overpriced", "negative"),
Example("The product was underwhelming", "negative"),
Example("I was dissatisfied with the product's features", "negative"),
Example("I received a different product than what I ordered", "negative"),
Example("The product arrived damaged", "negative"),
Example("The product was too complex to use", "negative"),
Example("The product was too bulky", "negative"),
Example("The product was too small", "negative"),
Example("The product was too heavy", "negative"),
Example("I was unimpressed with the product's design", "negative"),
Example("I wasn't satisfied with the product's performance", "negative"),
Example("The product was difficult to assemble", "negative"),
Example("The product was difficult to install", "negative"),
Example("The product was difficult to operate", "negative"),
Example("The product was difficult to use", "negative"),
Example("The product was difficult to clean", "negative"),
Example("The product was difficult to store", "negative"),
Example("The product was not worth the price", "negative"),
Example("The product was not what I expected", "negative"),
Example("The product was not as described in the advertisement", "negative"),
Example("The product was not as durable as I expected", "negative"),
Example("The product was not as high quality as I expected", "negative"),
Example("I wasn't impressed with the product's color", "neutral"),
Example("I wasn't impressed with the product's size", "neutral"),
Example("I wasn't impressed with the product's shape", "neutral"),
Example("I wasn't impressed with the product's weight", "neutral"),
Example("I wasn't impressed with the product's texture", "neutral"),
Example("I wasn't impressed with the product's style", "neutral"),
Example("The product was just okay", "neutral"),
Example("I wasn't impressed with the product's performance", "neutral"),
Example("I didn't see any significant improvements with the product", "neutral"),
Example("I wasn't impressed with the product's results", "neutral"),
Example("The product was average", "neutral"),
Example("I had mixed feelings about the product", "neutral"),
Example("I didn't feel like the product was necessary", "neutral"),
Example("I didn't feel like the product added much value", "neutral"),
Example("I wasn't sure about the product's purpose", "neutral"),
Example("I wasn't sure about the product's use", "neutral"),
Example("I wasn't sure about the product's effectiveness", "neutral"),
Example("I wasn't sure about the product's quality", "neutral"),
]



# endpoint to provide sentiment analysis of customer in the sales conversation
@app.post("/sentiment/")
async def sentiment(text: str= Form(...)):
    print("sentiment input:", text)
    sentiment_list = []
    response = co.classify(  
        model='large',  
        inputs=[text],  
        examples=sales_pitch_conversation_examples
    )

    print("testing data-----------",response.classifications[0].labels.keys())
    print("labels-------------------",response.classifications)
    # obtaining suggession keys for forming a new list of dict 
    # consisting of suggessions and their corresponding confidences
    sentiment_keys = list(response.classifications[0].labels.keys())
    for sentiment in sentiment_keys:
        # steps to convert the confidence in the response to a percentage form
        sentiment_level = str(response.classifications[0].labels[sentiment])
        sentiment_level = float(sentiment_level.split('=')[1].strip().rstrip(')'))
        sentiment_level = round(sentiment_level* 100,2)
        print("===============================")
        print("sentiment:", sentiment, "sentiment_level:", sentiment_level)
        sentiment_list.append({"sentiment:": sentiment, "sentiment_level:": sentiment_level})
    



    # sorts the list to show higher confidence suggessions first
    sentiment_list.sort(key=lambda x: x['sentiment_level:'], reverse=True)
    # Retrieve the first item (the one with the highest "sentiment_level")
    highest_sentiment = sentiment_list[0]
    print("sentiment_list", sentiment_list, "highest_sentiment",highest_sentiment)
    return {"sentiment_list":sentiment_list, "highest_sentiment":highest_sentiment}




# toxicity examples to train Cohere classify model
toxicity_examples = [
  Example("you are hot trash", "Toxic"),  
  Example("go to hell", "Toxic"),
  Example("get rekt moron", "Toxic"),  
  Example("get a brain and use it", "Toxic"), 
  Example("say what you mean, you jerk.", "Toxic"), 
  Example("Are you really this stupid", "Toxic"), 
  Example("I will honestly kill you", "Toxic"),  
  Example("yo how are you", "Benign"),  
  Example("I'm curious, how did that happen", "Benign"),  
  Example("Try that again", "Benign"),  
  Example("Hello everyone, excited to be here", "Benign"), 
  Example("I think I saw it first", "Benign"),  
  Example("That is an interesting point", "Benign"), 
  Example("I love this", "Benign"), 
  Example("We should try that sometime", "Benign"), 
  Example("You should go for it", "Benign")
]



# endpoint to provide sentiment analysis of customer in the sales conversation
@app.post("/toxicity/")
async def toxicity(text: str= Form(...)):
    print("sentiment input:", text)
    toxicity_list = []
    response = co.classify(  
        model='large',  
        inputs=[text],  
        examples=toxicity_examples
    )

    print("testing data-----------",response.classifications[0].labels.keys())
    print("labels-------------------",response.classifications)
    # obtaining suggession keys for forming a new list of dict 
    # consisting of suggessions and their corresponding confidences
    toxicity_keys = list(response.classifications[0].labels.keys())
    for toxicity in toxicity_keys:
        # steps to convert the confidence in the response to a percentage form
        toxicity_level = str(response.classifications[0].labels[toxicity])
        toxicity_level = float(toxicity_level.split('=')[1].strip().rstrip(')'))
        toxicity_level = round(toxicity_level* 100,2)
        print("===============================")
        print("toxicity:", toxicity, "toxicity_level:", toxicity_level)
        toxicity_list.append({"toxicity": toxicity, "toxicity_level": toxicity_level})
    # sorts the list to show higher confidence suggessions first
    # sentiment_list.sort(key=lambda x: x['sentiment_level:'], reverse=True)
    print("toxicity_list", toxicity_list)
    return {"toxicity_list":toxicity_list}