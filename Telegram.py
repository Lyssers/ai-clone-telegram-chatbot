import logging
from telegram import Update
import requests
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import fire
import torch
from transformers import AutoTokenizer
import json
from openai import OpenAI
import random

from inference.model_utils import load_model, load_peft_model

secretChatID = -9999999999
secretPrivateChatID = 999999999

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
peft_model: str="checkpoints"
quantization: bool=True
max_new_tokens =4000 #The maximum numbers of tokens to generate
seed: int=42 #seed value for reproducibility
do_sample: bool=True #Whether or not to use sampling ; use greedy decoding otherwise.
min_length: int=5 #The minimum length of the sequence to be generated, input prompt + min_new_tokens
use_cache: bool=True  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
top_p: float=0.05 # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
temperature: float=0.3 # [optional] The value used to modulate the next token probabilities.
top_k: int=20, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
repetition_penalty: float=1.1 #The parameter for repetition penalty. 1.0 means no penalty.
length_penalty: int=1 #[optional] Exponential penalty to the length that is used with beam-based generation. 
max_padding_length: int=0 # the max padding length to be used with tokenizer padding the prompts.

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
model = load_model(model_name, quantization)

if peft_model:
    model = load_peft_model(model, peft_model)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

#this is the starting/system prompt for the chatbot
start_prompt_english = [
    {"role": "user", "content": "Your name is Bot McBot. You are a clone, trained on a user's  WhatsApp chats."},
    {"role": "assistant", "content": "I will follow these commands. I will now start the conversation."},
    ]

# default to german
prompt = start_prompt_english.copy()

def initVar():
    global OAI_key
    global OAI
    global client

    try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file.")
        exit()

    class OAI:
        key = data["keys"][0]["OAI_key"]
        model = data["OAI_data"][0]["model"]
        prompt = data["OAI_data"][0]["prompt"]
        temperature = data["OAI_data"][0]["temperature"]
        max_tokens = data["OAI_data"][0]["max_tokens"]
        top_p = data["OAI_data"][0]["top_p"]
        frequency_penalty = data["OAI_data"][0]["frequency_penalty"]
        presence_penalty = data["OAI_data"][0]["presence_penalty"]
    client = OpenAI(api_key = "openAI_API_Key")


#def llm(message):
#    message = message[15:]
#    response = client.chat.completions.create(model=OAI.model, messages=[{"role": "system", "content": (OAI.prompt + "\n\n#########\n" + message + "\n#########\n")}])
#    print(response)
#    return(response.choices[0].message.content)

def main(
    input,
    model_name,
    peft_model: str=None,
    quantization: bool=True,
    max_new_tokens =4000, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=5, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.05, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.3, # [optional] The value used to modulate the next token probabilities.
    top_k: int=20, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.1, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    max_padding_length: int=0, # the max padding length to be used with tokenizer padding the prompts.
    **kwargs):
    
    if input:
        prompt.append({
            "role": "user",
            "content": input
        })
        tokenizer_input = tokenizer.apply_chat_template(conversation=prompt, tokenize=False)
        batch = tokenizer(tokenizer_input, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = output_text.split('[/INST]')[-1]
        print("The bot said")
        print (reply)
        prompt.append({
            "role": "assistant",
            "content": reply
        })
        return reply


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# This is the main function for the chat features

async def bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
   print(update.message.chat_id)
   if update.message.chat_id == secretChatID or update.message.chat_id == secretPrivateChatID: 
        print(update.message.text.upper())
       if ("ASK CHATGPT" in update.message.text.upper()):
       #     response = "CHATGPT Said: " + (llm(update.message.text))   #ChatGPT integration
       #     await context.bot.send_message(chat_id=update.message.chat_id, text=response)  #ChatGPT integration
        else:
            response = fire.Fire(main(input = update.message.text, peft_model="checkpoints", model_name="mistralai/Mistral-7B-Instruct-v0.2"))
            await context.bot.send_message(chat_id=update.message.chat_id, text=response)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.chat_id == secretChatID or update.message.chat_id == secretPrivateChatID:  
        trimmed_echo = update.message.text[6:]
        await context.bot.send_message(chat_id = update.message.chat_id, text = "You said: " + trimmed_echo)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.chat_id == secretChatID or update.message.chat_id == secretPrivateChatID:  
        print(update.message.chat_id)
        await context.bot.send_message(
            chat_id= update.message.chat_id,
            text="Hello There!"
        )

async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.chat_id == secretChatID or update.message.chat_id == secretPrivateChatID:
        await context.bot.send_message(update.message.chat_id, "I'm thankful for the short time I was given... Goodbye") 
        import sys
        print("argv was",sys.argv)
        print("sys.executable was", sys.executable)
        print("restart now")

        import os
        os.execv(sys.executable, ['python'] + sys.argv)

# Uncomment the below lines if you want the bot to have the ability to send images and videos via /image or /video commands
# Put your videos in video_folder and images in image_folder respectively, with filenames as numbers
# The below code will send a random image or video depending on the command. 
# Make sure to change the bounds for the randint() function to match how many images and videos you have

#async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
#    if update.message.chat_id == secretChatID or update.message.chat_id == secretPrivateChatID:
#        random_number = str(random.randint(1,554))
#        await context.bot.send_photo(update.message.chat_id, photo=open("image_folder" + "/" + random_number + ".jpg", 'rb'))

#async def video(update: Update, context: ContextTypes.DEFAULT_TYPE):
#    if update.message.chat_id == secretChatID or update.message.chat_id == secretPrivateChatID:
#        await context.bot.send_message(update.message.chat_id, "lemme pick one")
#        random_number = str(random.randint(1,77))
#        await context.bot.send_video(update.message.chat_id, video=open("video_folder" + "/" + random_number + '.mp4', 'rb'), supports_streaming=True)

# Below code will will add a shutdown command to the bot. By default this will just print the message "Nice Try" but you can add your own logic for shutting down the bot

async def shutdown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.chat_id == secretChatID or update.message.chat_id == secretPrivateChatID:
        await context.bot.send_message(chat_id = update.message.chat_id, text="Nice Try")
        

if __name__ == '__main__':
    
    application = ApplicationBuilder().token('token').build()
    
    start_handler = CommandHandler('start', start)

    echo_handler = CommandHandler('echo', echo)

    restart_handler = CommandHandler('restart', restart)

    shutdown_handler = CommandHandler('shutdown', shutdown)

    #image_handler = CommandHandler('image', image)

    #video_handler = CommandHandler('video', video)

    bot_handler = MessageHandler(filters.TEXT & (~ filters.COMMAND) ,bot) # This makes sure the LLM doesn't respond to commands, only to text
    
    initVar()
    # print("\n\nChat GPT integration Running!\n\n")
    
    application.add_handler(start_handler)
    application.add_handler(bot_handler)
    application.add_handler(echo_handler)
    application.add_handler(restart_handler)
    application.add_handler(shutdown_handler)
    #application.add_handler(image_handler)
    #application.add_handler(video_handler)

    application.run_polling()





    
