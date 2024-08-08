# Create an AI clone of yourself from your WhatsApp or Telegram chats (using Mistral 7B) and chat with it over Telegram!


## About
This is a fork of the (ai-clone-whatsapp repository)[ https://github.com/kinggongzilla/ai-clone-whatsapp/tree/6677acb8c087cd3a6da91a2dbc10bbbfdda92b49] that lets you create an AI chatbot clone of yourself, using your WhatsApp or Telegram chats as training data. 

The default model used is **Mistral-7B-Instruct-v0.2**. The code in this repository heavily builds upon llama-recipes (https://github.com/facebookresearch/llama-recipes), where you can find more examples on different things to do with llama models.

This repository includes code to:
* Preprocess exported WhatsApp or Telegram chats into a suitable format for finetuning
* Finetune a model on your WhatsApp or Telegram chats, using 4-bit quantized LoRa
* Chat with your finetuned AI clone, via a commandline interface or in Telegram.

## Hardware requirements
At least 22GB VRAM required for Mistral 7B finetune. I ran the finetune on a RTX 3090.
When experimenting with other models, VRAM requirement might vary.
VRAM requirements could probably be significantly reduced with some optimizations (maybe unsloth.ai).

## Not maintained

This code is not maintained. The original is much much more up to date and probably better. But this version has the added feature of letting you chat with the model over Telegram, and to use UK-formatted whatsapp chat exports and Telegram chat exports. If someone wants to add that to the newer version then I'm happy to help!

## What's new:

This version adds two files:

- Telegram.py

This lets you chat with the model in Telegram as a private bot. 
Includes ChatGPT integration so you can talk to both the local Mistral-7B and ChatGPT at the same time.
See guide below on how to get it working.

- preprocess_Telegram.py

This will let you use Telegram chat exports to train the model. Simply run the file exactly how you would with the whatsapp version. The output will be in the `/data/preprocessing/processed_chats/validation/telegram` directory, this won't be picked up by the training code by default - this is so you can manually (or write code to) merge the standard whatsapp outputs and the telegram outputs in case you use both (my use case).

This version also makes major changes to

- preprocess.py

To use the .txt WhatsApp file exports that are standard in the UK.


## Setup

1. Clone this repository
2. To install the required dependencies, run the following commands from inside the cloned repository:

```
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

Note: Python 3.12.0 or higher is recommended. Lower versions might run into issues during finetuning.

## Obtaining and preprocessing your WhatsApp chats

To prepare your WhatsApp chats for training, follow these steps:

1. Export your WhatsApp chats as .txt files. This can be done directly in the WhatsApp app on your phone, for each chat individually. You can export just one .txt from a single chat or many .txt files from all your chats.
Unfortunately, formatting seems to vary between regions. I am based in the UK, so the regex in the preprocessing.py might have to be adjusted if you are based in a different region.
2. Copy the .txt files you exported into ```data/preprocessing/raw_chats/train```. If you want to have a validation set, copy the validation chats into ```data/preprocessing/raw_chats/validation```.
3. Run ```python data/preprocessing/preprocess.py```. This will convert your raw chats into a format suitable for training and save CSV files to ```data/preprocessing/processed_chats```

## Obtaining and preprocessing your Telegram chats

Same as above, however format exports for Telegram chats are standard globally (afaik) so you do not need to worry.

2. Copy the result.json obtained from Telegram chat export (individual chat) into `/data/preprocessing/raw_chats/train/telegram/` 
Optionally: copy the validation if you hvae one into `/data/preprocessing/raw_chats/validation/telegram/`. 
3. Run `python data/preprocessing/prerocess_Telegram.py`. This will convert your raw chats into a format suitable for training and save CSV files to `data/preprocessing/processed_chats/train/telegram` for training and `data/preprocessing/processed_chats/validation/telegram` for validation set respectively (if you had one - validation set is completely optional).


## Start finetune/training
Run ```python -m finetuning --dataset "custom_dataset" --custom_dataset.file "scripts/custom_dataset.py" --whatsapp_username "[insert whatsapp_username]"```. Where you replace```[insert whatsapp_username]``` with your name, as it appears in the exported .txt files from WhatsApp. This is necessary to assign the correct role of "assistant" and "user" for training.

Note: Use the --whatsapp_username parameter even if you exported telegram chats. It doesn't matter as the format should be the same.

This will first download the base model from Huggingface, if necessary, and then start a LoRa finetune with 4-bit quantization.

The config for the training can be set in ```configs/training.py```. In particular, you can enable evaluation on the validation set after each epoch by setting ```run_validation: bool=True``` 

## Chatting with your AI clone on the command line

After successful finetuning, run ```python3 commandline_chatbot.py --peft_model [insert checkpoint folder] --model_name mistralai/Mistral-7B-Instruct-v0.2```, where ```[insert checkpoint folder]``` should be replaced with the output directory you specified in ```configs/training.py```. Default is ```checkpoints```folder.

Running this command loads the finetuned model and let's you have a conversation with it in the commandline.

You can define your own system prompt by changing the ```start_prompt_english``` prompt text in the ```commandline_chatbot.py``` file.

## Chatting with your AI clone in Telegram

Do not run the Telegram.py file as is, it will not work. Follow guide below on how to get that working:

1. Follow this guide to create a bot user on Telegram: https://core.telegram.org/bots/features#creating-a-new-bot
2. Open Telegram.py in any text editor, and find `application = ApplicationBuilder().token('token').build()` on line 204 and replace `'token'` with your token you got from step 1
3. At this stage the bot will be public. This is highly undesirable with a model that could hypothetically spill your private information from the chats you fed it. There is no built-in functionality to make a Telegram bot private, but lucky for you this code already handles this. You just need to manually add a chat ID to one or both variables on lines 14 and 15 (the variables named `secretChatID` and `secretPrivateChatID`. 
4. To acquire these, add your bot to the desired chats (by default it should already be in private DMs with you), but you can also add it to a group chat for example via the drop-down menu on the bot's profile and "add to group", then go to https://api.telegram.org/bot<YourBOTToken>/getUpdates in browser, replacing the <YourBotToken> with your bot token, and look for 
```
"chat": {
            "id": <group_ID>,
            "title": "<Group name>"
```
Where the id is your group_ID or chat ID. If this does not work, and just shows "ok true result" or some such, remove bot from a group and re-add. Group ID will always have a minus sign, while a DM ID will be a normal number. 
5. Add this into either the `secretPrivateChatID` or `secretChatID` (or both if you want the bot to respond to both DMs and group chats) as described in Step 3. Your bot will be searchable and available to everyone, but will only respond to commands and messages from the specified chat IDs.
6. Run Telegram.py and start chatting!

Simply DM the bot to start chatting!

## (Optional): If Mistral-7B isn't quite qutting it, you can add the ability to forward your chats to ChatGPT. The Mistral-7B bot will have no awareness of them.

1. Add an OpenAI API key if you have one at `openAI_API_Key` on line 74 at `client = OpenAI(api_key = "openAI_API_Key")` Then you can prepend your message with "Ask ChatGPT" (capitalization does not matter) to forward the query to ChatGPT.
2. Uncomment the llm(message) function on lines 77 - 81
3. Uncomment the functionality in lines 146 - 147 that checks for the presence of "ASK CHATGPT" in a message.


## Troubleshooting The Telegram features:

* If there are errors about missing libraries just install them with pip as normal
* The logging is quite verbose. Look at the code's logging and print statements and see how far it's getting (aka debugging)
* Use `/echo` in Telegram to have the bot echo back what you wrote. Useful for troubleshooting to distinguish between LLM errors and Telegram API handling errors. 
* Use `/restart` to have the Telegram bot restart. This will unload and reload the LLM completely. Useful for when the model starts repeating itself or doing other weird undesirable stuff.

## Other common Issues / Notes
* If your custom_dataset.py script is not finishing or throws a recursion error, you may need to increase the maximum number of recursions in ```train.py```. Please edit ```sys.setrecursionlimit(50000)```.
* The system language when exporting chats should probably be English. Some data cleaning steps depend on this.
* Finetuning works best with English chats. If your chats are in another language, you may need to adjust the preprocessing and training parameters accordingly.
* This code should also work with other models than Mistral 7B, but I haven’t tried it myself. If you want to experiment with different models, you can find them here.
* In my finetunes, I have not exported group chats from WhatsApp. I am unsure if this would also work or cause any errors.


