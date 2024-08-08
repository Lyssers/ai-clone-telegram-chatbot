# Import the modules
import csv
import glob
import os
import random
import re
import json

# Define a function to generate a random message id
def generate_id():
  # Use a combination of letters and digits
  chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  # Return a string of 8 random characters
  return "".join(random.choice(chars) for _ in range(8))


# Define a function to convert a txt file to a csv file
def json_to_csv(json_path, csv_writer, parent_id):
  # Open the json file for reading
  print("Parsing JSON")
  with open(json_path, encoding="utf8") as f:
    # Initialize the message to None
    message_id = None
    message = None
    #Load JSON
    data = json.load(f)
    # Initialize the flag to False
    is_message = False
    for i, msg in enumerate(data["messages"]):
      if msg["type"] == "message":
        for ent in msg["text_entities"]:
            if ent["type"] == "plain":
              parent_id = message_id
              message_id = generate_id()
              csv_writer.writerow([message_id, parent_id, '<sender>' + (msg["from"]) + '</sender>' + (ent["text"]), (msg["date"][:10]), (msg["date"][11:]), (msg["from"])])
              print ("Wrote: " + message_id, parent_id, '<sender>' + (msg["from"]) + '</sender>' + (ent["text"]), (msg["date"][:10]), (msg["date"][11:]), (msg["from"]))          
      # If the line does not start with a digit, it is a continuation of the previous message
      else:
        print ("No message here.")
  # Return the last message id
  return message_id

# Define a function to convert a folder of txt files to a csv file
def folder_to_csv(folder_path, csv_path):
  # Get the list of txt files in the folder
  json_files = glob.glob(os.path.join(folder_path, "*.json"))
  print("Json files sorted")
  # Sort the txt files by name
  json_files.sort()
  # Open the csv file for writing
  with open(csv_path, "w", encoding='utf-8', newline='') as csv_file:
    # Create a csv writer object
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(["message_id", "parent_id", "text", "date", "timestamp", "sender"])
    print("Wrote first row")
    # Initialize the parent id and the previous txt file to None
    parent_id = None
    prev_json_file = None
    # Loop through the txt files
    for json_file in json_files:
      # If the txt file name is different from the previous one, reset the parent id to None
      if json_file != prev_json_file:
        parent_id = None
      # Convert the txt file to a csv file
      parent_id = json_to_csv(json_file, csv_writer, parent_id)
      # Update the previous txt file to the current one
      prev_json_file = json_file


#if name main
if __name__ == "__main__":

  #get commandline argument data_folder
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_folder", type=str, default="data/preprocessing/raw_chats")
  parser.add_argument("--output_folder", type=str, default="data/preprocessing/processed_chats/")
  args = parser.parse_args()

  # Convert the txt files to a csv file
  folder_to_csv(f"{args.data_folder}/train/telegram", f"{args.output_folder}/train/telegram/train_chats.csv")
  folder_to_csv(f"{args.data_folder}/validation/telegram", f"{args.output_folder}/validation/telegram/validation_chats.csv")