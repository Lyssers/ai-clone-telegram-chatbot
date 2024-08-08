import json

#json_str = '''{"name":"June","type":"personal_chat","id":6248608289,"messages":[{"id":76030,"type":"message","date":"2023-06-27T03:11:29","date_unixtime":1687831889,"from":"Alyssa","from_id":"user1989752798","text":"I live inside your walls","text_entities":[{"type":"plain","text":"I live inside your walls"}]}]}'''

#data = json.loads(json_str)


with open('result.json', encoding="utf8") as f:
    data = json.load(f)
#    print(d)

for msg in data["messages"]:
    if msg["type"] == "message":
        for ent in msg["text_entities"]:
            if ent["type"] == "plain":
                #print(ent["text"])
                #print(msg["date"])
                #print(msg["from"])
                result = {
                    "date": (msg["date"][:10]),
                     "timestamp": (msg["date"][11:]),
                     "sender": (msg["from"]),
                    "text": '<sender>' + (msg["from"]) + '</sender>' + (ent["text"])
                 }
                print(result)