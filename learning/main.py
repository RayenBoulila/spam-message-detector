message = input("donner un message : ")
text = message.lower()

if "win" in text or "prize" in text  or "gift" in text:
    print("Spam probable")
else:
    print("Message normal")