from django.http import JsonResponse
import telebot
# from .views import classification

bot = telebot.TeleBot("5989229755:AAG1wMh1a-3vlWYRJkll-mq3JR3CM5RMfro")

def send_message_view(msg, classnya, lokasi):
    chat_id = ""
    full_message = f"Isi pesan pelapor: {msg.decode('utf-8')}\nTitik lokasi: {lokasi}"

    if classnya == "Damkar":
        chat_id = '-917907649'
        bot.send_message(chat_id, full_message)
    elif classnya == "BPBD":
        chat_id = '-940400678'
        bot.send_message(chat_id, full_message)
    elif classnya == "Basarnas":
        chat_id = '-949930165'
        bot.send_message(chat_id, full_message)
    else:
        return JsonResponse({"status": "success"})
    
    return JsonResponse({"status": "success"})




# def send_message_view(msg, classnya, lokasi):
#     chat_id = ""
#     if classnya == "Damkar":
#         chat_id = '-917907649'
#         bot.send_message(chat_id, msg)
#         bot.send_message(chat_id, lokasi)
#     elif classnya == "BPBD":
#         chat_id = '-940400678'
#         bot.send_message(chat_id, msg)
#         bot.send_message(chat_id, lokasi)
#     elif classnya == "Basarnas":
#         chat_id = '-949930165'
#         bot.send_message(chat_id, msg)
#         bot.send_message(chat_id, lokasi)
#     else :
#         return JsonResponse({"status": "success"})
#     return JsonResponse({"status": "success"})
        
