from django.http import JsonResponse
import telebot
# from .views import classification

bot = telebot.TeleBot("5989229755:AAG1wMh1a-3vlWYRJkll-mq3JR3CM5RMfro")

def send_message_view(msg, classnya):
    chat_id = ""
    if classnya == "Damkar":
        chat_id = '-917907649'  # Gantikan dengan chat_id Anda
        bot.send_message(chat_id, msg)
    elif classnya == "BPBD":
        chat_id = '-940400678'  # Gantikan dengan chat_id Anda
        bot.send_message(chat_id, msg)
    elif classnya == "Basarnas":
        chat_id = '-949930165'  # Gantikan dengan chat_id Anda
        bot.send_message(chat_id, msg)
    else :
        return JsonResponse({"status": "success"})
    return JsonResponse({"status": "success"})
        
