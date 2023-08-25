from django.http import JsonResponse
import telebot
# from .views import classification

bot = telebot.TeleBot("5989229755:AAG1wMh1a-3vlWYRJkll-mq3JR3CM5RMfro")

def send_message_view(msg):
    chat_id = '-917907649'  # Gantikan dengan chat_id Anda
    bot.send_message(chat_id, msg)
    return JsonResponse({"status": "success"})



# def get_anallisis(request):
#    # kelola file
#    text = get_voice_to_text()
#    # kita sudah dapat teks
#    # ketika teks sudah di dapat olah teksnya di mitigasi
#    # untuk mendapatkan algoritmanya
#    hasil = classification(text)
#    # kirim bot ke telegram
#    send_message_view(text)

#    return hasil