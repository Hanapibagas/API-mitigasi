from django.core.management.base import BaseCommand
import telebot

bot = telebot.TeleBot("5989229755:AAG1wMh1a-3vlWYRJkll-mq3JR3CM5RMfro")

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")

class Command(BaseCommand):
    help = 'Run the Telegram bot'

    def handle(self, *args, **kwargs):
        bot.infinity_polling()
