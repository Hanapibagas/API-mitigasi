import string
import pandas as pd
import numpy as np
import re
import nltk
import csv
import base64

#import keras
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from django.http import JsonResponse, HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view, schema, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from . models import Classification
from . serializers import getClassification
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, Flatten, MaxPooling1D    
from tensorflow.keras.models import Sequential
import tensorflow as tfr
from .viewtelegram import send_message_view

nltk.download('punkt')
# Create your views here.

def casefolding(input):
    input = input.lower()
    return input

def cleansing(input):
    input = input.strip(" ")
    input = re.sub(r'[?|$|.|!_:")(-+,]', '', input)
    input = re.sub(r'\d+', '', input)
    input = re.sub(r"\b[a-zA-Z]\b", "", input)
    input = re.sub('\s+',' ', input)
    input = input.replace("'","")
    return input
        
def word_tokenize_wrapper(text):
 return word_tokenize(text)


def preprocess(text, words):
    good = []
    text = text.lower()
    for a_word in text.split():
        if a_word not in words:
            good.append(a_word)

    s = " "
    return s.join(good)

@api_view(['GET'])
@permission_classes([AllowAny])
def classification(request) :
    msg = request.query_params['msg']
    msg = base64.b64decode(msg)
    
    #msg = "Banjir Bandang yang"
    #msg =  word_tokenize_wrapper([msg])
    #arr = pd.array([msg], dtype=str)
    arr = pd.DataFrame({"check": [msg]})
    
    #arr = np.append(arr, msg)
    training = pd.read_csv('D:/Project CODING/Project Skripsi/mitigasiAPI/mitigasiAPI/MyAPI/datapelaporan.csv',encoding='latin1',sep=',')
    datawords = []
    with open("D:/Project CODING/Project Skripsi/mitigasiAPI/mitigasiAPI/MyAPI/removewords.txt") as file:
        for item in file:
            datawords.append(item.replace('\n',''))
            #datawords = np.array(list(training.values()))
    
    data_clean = training.astype({'instansi' : 'category'})
    data_clean = training.astype({'input' : 'string'})
    data_clean.dtypes
    #print(datawords)
    data_clean['input'] = data_clean['input'].apply(casefolding)
    data_clean['input'] = data_clean['input'].apply(cleansing)
    data_clean['input'] = data_clean['input'].apply(word_tokenize_wrapper)
    data_clean['input'] = data_clean['input'].apply(lambda x: [item for item in x if item not in datawords])
    
    x = data_clean['input']
    y = data_clean['instansi']
    #print(x)
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    data_check = arr.astype({'check': 'string'})
    data_check.dtypes
    
    data_check['check'] = data_check['check'].apply(casefolding)
    data_check['check'] = data_check['check'].apply(cleansing)
    data_check['check'] = data_check['check'].apply(word_tokenize_wrapper)
    #data_check['check'] = data_check['check'].apply(lambda x: [item for item in x if item in data_clean['input']])
    gabung = []
    for xy in data_clean['input'] :
        #print(xy)
        for ab in xy :
            if ab not in gabung :
                gabung.append(ab)
                
        #gabung.append(xy)

    #print(gabung)
    data_check['check'] = data_check['check'].apply(lambda x: [item for item in x if item in gabung])
    xx = data_check['check']
    print(data_check)
    if len(xx[0]) == 0 :
        classnya = "kosong"
    else :
        vocab_size = 5000
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(x)
        #tokenizer.fit_on_texts(arr[0])

        X_train_sequences = tokenizer.texts_to_sequences(x)
        X_test_sequences = tokenizer.texts_to_sequences(xx)
        yytrain = to_categorical(y,3)

        #print("ini sequencenya training", X_train_sequences)
        #print("ini sequencenya", X_test_sequences)
        max_length = 10
        padding_type = "post"
        trunction_type="post"
        X_train_padded = pad_sequences(X_train_sequences,maxlen=max_length, padding=padding_type,truncating=trunction_type)
        X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length, padding=padding_type,truncating=trunction_type)
        
        random.seed(42)
        tfr.random.set_seed(42)

        model = Sequential([
            Embedding(vocab_size, 10, input_length=max_length),
            Conv1D(20, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(10, activation='relu'),
            Dense(3, activation='softmax')
        ])

        opt = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])
        model.fit(X_train_padded, yytrain, epochs=20 , batch_size = 5)

        #convert text
        #data_test = msg.astype('string')
        
        
        #arr =  word_tokenize_wrapper(arr)
        
        
        
        output = model.predict(X_test_padded)
        lengthoutput = len(output)
        #print(lengthoutput)
        for x in range(lengthoutput) :
            arr = np.array(output[x])
            index = np.where(arr == np.amax(arr))

        if index[0][0] == 0 :
            classnya = "Basarnas"
        elif index[0][0] == 1 :
            classnya = "BPBD"
        else :
            classnya = "Damkar"

        send_message_view(msg, classnya)
        # send_message("hallo")
    return JsonResponse({"instansi": classnya})


    #return HttpResponse(x)


def testing(request):
    
    return JsonResponse({"instansi":"Damkar"})
    #serializer = getClassification(getC, many=True)
    #return Response("response")
    

# from django.http import JsonResponse
# from telegram import Bot

# def start(update, context):
#     # Dapatkan informasi pengguna yang mengirim perintah
#     user = update.message.from_user

#     # Kirim pesan balasan
#     context.bot.send_message(chat_id=update.message.chat_id, text=f"Hello, {user.first_name}!")

#     return JsonResponse({"status": "Message sent!"})

# def send_message(msg):
#     # Anda dapat mengganti ini dengan token bot Anda
#     TOKEN = '5989229755:AAG1wMh1a-3vlWYRJkll-mq3JR3CM5RMfro'
#     bot = Bot(token=TOKEN)

#     # Misalnya, Anda ingin mengirim pesan ke chat_id tertentu
#     chat_id = '-917907649'
#     message = 'Hello from Django!'

#     bot.send_message(chat_id=chat_id, text=message)

#     return JsonResponse({"status": "Message sent!"})

