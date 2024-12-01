import base64
import requests
import os
# from transformers import BertTokenizer, BertModel
# import torch
# import torch.nn.functional as F
import json
import re
import csv

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def get_base64_image_files(image_files):
    base64_image_files=[]
    for index in range(len(image_files)):
        base64_image_files.append(encode_image(image_files[index])) 
    return base64_image_files

def get_sub_dirs(path): 
    sub_files=[]
    for foldername, subfolders, filenames in os.walk(path):
        #print('Folder: ' + foldername)
        for subfolder in subfolders:
            #print('  Subfolder: ' + os.path.join(foldername, subfolder))
            sub_files.append(os.path.join(foldername,subfolder))
    #print(sub_files)
    return sub_files

def get_prompt_payload_json(base64_image_files,text_prompt,max_fig_num):

    image_url_content_load=[]

    json_prompt={ 
        "type":"text",
        "text":text_prompt
    }
    image_url_content_load.append(json_prompt)

    for index in range(max_fig_num):
        #print("index=",index);print()
        json_image_url = {
            "type": "image_url",
            "image_url": {
                "url":f"data:image/jpeg;base64,{base64_image_files[index]}",
                #"url":f"data:image/jpeg;base64,{index}",
            }
        }
        image_url_content_load.append(json_image_url)
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": image_url_content_load
            }
        ],
        "max_tokens": 200 
    }

    return payload


def get_image_files(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_files.append(os.path.join(root, file))
    return image_files

def get_sample_image_files(image_folder, sample_number,partition):
    image_files=[]
    for image_name in os.listdir(image_folder):
        if image_name.endswith((".jpg",".png",".jpeg",".bmp",".tif",".gif")):
            image_files.append(os.path.join(image_folder,image_name))
    images_number=len(image_files)
    image_sample_files=[]
    for index in range(len(image_files)):
        if index*(images_number/sample_number)<images_number:
            image_sample_files.append(image_files[index*int((images_number/sample_number))])
    if partition=="25-1":
        return image_files[:20]
    if partition=="25-2":
        return image_files[int(len(image_files)*0.25):(int(len(image_files)*0.25)+15)]
    if partition=="25-3":
        return image_files[int(len(image_files)*0.5):(int(len(image_files)*0.5)+15)]
    if partition=="25-4":
        return image_files[int(len(image_files)*0.75):(int(len(image_files)*0.75)+15)]
    if partition=="all":
        return image_sample_files

def get_image_description(text_prompt, base64_image_files):
    api_key="API Key"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload =get_prompt_payload_json(base64_image_files,text_prompt,15)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def save_describe_sentence(file_path, folder_name, file_content):
    with open(file_path, 'a+') as f:
        data={
            "video_folder_name":folder_name.split("\\")[-1],
            "describe_content":file_content
        }
        json.dump(data,f)
        f.write("\n")

def remove_incomplete_sentences(paragraphs):
    sentences = re.split(r'(?<=[.!?])\s+', paragraphs)
    if not re.search(r'[.!?]$', sentences[-1]):
        sentences.pop()  
    cleaned_paragraph = " ".join(sentences)
    return cleaned_paragraph

if __name__ == "__main__":
    text_prompt = "These pictures are frames extracted from a video. Please recognize the visual elements from the video and generate three sentences to describe the video, with each sentence as diverse as possible."
    
    csv_reader = csv.reader(open("D:\MSRVTT\MSRVTT-Frames\msrvtt_data\MSRVTT_JSFUSION_test.csv"))
    num = 0
    for row in csv_reader:
        print(row)
        if row[0] == "key":
            continue
        image_path = "D:\MSRVTT\MSRVTT-Frames\Frames_30fps" 
        folder_path = os.path.join(image_path, row[2])
        image_sample_files=get_sample_image_files(folder_path,20,"25-1")
        
        base64_image_files=get_base64_image_files(image_sample_files)
        try:
            describe_sentence_response = get_image_description(text_prompt, base64_image_files) 
            describe_sentence_response_refresh = remove_incomplete_sentences(describe_sentence_response)
            save_describe_sentence(r"D:\testSaveDescribe\saveGenMSRVTT.json",row[2],describe_sentence_response_refresh)

        except:
            print("Exception happend,dir name=",row[2])
            save_describe_sentence(r"D:\testSaveDescribe\saveException.json",row[2],"Exception happened.")

        if num% 10 == 0 and num>0 :
            print("Captions of "+str(num)+" videos has been generated.") 

        num = num + 1  

   
