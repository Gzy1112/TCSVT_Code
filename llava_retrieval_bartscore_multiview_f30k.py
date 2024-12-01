import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from dataset_zoo import COCO_Order, Flickr30k_Order, VG_Relation, VG_Attribution
from torch.utils.data import DataLoader
from bart_score import BARTScorer
from tqdm import tqdm
from model_zoo import get_model
import numpy as np
import clip
import pandas as pd

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_llava_texts(tokenizer, model, image_tensor,conv_mode,inp=""):
        
    #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    if not inp:
        print("exit...")
        
    conv = conv_templates[conv_mode].copy()
    #print(f"{roles[1]}: ", end="")

    if True:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    #print(prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=60,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    #conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return outputs

@torch.no_grad()
def main(args):
    
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    print(image_processor)
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()

    #image = load_image(args.image_file)
    #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # vgr_dataset = VG_Relation(image_preprocess=image_processor, download=True, root_dir="/data/gongziyu/LLaVA-main/llava/serve/notebook/cache")
    vgr_dataset = Flickr30k_Order(image_preprocess=image_processor, split="test", root_dir="/data/gongziyu/LLaVA-main/llava/serve/notebook/cache")
    vgr_loader = DataLoader(vgr_dataset, batch_size=1, shuffle=False)
    
    
    bart_scorer = BARTScorer(device='cuda:0', checkpoint="/data/gongziyu/BARTScore-main/bart_large_cnn/")#checkpoint='facebook/bart-large-cnn') #:0'
    bart_scorer.load(path='/data/gongziyu/BARTScore-main/bart_score.pth')

    scores = []
    tqdm_loader = tqdm(vgr_loader)
    tqdm_loader.set_description("Computing retrieval scores")
    last_image_id = ""
    correct = 0
    NUM = 0
    for batch in tqdm_loader:
        NUM += 1
        image_id_now = batch["image_id"][0]
        image_options = []
        i_option = batch["image_options"][0] #1 picture
        ##image_embeddings = clip_model.encode_image(i_option.to("cuda:0")).cpu().numpy() # B x D
        input = "Please describe the above image in different sentences, including as many objects as possible, with each sentence as diverse as possible. Try to demonstrate relationships between objects and their properties in each sentence. Each sentence should not exceed 20 words." 
        #input = ""
        if last_image_id != image_id_now:
           last_image_id = image_id_now
           llava_txt = get_llava_texts(tokenizer, model,i_option,args.conv_mode,inp=input)
           result_list = re.split(r'[.]', llava_txt) 

        if '</s>' in result_list[-1]:
           result_list.pop()
        llava_txt_num = len(result_list)
        
        bart_score = []
        for c_option in batch["caption_options"]: #5 sentence
            for llava_result in result_list:
                bart_score_tmp = bart_scorer.score((llava_result,),c_option,batch_size=1)
                #bart_score_tmp = bart_scorer.score(c_option,c_option,batch_size=len(c_option))
                bart_score.extend(bart_score_tmp)
        
        first_max = sum(bart_score[:llava_txt_num]) / llava_txt_num
        # second_max = sum(bart_score[llava_txt_num :]) / llava_txt_num
        # bart_score_final = [first_max, second_max] 
        second_max = sum(bart_score[llava_txt_num : 2*llava_txt_num]) / llava_txt_num
        third_max = sum(bart_score[2*llava_txt_num : 3*llava_txt_num]) / llava_txt_num
        fourth_max = sum(bart_score[3*llava_txt_num : 4*llava_txt_num]) / llava_txt_num
        fifth_max = sum(bart_score[4*llava_txt_num : 5*llava_txt_num]) / llava_txt_num
        bart_score_final = [first_max, second_max, third_max, fourth_max, fifth_max]  
        
        if first_max == max(first_max, second_max, third_max, fourth_max, fifth_max):
           correct += 1
            
        batch_scores = np.expand_dims(np.expand_dims(np.array(bart_score_final),axis=0),axis=0) # B x K x L #1,1,5
        scores.append(batch_scores)
        #break
        print(correct/NUM)
    # all_scores = np.concatenate(scores, axis=0) # N x K x L
    # vgr_records = vgr_dataset.evaluate_scores(all_scores)
    # # symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
    # df = pd.DataFrame(vgr_records)
    # # df = df[~df.Relation.isin(symmetric)]
    # print(f"VG-Relation Macro Accuracy: {df.Accuracy.mean()}")
    all_scores = np.concatenate(scores, axis=0) # N x K x L
    records = vgr_dataset.evaluate_scores(all_scores)
    print(f"F30K Order Accuracy: {records}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
