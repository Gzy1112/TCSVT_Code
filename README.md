# ML<sup>2</sup>MG-VLCR++: A Multimodal LLM Guided Zero-shot Method for Visio-Linguistic Compositional Reasoning with Autoregressive Generative Language Model



Experiments and data for the paper "ML<sup>2</sup>MG-VLCR++: A Multimodal LLM Guided Zero-shot Method for Visio-Linguistic Compositional Reasoning with Autoregressive Generative Language Model". 



#  Installation

Our model is based on [LLaVA]([GitHub - haotian-liu/LLaVA: [NeurIPS'23 Oral\] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond.](https://github.com/haotian-liu/LLaVA)) and [NegCLIP]([GitHub - mertyg/vision-language-models-are-bows: Experiments and data for the paper "When and why vision-language models behave like bags-of-words, and what to do about it?" Oral @ ICLR 2023](https://github.com/mertyg/vision-language-models-are-bows)) , please prepare environment for LLaVA and NegCLIP.



#  Datasets

## Visual Genome Relation & Attribution Datasets
The VG-Relation and VG-Attribution datasets are simple to use. For instance:
```python
import clip
from dataset_zoo import VG_Relation, VG_Attribution

model, image_preprocess = clip.load("ViT-B/32", device="cuda")

root_dir="/path/to/aro/datasets"
# Setting download=True will download the dataset to `root_dir` if it's not already there. 
# For VG-R and VG-A, this is a 1GB zip file that is a subset of GQA.

vgr_dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=root_dir)
vga_dataset = VG_Attribution(image_preprocess=preprocess, download=True, root_dir=root_dir)

# Do anything with the dataset. Each item will look like this : 
# item = {"image_options": [image], "caption_options": [false_caption, true_caption]}
```

## COCO-Order and Flickr30k-Order Datasets
These datasets require the Flickr30k retrieval datasets.  You can find the Flickr30k retrieval dataset [here](https://forms.illinois.edu/sec/229675).

```python
from dataset_zoo import Flickr30k_Order

flickr_order_dataset = Flickr30k_Order(image_preprocess=preprocess, root_dir=root_dir)
```

## MSRVTT-Order and MSVD-Order Datasets

These datasets require videos(frames) from MSRVTT and MSVD datasets.  You can find two datasets [here]([GitHub - whwu95/Cap4Video: 【CVPR'2023 Highlight & TPAMI】Cap4Video: What Can Auxiliary Captions Do for Text-Video Retrieval?](https://github.com/whwu95/Cap4Video)).

These two new datasets have been uploaded to `./datasets/MSRVTT-Order.json` and `./datasets/MSVD-Order.json`

```python
import json
fcc_file = "MSRVTT-Order.json" #MSRVTT-Order原文本
with open(fcc_file, 'r') as fcc_file:
    fcc_data = json.load(fcc_file)
    print(fcc_data)

# Do anything with the dataset. Each item will look like this : 
# item = {"video_id": [video_id], "caption_options": [true_caption, false_caption 1, false_caption 2, false_caption 3, false_caption 4]}
```



# How to use

## Image-Text Compositional Reasoning 

For VG-Relation

```python
python ./llava_retrieval_bartscore_multiview_relation.py
```

For VG-Attribution

```python
python ./llava_retrieval_bartscore_multiview_attrib.py
```

For Flickr30K-Order

```python
python ./llava_retrieval_bartscore_multiview_f30k.py
```



## Video-Text Compositional Reasoning 

Use ChatGPT-4o to transform video information into text information. 

```python
python ./ChatGPT_Description.py
```

For MSRVTT-Order

```python
run msrvtt_order_bartscore_multiview.ipynb
```

For MSVD-Order

```python
run msvd_order_bartscore_multiview.ipynb
```



# Contact 

Please let us know if you have further questions or comments. You can reach out to me at `ziyugong@smail.nju.edu.cn`. 
