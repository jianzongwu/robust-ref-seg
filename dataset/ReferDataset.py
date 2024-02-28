import os 
import sys 
import torch.utils.data as data
import torch 
import numpy as np 
from PIL import Image 

import transformers

from dataset.refer import REFER

class ReferDataset(data.Dataset):
    def __init__(self,
                 args,
                 image_transforms=None,
                 max_tokens=20, 
                 split='train',
                 eval_mode=True,
                 logger=None) -> None:
        """
        parameters:
            args: argparse obj
            image_transforms: transforms apply to image and mask
            max_tokens: determined the max length of token 
            split: ['train','val','testA','testB']
            eval_mode: whether in training or evaluating 
        """

        self.classes = []
        self.image_transforms = image_transforms
        self.split = split
        self.args = args
        self.eval_mode=eval_mode
        self.refer=REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens=max_tokens

        self.get_ids()
        
        if logger:
            logger.info(f"=> loaded successfully '{args.dataset}', split by {args.splitBy}, split {split}")

    def get_ids(self):
        ref_ids=self.refer.getRefIds(split=self.split)
        img_ids=self.refer.getImgIds(ref_ids)
        # change dict to list
        all_imgs=self.refer.Imgs
        self.imgs=list(all_imgs[i] for i in img_ids)
        
        self.ref_ids=ref_ids
        # input_ids -> input sentence 对应的id
        # attention_masks -> mask掉pad的部分
        self.input_ids=[]
        self.attention_masks=[]
        self.tokenizer = transformers.BertTokenizer.from_pretrained('./checkpoints/bert-base-uncased')
        
        # pad_id=[0]
        # pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[PAD]'))
        for r in self.ref_ids:
            # for each image
            ref=self.refer.Refs[r]
            # List[Tensor] Tensor shape [1,len]
            sentences_for_ref=[]
            attentions_for_ref=[]
            # for each sentence
            for el in ref['sentences']:
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                # `add_special_tokens=True`加入<SOS>和<EOS>
                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))
            # List[List[Tensor]]
            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def get_exist(self, ref, sent_index):
        if "exist" in ref["sentences"][sent_index].keys():
            exist = torch.Tensor([ref["sentences"][sent_index]["exist"]])
        else:
            exist = torch.Tensor([True])
        return exist
        
    def __len__(self):
        return len(self.ref_ids)
        
    def __getitem__(self, index):
        this_ref_id=self.ref_ids[index]
        this_img_id=self.refer.getImgIds(this_ref_id)
        this_img=self.refer.Imgs[this_img_id[0]]
        
        img = Image.open(os.path.join(self.refer.IMAGE_DIR,this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)[0]

        ref_mask = np.array(self.refer.getMask(ref)['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask==1] = 1
        # convert it to a Pillow image
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.eval_mode:
            """
            torch.Size([3, 384, 384]) torch.Size([384, 384]) torch.Size([1, 20, 3]) torch.Size([1, 20, 3])
            """
            embedding = []
            att = []
            exist_list = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                ext = self.get_exist(ref, s)

                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
                exist_list.append(ext.unsqueeze(-1))
            # all sentence
            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
            exist = torch.cat(exist_list, dim=-1)
            # default the last annot as our
        else: # for training random select one sentence
            """
            torch.Size([3, 384, 384]) torch.Size([384, 384]) torch.Size([1, 20]) torch.Size([1, 20])
            """
            choice_sent=np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]
            exist = self.get_exist(ref, choice_sent)
            if exist == 0:
                annot = np.zeros(ref_mask.shape)
                annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)
        else:
            target = annot

        return img, target, tensor_embeddings, attention_mask, exist

    def getNewRef(self, sentence):
        """
        get the tensor_embeddings and attention_mask (padding mask) of self-defined sentences
        """
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        input_ids = self.tokenizer.encode(text=sentence, add_special_tokens=True)
        # truncation of tokens
        input_ids = input_ids[:self.max_tokens]

        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1]*len(input_ids)

        return torch.tensor(padded_input_ids).unsqueeze(-1), torch.tensor(attention_mask).unsqueeze(-1)