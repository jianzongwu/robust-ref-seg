import torch.nn as nn
import transformers
import torch.nn.functional as F
from utils.util import load_pretrained_swin

from swin_transformer_encoder_fusion import SwinTransformerEncoderFusion
from model.position_encoding import build_position_encoding
from model.segmentation_decoder_fusion import SegmentationDecoderFusion


class RefSegFormer(nn.Module):
    def __init__(self, config, args, logger):
        super().__init__()
        self.config = config
        self.args = args
        self.model_type = config.MODEL.TYPE
        self.model_name = config.MODEL.NAME

        self.position_encoding = build_position_encoding(args)

        self.text_encoder = transformers.BertModel.from_pretrained('./checkpoints/bert-base-uncased')
        self.text_encoder.pooler = None
        self.image_encoder, self.segmentation = build_model(self.config, logger, args, self.position_encoding)
    
    def forward(self, img, emb, att_mask):
        _, _, H, _ = img.size()

        text_feature = self.text_encoder(emb, attention_mask=att_mask)[0]
        fuse_features, memory_list, vision_lan_att_list,  mem_att_list, feature_att_list = self.image_encoder(img, text_feature, att_mask)
        
        output = self.segmentation(fuse_features, memory_list, att_mask)

        out_list = []
        for pred in output["mask_list"]:
            _, _, h, _ = pred.size()
            assert H % h == 0
            out_list.append(F.interpolate(pred, scale_factor=int(H//h), mode='bilinear', align_corners=True))

        return out_list, output["exist_pred"]

def build_model(config, logger, args, position_encoding):
    model = SwinTransformerEncoderFusion(img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        num_mem=args.num_mem,
        num_neg_mem=args.num_neg_mem,
        hidden_dim=args.hidden_dim,
        position_encoding=position_encoding,
        args=args
    )
    load_pretrained_swin(config, model, logger)

    segmentation = SegmentationDecoderFusion(config, args, position_encoding)

    return model, segmentation