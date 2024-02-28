import numpy as np
import torch 
import torch.nn.functional as F
from .util import AverageMeter, reduce_tensor
import time

def validate(args, logger, data_loader, model, local_rank=0, eval_mode=True):
    '''
    evaluate R-RIS with new metrics.
    '''
    seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(seg_iou_list), dtype=np.int32)
    seg_total = 0
    with torch.no_grad():
        model.eval()

        batch_time=AverageMeter()
        I_meter=AverageMeter()
        U_meter=AverageMeter()
        mIOU_meter=AverageMeter()
        r_meter=AverageMeter()
        rIoU_meter=AverageMeter()

        end=time.time()

        for idx, (img, target, emb, att_mask, exist) in enumerate(data_loader):
            batch_size = img.size(0)
            emb = emb.squeeze(1)
            att_mask = att_mask.squeeze(1)

            img = img.cuda(local_rank, non_blocking=True) # [B,3,H,W]
            target = target.cuda(local_rank, non_blocking=True) #[B, ori_H, ori_W]
            emb = emb.cuda(local_rank, non_blocking=True) # [B,len] or [B, len, n_sent]
            att_mask = att_mask.cuda(local_rank, non_blocking=True) # [B,len] or [B, len, n_sent]
            exist = exist.cuda(local_rank, non_blocking=True) # [B, 1, n_sent]
            _, o_H, o_W=target.size()

            # compute positive IOU
            positive_index = exist == 1
            positive_index = positive_index.squeeze()
            positive_emb = emb[:, :, positive_index]
            positive_att_mask = att_mask[:, :, positive_index]
            if positive_index.numel() == 1:
                positive_emb = positive_emb.squeeze(-1)
                positive_att_mask = positive_att_mask.squeeze(-1)
            # concatenation prompt, concatenate several sentences to one sentence, only used in positive sentences.
            if args.eval_mode == "cat":
                positive_emb = positive_emb.view(batch_size, -1)
                positive_att_mask = positive_att_mask.view(batch_size, -1)
                output, exist_pred = model(img, positive_emb, positive_att_mask)
                outputs = [output]
            else:
                _, _, num_of_sent = positive_emb.size()
                outputs = []
                for s in range(num_of_sent):
                    emb_s, att_mask_s = positive_emb[:, :, s], positive_att_mask[:, :, s]
                    output, exist_pred = model(img, emb_s, att_mask_s)
                    if args.use_exist:
                        if exist_pred < 0.5:
                            output = [torch.zeros_like(output[0], device=output[0].device)]
                    outputs.append(output)
            I = torch.Tensor([0]).cuda()
            U = torch.Tensor([0]).cuda()
            IoU = torch.Tensor([0]).cuda()
            for output in outputs:
                output = output[0]  # finest mask
                output = F.interpolate(output, (o_H, o_W), align_corners=True, mode='bilinear')
                pred = output.argmax(1)

                temp_I = torch.sum(torch.mul(pred,target))*1.0
                temp_U = torch.sum(torch.add(pred,target))*1.0 - temp_I
                I = I + temp_I
                U = U + temp_U
                IoU = IoU + temp_I*1.0/temp_U
                for n_eval_iou in range(len(seg_iou_list)):
                    eval_seg_iou = seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (IoU >= eval_seg_iou)
                seg_total += 1
            IoU = IoU / len(outputs)

            if eval_mode == False:
                torch.cuda.synchronize()
                I = reduce_tensor(I)
                U = reduce_tensor(U)
                IoU = reduce_tensor(IoU)

            I_meter.update(I.item())
            U_meter.update(U.item())
            mIOU_meter.update(IoU.item())

            # compute negateve robustness r
            negative_index = exist == 0
            negative_index = negative_index.squeeze()
            negative_emb = emb[:, :, negative_index]
            negative_att_mask = att_mask[:, :, negative_index]
            if negative_index.numel() == 1:
                negative_emb = negative_emb.squeeze(-1)
                negative_att_mask = negative_att_mask.squeeze(-1)
            _, _, num_of_sent = negative_emb.size()
            outputs = []
            for s in range(num_of_sent):
                emb_s, att_mask_s = negative_emb[:, :, s], negative_att_mask[:, :, s]
                output, exist_pred = model(img, emb_s, att_mask_s)
                if args.use_exist:
                    if exist_pred < 0.5:
                        output = [torch.zeros_like(output[0], device=output[0].device)]
                outputs.append(output)
            TN = torch.Tensor([0]).cuda()
            FP = torch.Tensor([0]).cuda()
            for output in outputs:
                output = output[0]  # finest mask
                pred = output.argmax(1)
                if torch.sum(pred) > 0:
                    FP += 1
                    U = U + torch.sum(pred)
                else:
                    TN += 1
            r = TN / (TN + FP)

            rIoU = I / U

            if eval_mode == False:
                torch.cuda.synchronize()
                r = reduce_tensor(r)
                rIoU = reduce_tensor(rIoU)
            
            r_meter.update(r.item())
            rIoU_meter.update(rIoU.item())

            batch_time.update(time.time()-end)
            end=time.time()

            if (idx + 1) % args.print_freq==0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}] '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    f'mIoU {100*mIOU_meter.avg:.3f} '
                    f'oIoU {100*float(I_meter.sum)/float(U_meter.sum):.3f} '
                    f'r {100*r_meter.avg:.3f} '
                    f'rIoU {100*rIoU_meter.avg:.3f} '
                    f'Mem {memory_used:.0f}MB')
        mIoU = 100*mIOU_meter.avg
        oIoU = 100*float(I_meter.sum)/float(U_meter.sum)
        r = 100*r_meter.avg
        rIoU = 100*rIoU_meter.avg
        results_str = ''
        for n_eval_iou in range(len(seg_iou_list)):
            results_str += f'P@{seg_iou_list[n_eval_iou]} {seg_correct[n_eval_iou] * 100. / seg_total:.2f} '
        logger.info(f'mIoU {mIoU:.3f} oIoU {oIoU:.3f} r {r:.3f} rIoU {rIoU:.3f} {results_str}')
        return {
            "mIoU": mIoU, 
            "oIoU": oIoU, 
            "r": r, 
            "rIoU": rIoU
        }