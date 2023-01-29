import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate
from .model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detect_lines (opt) :
    
    log = open(f'transcriber/transcription.txt', 'w')
    
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    opt.input_channel = 1
    
    # Load the model into the CUDA Memory.
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    
    # Loading the Weights.
    model.load_state_dict(torch.load("transcriber/best_norm_ED.pth", map_location=device))
    
    # torch.save(model.state_dict(), "/home/saksham/Desktop/Final BTP Work/Website/pytorch-django/transcriber/best_norm_ED.pth", _use_new_zipfile_serialization=False)
    
    #print ("Weights Loaded")
    
    #print (opt.image_folder)

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    count = 0
    final_text = ""

    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            count += 1
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            
            preds = model(image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
            
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            
            final_text = final_text + pred + "\n"

            #print (pred)  
            # log.write(f'{pred}\t\n')
    
    log.write(final_text)
    return final_text
    # print ("COUNT   ", count)
