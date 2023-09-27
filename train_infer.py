from main_model import MultitaskModel
from config import Config
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from tqdm import tqdm
import os
import torch

config = Config()

def infer(model, im_path, tokenizer=None):
    with torch.no_grad():
        model.eval()
        im = resize_n_pad(cv2.imread(im_path))
        tags, bboxs = model(im, train=False)
        tags = tokenizer.reverse_html_tags(tags)
        return tags, bboxs*480

if __name__ == "__main__":
    device="cuda"

    # define transforms and generate dataloader
    trans = {"train": transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
            "val": transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])}

    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    torch.autograd.set_detect_anomaly(True)
    model = MultitaskModel(config)
    def param_to_update(net):
        encoder_param = []
        decoder_param = []
        for name, param in net.named_parameters():
            param.requires_grad = True
            if "encoder" in name:
                encoder_param.append(param)
            else:
                decoder_param.append(param)
        return encoder_param, decoder_param

    param1, param2 = param_to_update(model)

    optimizer = torch.optim.Adam([{'params': param1, 'lr': 0.0005},
                            {'params': param2, 'lr': 0.001}])

    trainanot = os.listdir("/kaggle/input/pubtabnet-anot/train/StructureLabelAddEmptyBbox_train")[:1000]
    traindataset = CustomDataset(trainanot, config, transforms=trans["train"])
    trainloader = DataLoader(dataset=traindataset, batch_size=8, shuffle=True, generator=torch.Generator(device=device))

    valanot = os.listdir("/kaggle/input/pubtabnet-anot/val/StructureLabelAddEmptyBbox_val")[:200]
    valdataset = CustomDataset(valanot, config, transforms=trans["train"], training=False)
    valloader = DataLoader(dataset=valdataset, batch_size=8, shuffle=True, generator=torch.Generator(device=device))
    
    n_epoch = 2
    trainloss_his = []
    valloss_his = []
    for epoch in range(n_epoch):
        model.train()
        # model.float()
        train_loss = 0

        for batch in tqdm(trainloader):
            img = batch["image"].to(device)
            in_tags = batch["tags"][:,:-1].to(device)
            in_conts = batch["contents"][:,:-1,:-1].to(device)
            
            optimizer.zero_grad()
            
            # forward
            s_preds, b_preds = model(img, in_tags, in_conts)
    #         s_preds = model(img, in_tags, in_conts)

            # ground truth
            mask = batch["mask"][:,1:].to(device)
            bboxs = batch["bboxs"][:,1:].to(device)
            s_targets = batch["tags"][:,1:].to(device)
            #c_targets = batch["contents"][:,1:,1:].to(device)
            
    #         print(mask)
            
            s_loss = model.struct_loss(s_preds, s_targets)
            b_loss = model.bboxs_loss(b_preds, bboxs/480, mask)
            #c_loss = model.cont_loss(c_preds, c_targets, mask)

            loss = s_loss + b_loss# + c_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            torch.save(model, "/kaggle/working/multitask_model.pth")
            print(f"train loss: {train_loss/len(trainloader)}")
            trainloss_his.append(train_loss/len(trainloader))
            valloss=0
            if True:    
                with torch.no_grad():
                    for batch in tqdm(valloader):
                        img = batch["image"].to(device)
                        in_tags = batch["tags"][:,:-1].to(device)
                        in_conts = batch["contents"][:,:-1,:-1].to(device)
                        mask = batch["mask"][:,1:].to(device)
                        bboxs = batch["bboxs"][:,1:].to(device)
                        s_targets = batch["tags"][:,1:].to(device)
                        s_preds, b_preds = model(img, in_tags, in_conts)
                        s_loss = model.struct_loss(s_preds, s_targets)
                        b_loss = model.bboxs_loss(b_preds, bboxs/480, mask)
                        valloss += (s_loss+b_loss).item()
                    
                    print(f"val loss: {valloss/len(valloader)}")
                    valloss_his.append(valloss/len(valloader))
            if (epoch+1)%20 == 0:
                with torch.no_grad():
                    model.eval()
                    tag_acc = 0
                    bbox_loss = 0
                    c_acc = 0
                    for batch in tqdm(valloader):
                        img = batch["image"].to(device)
                        o_tags = batch["tags"][0,1:].to(device)
                        o_conts = np.array(batch["contents"][0,1:,1:].cpu())
                        bboxs = np.array(batch["bboxs"][0,1:].cpu())
                        mask = batch["mask"][0,1:]
                        non_empty_cell = mask.sum().item()

                        # print(type(o_conts))
                    
                        t_pred, b_pred = model(img, train=False) # tensor, list, list
    #                     t_pred = model(img, train=False)
        
                        tag_acc += ((t_pred==o_tags[:len(t_pred)]).float().sum()/non_empty_cell).item()*20

                        c_acc_i = 0
                        bbox_i = 0
                        j = 0
                        for i in range(len(mask)):
                            if mask[i] == 1:
                                if j < len(b_pred):
                                    # c_acc_i += (c_pred[j] == o_conts[i,:len(c_pred[j])]).mean()
                                    bbox_i += np.abs(bboxs[i] - b_pred[j]*480).mean()/4
                                    j += 1

                        try:
                            bbox_loss += bbox_i/non_empty_cell
                            # c_acc += c_acc_i/non_empty_cell
                        except:
                            print("no non-empty cell in input image")
                    print(f"html tags accuracy: {tag_acc/len(valloader)}")
                    # print(f"contents accuracy: {c_acc/len(valloader)}")
                    print(f"bboxs loss: {bbox_loss/len(valloader)}")