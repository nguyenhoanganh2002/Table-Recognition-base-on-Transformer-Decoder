import cv2
import torch

class CustomDataset(Dataset):
    def __init__(self, anot, config, transforms=None, training=True):
        super().__init__()
        self.img_source = "/content/examples/examples/"
        self.anot_source = "/content/examples/groundtruth/StructureLabelAddEmptyBbox_train/"
        self.tokenizer = Tokenizer(config)
        self.anot = anot
        self.transforms = transforms
        self.training = training

    def __len__(self):
        return len(self.anot)

    def __getitem__(self, idx):
        if self.training:
            img_path = self.img_source + self.anot[idx][:-3] + "png"
            anot_path = self.anot_source + self.anot[idx]
        else:
            img_path = "/content/drive/MyDrive/multilabels_data/testset/" + self.anot['name'][idx]
            anot_path = self.anot_source + self.anot[idx]

        image = self.resize_n_pad(img_path)
        if self.transforms:
            image = self.transforms(image).to(torch.float64)

        tokenized_tags, mark_cell, new_bboxs, new_contents = self.tokenizer.parsing(anot_path)
        # print(tokenized_tags)
        return {
            "image":    image.to(torch.float64),
            "tags":     torch.tensor(tokenized_tags).to(torch.int64),
            "mask":     torch.tensor(mark_cell).to(torch.float64),
            "bboxs":    torch.tensor(new_bboxs).to(torch.float64),
            "contents": torch.tensor(new_contents).to(torch.int64)
        }
    def resize_n_pad(self, img_path):
        desired_size = 480

        im = cv2.imread(img_path)
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [255,255,255]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        return new_im