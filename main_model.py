class MultitaskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderBlock(config)
        self.sharedDecoder = SharedDecoder(config)
        self.structDecoder = StructureDecoder(config)
        self.bboxDecoder = BBoxDecoder(config)
        self.contDecoder = ContentDecoder(config)
        self.softmax = nn.Softmax(dim=-1)

        # for cal loss
        self.weights = torch.full((config.tags_vocab_size,), 3)
        self.weights[3] = 1
        self.CE_loss = nn.CrossEntropyLoss(weight=self.weights.double(), reduction="none")
        self.L1_loss = nn.L1Loss(reduction="none")

        self.tag_pad_value = config.tags_vocab_size-1
        self.cont_pad_value = config.content_vocab_size-1

        # for inference
        self.tokenizer = Tokenizer(config)
        self.s_tag_vector = torch.zeros((1,1)).to(torch.int64)
        self.s_cont_vector = torch.zeros((1,1,1)).to(torch.int64)
        self.e_tag_value = config.tags_vocab_size-2
        self.e_cont_value = config.content_vocab_size-2
        self.tag_maxlen = config.tags_maxlen
        self.c_maxlen = config.content_maxlen

    def forward(self, img, tags=None, letters=None, train=True):
        # train phase
        if train:
            encoder_output = self.encoder(img)                          # B, H*H, n_embd
            x = self.sharedDecoder(tags, encoder_output)                # B, tag_maxlen, n_embed
            struct_outs = self.structDecoder(x, encoder_output)         # B, tag_maxlen, tag_vocab_size
            bbox_outs = self.bboxDecoder(x, encoder_output)             # B, tag_maxlen, 4
            #letter_outs = self.contDecoder(letters, encoder_output, x)  # B, tag_maxlen, content_maxlen, content_vocab_size
            return struct_outs, bbox_outs#, letter_outs

        # inference phase
        else:
            assert img.shape[0] == 1, "inference batch size must be 1"
            encoder_output = self.encoder(img)                          # B, H*H, n_embd

            out_tags = self.s_tag_vector                                # B, 1
            out_bboxs = []
            out_contents = []


            newCell_token = [self.tokenizer.html_vocab["<td></td>"], self.tokenizer.html_vocab["<td"]]

            while out_tags.shape[1] <= self.tag_maxlen:
                decoder_output = self.sharedDecoder(out_tags, encoder_output)
                tag_outs = self.structDecoder(decoder_output, encoder_output)   # B, _, tag_vocab_size

                # new tag
                tag_prob = tag_outs[:,-1,:].unsqueeze(1)                        # B, 1, tag_vocab_size
                _, tag = torch.max(self.softmax(tag_prob), dim=-1)

                # update tag
                out_tags = torch.cat((out_tags, tag), dim=-1)

                # end tag
                if  tag.item() == self.e_tag_value:
                    break

                # new cell
                if tag.item() in newCell_token:
                    decoder_output = decoder_output[:,-1,:].unsqueeze(1)

                    # bboxs
                    bbox = self.bboxDecoder(decoder_output, encoder_output)
                    out_bboxs.append(np.array(bbox[0][0].cpu()))

                    # content
                    # s_cont = self.s_cont_vector
                    # while s_cont.shape[2] < self.c_maxlen:
                    #     out_letters = self.contDecoder(s_cont, encoder_output, decoder_output)

                    #     # new letter
                    #     letter_prob = out_letters[:,-1,-1,:].unsqueeze(1)                        # B, 1, tag_vocab_size
                    #     letter = torch.argmax(self.softmax(letter_prob), dim=-1).unsqueeze(1)

                    #     # update content
                    #     s_cont = torch.cat((s_cont, letter), dim=-1)

                    #     # end content
                    #     if letter.item() == self.e_cont_value:
                    #         break
                    # # add contents
                    # out_contents.append(np.array(s_cont[0][0].cpu()))

            return out_tags[0:1, 1:], out_bboxs#, out_contents


    def struct_loss(self, preds, out_tags):
        # preds:            B, tags_maxlen, tags_vocab_size
        # out_tags:         B, tags_maxlen
        # t_mask (padding):   B, tags_maxlen
        t_mask = ~torch.eq(out_tags, torch.full(out_tags.shape, self.tag_pad_value))

        preds = preds.permute(0, 2, 1).contiguous()
        losses = self.CE_loss(preds, out_tags)
        losses = (losses*t_mask.float()).sum(dim=-1).div(t_mask.sum(dim=-1)+1)

        return losses.mean()

    def bboxs_loss(self, preds, bboxs, c_mask):
        # preds, bboxs:     B, tags_maxlen, 4
        # c_mask:           B, tags_maxlen
        losses = self.L1_loss(preds, bboxs).sum(dim=-1)
        losses = (losses*c_mask).sum(dim=-1).div(c_mask.sum(dim=-1)+1)

        return losses.mean()

    def cont_loss(self, preds, out_conts, c_mask):
        #preds:             B, tag_maxlen, c_max_len, c_vocab_size
        #targets:           B, tag_maxlen, c_max_len
        #c_mask(cell mask): B, tag_maxlen
        #l_mask(letter):    B, tag_maxlen, c_max_len
        l_mask = ~torch.eq(out_conts, torch.full(out_conts.shape, self.cont_pad_value))

        losses = self.CE_loss(preds.transpose(1,-1).transpose(-1,-2), out_conts)
        losses = (losses*l_mask).sum(dim=-1).div(l_mask.sum(dim=-1)+1)
        losses = (losses*c_mask).sum(dim=-1).div(c_mask.sum(dim=-1)+1)

        return losses.mean()
