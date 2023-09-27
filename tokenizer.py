import numpy as np

class Tokenizer():
    def __init__(self, config):
        content_vocab = ["<sos>"] + open(config.c_vocab_path, "r").read().split("\n") + ["<eos>", "pad"]
        html_vocab = ["<sos>"] + open(config.html_vocab_path, "r").read().split("\n") + ["<eos>", "pad"]

        self.c_vocab = dict(zip(content_vocab, np.arange(len(content_vocab))))
        self.html_vocab = dict(zip(html_vocab, np.arange(len(html_vocab))))
        self.reverse_vocab = list(self.html_vocab.items())
        self.max_c_length = config.content_maxlen
        self.max_html_length = config.tags_maxlen
        self.new_cell = {"<td", "<td></td>"}

    def tokenize_content(self, content):
        res = []
        for c in content:
            try:
                res.append(self.c_vocab[c])
            except:
                print(c)
                res.append(0)
        if len(res) < self.max_c_length:
            res += [self.c_vocab["pad"]]*(self.max_c_length-len(res)+1)
        elif len(res) > self.max_c_length:
            res = res[:self.max_c_length-len(res)] + [self.c_vocab["<eos>"]]
        else:
            res += [self.c_vocab["pad"]]
        return res

    def tokenize_html(self, html_tags):
        res = []
        mark_cell = []
        for tag in html_tags:
            if tag in self.new_cell:
                mark_cell.append(1)
            else:
                mark_cell.append(0)
            try:
                res.append(self.html_vocab[tag])
            except:
                res.append(0)
                print(tag)
        if len(res) < self.max_html_length:
            res += [0]*(self.max_html_length-len(res)+1)
            mark_cell += [0]*(self.max_html_length-len(res)+1)
        elif len(res) > self.max_html_length:
            res = res[:self.max_html_length-len(res)] + self.html_vocab["<eos>"]
            mark_cell = mark_cell[:self.max_html_length-len(res)+1]
        return res, mark_cell

    def parsing(self, anot_path):
        lines = open(anot_path, "r").readlines()
        img_path = lines[0][:-1]
        html_tags = ["<sos>"] + lines[1][:-1].split(",") + ["<eos>"]

        bboxs, contents = [], []
        for line in lines[2:]:
            if line == "0,0,0,0<;><UKN>\n":
                continue
            bbox, content = line.split("<;>")
            bboxs.append(list(map(int, bbox.split(","))))
            content = ["<sos>"] + content[:-1].split("\t") + ["<eos>"]
            if len(self.tokenize_content(content)) != self.max_c_length + 1:
                print(anot_path)
                print(line)
            contents.append(self.tokenize_content(content))

        new_bboxs, new_contents = [], []
        tokenized_tags = []
        mark_cell = []
        for tag in html_tags:
            if tag in self.new_cell:
                mark_cell.append(1)
                new_bboxs.append(bboxs.pop(0))
                new_contents.append(contents.pop(0))
            else:
                mark_cell.append(0)
                new_bboxs.append([0,0,0,0])
                new_contents.append([self.c_vocab["pad"]]*(self.max_c_length + 1))
            try:
                tokenized_tags.append(self.html_vocab[tag])
            except:
                tokenized_tags.append(0)
                print(tag)
        if len(tokenized_tags) < self.max_html_length:
            padding = (self.max_html_length-len(tokenized_tags)+1)
            tokenized_tags += [self.html_vocab["pad"]]*padding
            mark_cell += [0]*padding
            new_bboxs += [[0,0,0,0]]*padding
            new_contents += [[self.c_vocab["pad"]]*(self.max_c_length + 1)]*padding
        elif len(tokenized_tags) > self.max_html_length:
            padding = self.max_html_length-len(tokenized_tags)
            tokenized_tags = tokenized_tags[:padding] + [self.html_vocab["<eos>"]]
            mark_cell = mark_cell[:padding+1]
            new_bboxs = new_bboxs[:padding+1]
            new_contents = new_contents[:padding+1]
        else:
            tokenized_tags += [self.c_vocab["pad"]]
            mark_cell += [0]
            new_bboxs += [0,0,0,0]
            new_contents += [self.c_vocab["pad"]]*(self.max_c_length + 1)

        return  np.array(tokenized_tags).astype(np.int64),\
                np.array(mark_cell).astype(np.float64),\
                np.array(new_bboxs).astype(np.float64),\
                np.array(new_contents).astype(np.int64)

    def reverse_html_tags(self, tokens):
        res = []
        for token in tokens:
            res.append(self.reverse_vocab[token][0])
            if token == len(self.html_vocab) - 2:
                break
        return res