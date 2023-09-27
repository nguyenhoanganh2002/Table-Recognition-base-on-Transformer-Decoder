# Table-Recognition-base-on-Transformer-Decoder
An end to end model for three sub-tasks of Table Recognition:  table structure recognition, cell detection, and cell-content recognition

## Dataset: [Pubtabnet](https://github.com/ibm-aur-nlp/PubTabNet.git)
## Architecture: base on [this paper](https://paperswithcode.com/paper/an-end-to-end-multi-task-learning-model-for-1)
* Consists one of Shared Encoder, one Shared Decoder and three separate Decoder for three sub-tasks
  * Shared Encoder using a CNN backbone network as the feature extractor
  * Four Decoders are inspired by original Transformer decoder 

![image](https://github.com/nguyenhoanganh2002/Table-Recognition-base-on-Transformer-Decoder/assets/79850337/d8bbf87f-5df9-4fa9-971d-b5dd68ff37c2)

![image](https://github.com/nguyenhoanganh2002/Table-Recognition-base-on-Transformer-Decoder/assets/79850337/4eda19bf-345f-4cbd-beac-12433c8ca922)

## Description:
* `config.py` contains hyperparameters
* `parsing_data.py` match raw data from Pubtabnet to anotation
* `tokenizer.py` encode characters, html tags
* `sub_module.py` build necessary sub-modules like Cross Attention, Self Attention, Positional Encoding, ...
* `main_model` build last model from sub-modules
* `train_infer.py` train loop
