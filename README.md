# Artificial Text Generation

The reposiory provides a basic fundamental scripts to train a LSTM based artificial text generator and also perform inference.
The neural network is trained using PyTorch framework. Hence PyTorch needs to be setup on Host using the requirements file provided.

The model checkpoint is saved every 30 epochs and can be adjusted. 
Also, two dummy text files are provided for training.

## Usage

### Training
cd scripts

python3 train.py --train_file ../data/train_news.txt

### Inference
cd scripts

python3 inference.py --start_sequence "You have" --num_words 10
