import os
import torch
from model import *
import sentencepiece as spm
from tqdm import tqdm

class Translator():
    def __init__ (self, transformer, source_tokenizer, target_tokenizer, device="cpu"):
        self.transformer = transformer
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.device = device

        # Make sure we are on the correct device
        self.transformer.to(device)

    def to(self, device):
        self.device = device
        self.transformer.to(device)

    def from_checkpoint(checkpoint, device="cpu"):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"The checkpoint file {checkpoint} does not exist.")
        
        checkpoint = torch.load(checkpoint, map_location=device)
        config = checkpoint["config"]

        source_tokenizer = spm.SentencePieceProcessor(model_file=f"{config["data_root"]}/tokenizers/source.model")

        target_tokenizer = spm.SentencePieceProcessor(model_file=f"{config["data_root"]}/tokenizers/target.model")

        transformer = Transformer.from_dict(config["model"])

        translator = Translator(transformer, source_tokenizer, target_tokenizer, device)

    def inference(self, input, max_tokens=32):

        # Tokenize the input
        input_ids = torch.tensor(self.source_tokenizer.encode(input, add_bos=True, add_eos=True)).unsqueeze(0).to(self.device)

        # Perform inference
        predicted_tokens = torch.LongTensor([self.target_tokenizer.bos_id()]).unsqueeze(0).to(self.device)

        self.transformer.eval()

        for i in range(max_tokens):
            with torch.no_grad():
                output = self.transformer(predicted_tokens, input_ids)

                # Greedy decoding 
                output = torch.argmax(output, dim=-1)

                # Add the last token to our predicted tokens
                predicted_tokens = torch.cat([predicted_tokens, output[:, -1].unsqueeze(0)], dim=-1)

                # If we predict the end of the sentence, stop
                if output[:, -1] == self.target_tokenizer.eos_id():
                    break

        output_text = self.target_tokenizer.decode(predicted_tokens.squeeze(0).tolist())

        return output_text