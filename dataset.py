import os
import torch
import linecache
import timeit
import numpy as np
import sentencepiece as spm
from queue import Queue
import threading

class PairedTextDataset():

    def __init__(self, source_file, target_file, source_tokenizer: spm.SentencePieceProcessor, target_tokenizer: spm.SentencePieceProcessor):
        self.source_file = source_file
        self.target_file = target_file
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        if not os.path.exists(source_file) or not os.path.exists(target_file):
            raise FileNotFoundError("Source or target file not found")
        
        self.source_lines = 0
        with open(self.source_file, 'r', encoding='utf-8') as src_file:
            for line in src_file:
                self.source_lines += 1
        
        self.target_lines = 0
        with open(self.target_file, 'r', encoding='utf-8') as tgt_file:
            for line in tgt_file:
                self.target_lines += 1
        
        if self.source_lines != self.target_lines:
            raise ValueError("Source and target files must have the same number of lines")

    def batch_length(self, batch_size=32):
        return np.ceil(self.source_lines / batch_size)
    
    def _get_batch(self, indicies):
            
            source_batch = []
            target_input_batch = []
            target_output_batch = []

            max_source_length = 0
            max_target_length = 0

            for idx in indicies:
                source_line = linecache.getline(self.source_file, idx+1).strip()
                target_line = linecache.getline(self.target_file, idx+1).strip()
                source_tokens = self.source_tokenizer.Encode(source_line, add_bos=True, add_eos=True)
                target_tokens = self.target_tokenizer.Encode(target_line, add_bos=True, add_eos=True)

                source_batch.append(source_tokens)
                target_input_batch.append(target_tokens[:-1])
                target_output_batch.append(target_tokens[1:])

                max_source_length = max(max_source_length, len(source_tokens))
                max_target_length = max(max_target_length, len(target_tokens) - 1)

            for idx in range(len(source_batch)):
                source_batch[idx] = torch.nn.functional.pad(torch.tensor(source_batch[idx]), (0, max_source_length - len(source_batch[idx])))
                target_input_batch[idx] = torch.nn.functional.pad(torch.tensor(target_input_batch[idx]), (0, max_target_length - len(target_input_batch[idx])))
                target_output_batch[idx] = torch.nn.functional.pad(torch.tensor(target_output_batch[idx]), (0, max_target_length - len(target_output_batch[idx])))

            return torch.stack(source_batch), torch.stack(target_input_batch), torch.stack(target_output_batch)
            
            


    def batch(self, batch_size=32):
        indices = np.arange(self.source_lines)
        np.random.shuffle(indices)

        cache_size = 200
        batch_queue = Queue(maxsize=cache_size)


        def fill_cache():
            local_indices = indices.copy()
            while len(local_indices) > 0:
                # print("Filling cache")
                chunk = local_indices[:batch_size]
                local_indices = local_indices[batch_size:]
                batch_data = self._get_batch(chunk)
                batch_queue.put(batch_data)
            batch_queue.put(None)

        threading.Thread(target=fill_cache, daemon=True).start()

        while len(indices) > 0:
            batch_indices = indices[:batch_size]
            indices = indices[batch_size:]
            
            the_batch = batch_queue.get()

            if the_batch is None:
                break

            yield the_batch 


if __name__ == "__main__":
    pass