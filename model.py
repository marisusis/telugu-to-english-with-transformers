from layers import *

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, h, vocab_size, N, p_dropout, max_context):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_context)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, h, p_dropout) for _ in range(N)])

    def get_self_attention_weights(self):
        return [layer.get_last_attention_weights() for layer in self.encoder_layers]

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, h, vocab_size, N, p_dropout, max_context):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_context)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, h, p_dropout) for _ in range(N)])

    def get_self_attention_weights(self):
        return [layer.get_self_attention_weights() for layer in self.decoder_layers]
    
    def get_cross_attention_weights(self):
        return [layer.get_cross_attention_weights() for layer in self.decoder_layers]

    def forward(self, x, context):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, context)
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, h, in_vocab, out_vocab, N, max_context):
        super().__init__()
        self.encoder = Encoder(d_model, d_ff, h, in_vocab, N, 0.1, max_context)
        self.decoder = Decoder(d_model, d_ff, h, out_vocab, N, 0.1, max_context)
        self.output = nn.Linear(d_model, out_vocab)
    
    def decoder_embedding(self):
        return self.decoder.embedding
    
    def encoder_embedding(self):
        return self.encoder.embedding

    def forward(self, target, source, target_mask, source_mask):
        # print("x shape", x.shape, "y shape", y.shape)
        context = self.encoder(source, source_mask)
        # print("Context shape", context.shape)
        target = self.decoder(target, context)
        # print("Decoder shape", x.shape)
        target = self.output(target)
        target = torch.log_softmax(target, -1)
        return target