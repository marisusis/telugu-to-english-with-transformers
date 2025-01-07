from layers import *

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, h, vocab_size, N, p_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, h, p_dropout) for _ in range(N)])

    def get_self_attention_weights(self):
        return [layer.get_last_attention_weights() for layer in self.encoder_layers]

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, h, vocab_size, N, p_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
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
    def __init__(self, d_model, d_ff, h, in_vocab, out_vocab, N):
        super().__init__()
        self.encoder = Encoder(d_model, d_ff, h, in_vocab, N, 0.1)
        self.decoder = Decoder(d_model, d_ff, h, out_vocab, N, 0.1)
        self.output = nn.Linear(d_model, out_vocab)
    
    def decoder_embedding(self):
        return self.decoder.embedding
    
    def encoder_embedding(self):
        return self.encoder.embedding

    def forward(self, target, source):
        # print("x shape", x.shape, "y shape", y.shape)
        context = self.encoder(source)
        # print("Context shape", context.shape)
        target = self.decoder(target, context)
        # print("Decoder shape", x.shape)
        target = self.output(target)
        target = torch.log_softmax(target, -1)
        return target
    
def plot_attention(attention, queries, keys):
    fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(8, 6))
    plot = ax.pcolormesh(attention)
    # plot = ax.pcolormesh(torch.softmax(mask[0], -1))
    ax.xaxis.set_ticks_position('top')
    # ax.set_xlabel("English"
    ax.set_xticks(np.arange(context_size) + 0.5)
    ax.set_yticklabels(tokenizer_en.encode(phrase_en).tokens)
    ax.set_yticks(np.arange(context_size) + 0.5)
    matplotlib.text.Text
    ax.set_xticklabels(tokenizer_te.encode(phrase_te).tokens, fontfamily="Kohinoor Telugu")
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    fig.colorbar(plot)

    plt.show()