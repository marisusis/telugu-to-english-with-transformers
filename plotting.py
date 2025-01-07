import numpy as np
import matplotlib.pyplot as plt


def plot_attention(file, attention, query_tokens=None, key_tokens=None):
    fig, ax = plt.subplots()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.80, bottom=0.1)
    plot = ax.pcolormesh(attention, cmap='plasma')
    # plot = ax.pcolormesh(torch.softmax(mask[0], -1))
    ax.xaxis.set_ticks_position('top')
    # ax.set_xlabel("English"
    ax.set_xticks(np.arange(attention.shape[1]) + 0.5)
    if query_tokens:
        ax.set_xticklabels(query_tokens, rotation=45)
    else:
        ax.set_xticklabels(range(attention.shape[1]))
        

    # ax.set_yticklabels(tokenizer_en.encode(phrase_en).tokens)
    ax.set_yticks(np.arange(attention.shape[0]) + 0.5)
    if key_tokens:
        ax.set_yticklabels(key_tokens, fontfamily="Kohinoor Telugu")
    else:
        ax.set_yticklabels(range(attention.shape[0]))

    ax.set_ylabel('Query Tokens')
    ax.set_xlabel('Key Tokens')

    ax.set_title("Attention")

    # ax.invert_yaxis()

    # ax.set_xticklabels(tokenizer_te.encode(phrase_te).tokens, fontfamily="Kohinoor Telugu")
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    fig.colorbar(plot)

    # save to file
    plt.savefig(file)