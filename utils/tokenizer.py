"""
GPT-2 vocabulary(50257 tokens) where i add 3 special tokens:
<s>: start of sequence
</s>: end of sequence
<pad>: padding
"""

import tiktoken

def custom_encoding():
    cl100k_base = tiktoken.get_encoding("gpt2")

    enc = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<s>": 50257,
            "</s>": 50258,
            "<pad>": 50259,
        }
    )

    return enc

enc = custom_encoding()
