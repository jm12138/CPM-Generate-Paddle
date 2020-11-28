import paddle
import numpy as np
from GPT2 import GPT2Model, GPT2Tokenizer, LoadParams

model = GPT2Model(
    vocab_size=30000,
    layer_size=32,
    block_size=1024,
    embedding_dropout=0.0,
    embedding_size=2560,
    num_attention_heads=32,
    attention_dropout=0.0,
    residual_dropout=0.0)

state_dict = LoadParams('GPT2/pretrained_model')

model.set_dict(state_dict)

model.eval()

tokenizer = GPT2Tokenizer(
    'GPT2/bpe/vocab.json',
    'GPT2/bpe/chinese_vocab.model',
    max_len=512)

def sample(text, max_len=10):
    ids = tokenizer.encode(text)
    input_id = paddle.to_tensor(np.array(ids).reshape(1, -1).astype('int64'))
    output, cached_kvs = model(input_id, use_cache=True)
    nid = int(np.argmax(output[0, -1].numpy()))
    ids += [nid]
    out = [nid]
    for i in range(max_len):
        input_id = paddle.to_tensor(np.array([nid]).reshape(1, -1).astype('int64'))
        output, cached_kvs = model(input_id, cached_kvs, use_cache=True)
        nid = int(np.argmax(output[0, -1].numpy()))
        ids += [nid]
        if nid==3:
            break
        out.append(nid)
    # print(out)
    print(tokenizer.decode(out))

def ask_question(question, max_len=10):
    sample('''问题：中国的首都是哪里？
    答案：北京。
    问题：李白在哪个朝代？
    答案：唐朝。
    问题：%s
    答案：''' % question, max_len)

def dictation_poetry(front, max_len=10):
    sample('''默写古诗:
    白日依山尽，黄河入海流。
    %s，''' % front, max_len)

def math(inputs, max_len=10):
    sample('''1+1=2
    2+2=4
    %s''' % inputs, max_len)

ask_question('红楼梦谁写的？')

ask_question('美国的首都是哪里？')

dictation_poetry('床前明月光')

dictation_poetry('沉舟侧畔千帆过')

math('4+4=')