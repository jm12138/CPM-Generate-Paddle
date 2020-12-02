import paddle
import numpy as np
from GPT2 import GPT2Model, GPT2Tokenizer

# 初始化GPT-2模型
model = GPT2Model(
    vocab_size=30000,
    layer_size=32,
    block_size=1024,
    embedding_dropout=0.0,
    embedding_size=2560,
    num_attention_heads=32,
    attention_dropout=0.0,
    residual_dropout=0.0)

print('正在加载模型，耗时需要几分钟，请稍后...')

# 读取CPM模型参数(FP16)
state_dict = paddle.load('data/data62514/CPM-LM.pdparams')

# FP16 -> FP32
for param in state_dict:
    state_dict[param] = state_dict[param].astype('float32')

# 加载CPM参数
model.set_dict(state_dict)

# 将模型设置为评估状态
model.eval()

# 加载编码器
tokenizer = GPT2Tokenizer(
    'GPT2/bpe/vocab.json',
    'GPT2/bpe/chinese_vocab.model',
    max_len=512)

# 初始化编码器
_ = tokenizer.encode('初始化')

print('模型加载完成.')

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

mode = 'q'
funs = ask_question
print('输入“切换”更换问答和古诗默写模式，输入“exit”退出')
while True:
    if mode == 'q':
        inputs = input("当前为问答模式，请输入问题：")
    else:
        inputs = input("当前为古诗默写模式，请输入古诗的上半句：")
    if inputs=='切换':
        if mode == 'q':
            mode = 'd'
            funs = dictation_poetry
        else:
            mode = 'q'
            funs = ask_question
    elif inputs=='exit':
        break
    else:
        funs(inputs)