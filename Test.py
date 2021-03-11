import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import platform
import utils.cpuinfo as cpuinfo
from psutil import virtual_memory
import pandas as pd
import numpy as np
import time
import math
from utils.bert_op import pad_sequences, flat_accuracy, format_time


# 테스트 데이터 로드
test = pd.read_csv("./data/Test.csv",
                   encoding='cp949', dtype={'rejectionContentDetail': np.str, 'label': np.int})

# 불러올 모델 경로
weightPath = './weight/model_state_dict_epoch_9.pt'

# 테스트 하드웨어 환경 출력
def get_current_enviroment():
    OS_NAME = platform.system()
    OS_RELEASE = platform.release()
    OS_VERSION = platform.version()
    OS = f'{OS_NAME} {OS_RELEASE} {OS_VERSION}'
    CPU = cpuinfo.cpu.info[0]['ProcessorNameString']
    MEMORY = virtual_memory()
    MEMORY = f'{round(MEMORY.total / math.pow(1024,3),4)}GB'
    GPU = torch.cuda.get_device_name(torch.cuda.current_device())

    name_list = ['OS', 'CPU', 'MEMORY', 'GPU']
    device_list = [OS, CPU, MEMORY, GPU]

    df = pd.DataFrame({'Hardware':name_list, 'Hardware info':device_list}, columns=['Hardware', 'Hardware info'])
    df.to_csv('./output/Hardware_information.csv', index=False, encoding='euc-kr')

    print(OS_NAME, OS_RELEASE, OS_VERSION, CPU, MEMORY, GPU)

### 전처리_test
sentences = test['rejectionContentDetail']
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

labels = test['label'].values

print("Tokenizing test data ...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# 입력 토큰의 최대 시퀀스 길이
MAX_LEN = 128

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# 어텐션 마스크 초기화
attention_masks = []

for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

# 데이터를 파이토치의 텐서로 변환
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)

# 배치 사이즈
batch_size = 16

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

### 모델 불러오기
# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
model.cuda()
model.load_state_dict(torch.load(weightPath))

### 테스트셋 평가
t0 = time.time()
model.eval()

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# 예측값 출력을 위한 df 생성
final = pd.DataFrame(columns=['label', 'logits'])

for step, batch in enumerate(test_dataloader):
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    batch = tuple(t.to(device).long() for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

    logits = np.argmax(logits, axis=1).flatten
    correct = label_ids.flatten
    final = final.append(pd.DataFrame([[correct, logits]], columns=['label', 'logits']), ignore_index=True)

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))

final['label'] = [i[1:-1].split() for i in final['label']]
final['logits'] = [i[1:-1].split() for i in final['logits']]
pd.DataFrame.from_dict({'label': np.concatenate(final['label'].values), 'logits':np.concatenate(final['logits'].values)}, orient='index').transpose().to_csv('./output/Output.csv', index=False, encoding='cp949')

Accuracy = pd.DataFrame({'Accuracy': [eval_accuracy / nb_eval_steps]})
Accuracy.to_csv('./output/Accuracy.csv',
                index=False, encoding='cp949')