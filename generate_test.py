import numpy as np
from model.rnnlm import BetterRnnlmGen
from data import load_dataset

data = load_dataset("./data/filtered")

model = BetterRnnlmGen()
model.load_params('./trained-models/Korean_essay.pkl')
model.reset_state()

start_words = '<type>주장<sub>개 식용 금지 입법에 관한 자신의 생각 작성<gen>'
start_ids = data.tokenizer.encode(start_words)[:-1]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], [])
print(data.tokenizer.decode(word_ids))
print()
