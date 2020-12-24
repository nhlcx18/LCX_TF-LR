import joblib

LABEL_INDEX = ['Cause-Effect', 'Instrument-Agency', 'Product-Producer',
               'Content-Container', 'Entity-Origin', 'Entity-Destination',
               'Component-Whole', 'Member-Collection', 'Message-Topic', 'Other']

model = joblib.load("./temp/models/base_model.joblib")
vectorizer = joblib.load("./temp/models/base_vectorizer.joblib")

text_list = []
entity_list = []
with open('./temp/test_generate_easy.txt', 'r', encoding='utf-8') as f:
    for line in f:
        data = line.strip().split('\t')
        entity_list.append((data[0],data[1]))
        text_list.append(data[3])
print('read_finish')

x = vectorizer.transform(text_list)

y = model.predict(x)
with open('submit/results.txt', 'w', encoding='utf-8') as f:
    for line in y:
        f.write(str(line) + '\n')

with open('submit/results1.txt', 'w', encoding='utf-8') as f:
    for line in y:
        f.write(LABEL_INDEX[int(line)] + '\n')

with open('submit/results2.txt', 'w', encoding='utf-8') as f:
    for i in range(2717):
        f.write(entity_list[i][0] + '\n')

with open('submit/results3.txt', 'w', encoding='utf-8') as f:
    for i in range(2717):
        f.write(entity_list[i][1] + '\n')

for i in range(10):
    print("文本：{}；\n实体一：{}；实体二：{}，关系预测：{}\n"\
          .format(text_list[i], entity_list[i][0], entity_list[i][1], LABEL_INDEX[int(y[i])]))