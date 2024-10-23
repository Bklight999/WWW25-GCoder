import json
with open('./dpo_datas_clustering.json','r') as f:
    datas1 = json.load(f)

with open('./dpo_datas_shortest.json','r') as f:
    datas2 = json.load(f)

with open('./dpo_datas_hamilton.json','r') as f:
    datas3 = json.load(f)

data=[]

data.extend(datas1)
data.extend(datas2)
data.extend(datas3)
print(len(datas1))
print(len(datas2))
print(len(datas3))
print(len(data))

with open('dpo_train_qwen.json','w') as f:
    json.dump(data,f)