import json

json_value = None
Openness = []
Conscientiousness = []
Extraversion = []
Agreeableness = []
Emotional_range = []
for i in range(273):
	json_value = json.load(open('./result/result'+str(i)+'.txt'))
	Openness.append(json_value['personality'][0]['percentile'])
	Conscientiousness.append(json_value['personality'][1]['percentile'])
	Extraversion.append(json_value['personality'][2]['percentile'])
	Agreeableness.append(json_value['personality'][3]['percentile'])
	Emotional_range.append(json_value['personality'][4]['percentile'])
print(max(Openness),min(Openness))
print(max(Conscientiousness),min(Conscientiousness))
print(max(Extraversion),min(Extraversion))
print(max(Agreeableness),min(Agreeableness))
print(max(Emotional_range),min(Emotional_range))

