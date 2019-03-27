import pandas as pd
import numpy as np

data = pd.read_csv("problem_data.csv")
sample = data.sort_values(by=['problem_id'])

for index, row in sample.iterrows():
	current_problem = row['problem_id']
	split_problem = current_problem.split('_')
	sample['problem_id'].replace(current_problem,split_problem[1], inplace = True)

sample.problem_id = sample.problem_id.astype('int64')
sample = sample[sample.problem_id <= 600]
final_sample = sample.sort_values(by=['problem_id'])

final_sample.to_csv('problem_sample.csv', sep=',')