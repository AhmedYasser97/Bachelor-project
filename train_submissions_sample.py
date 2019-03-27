import pandas as pd
import numpy as np

sample = pd.read_csv("train_submissions.csv")
sorted_sample = sample.sort_values(by=['user_id'])

for index, row in sorted_sample.iterrows():
	current_user = row['user_id'] 
	current_problem = row['problem_id']
	split_user = current_user.split('_')
	split_problem = current_problem.split('_')
	sorted_sample['user_id'].replace(current_user,split_user[1], inplace = True)
	sorted_sample['problem_id'].replace(current_problem,split_problem[1], inplace = True)

sorted_sample.user_id = sorted_sample.user_id.astype('int64') 
sorted_sample.problem_id = sorted_sample.problem_id.astype('int64')
sorted_sample = sorted_sample[sorted_sample.user_id <= 300]
sorted_sample = sorted_sample[sorted_sample.problem_id <= 600]

size = (sorted_sample.size) // 3

final_sample = sorted_sample.sample(n=size)
final_sample = final_sample.sort_values(by=['user_id'])

final_sample.to_csv('sample.csv', sep=',')




