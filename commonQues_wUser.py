# 3shan ne3raf el most common question (including el user)

import pandas as pd
import numpy as np
import math

data = pd.read_csv("sample.csv")

def checkOtherUsers( current_problem, ids, dataset ):
	occurrence = 0
	for user in ids:
		index = dataset.index[dataset['user_id'] == user]
		temp_user = dataset.loc[index]
		index2 = temp_user.index[temp_user['problem_id'] == current_problem]
		real_temp = temp_user.loc[index2]
		if not real_temp.empty:
			occurrence = occurrence + 1

	return occurrence


def findUsers(question, ids, dataset):
	solved = []
	for user in ids:
		index = dataset.index[dataset['user_id'] == user]
		temp_user = dataset.loc[index]
		index2 = temp_user.index[temp_user['problem_id'] == current_problem]
		real_temp = temp_user.loc[index2]
		if not real_temp.empty:
			solved.append(user)
	return solved

def predictRating(users, similarities, ratings, mean):
	i = 0
	numerator = 0
	denomenator = 0
	for user in users:
		numerator = numerator + (similarities[i]*ratings[i])
		denomenator = denomenator + similarities[i]
	prediction = mean + (numerator/denomenator)
	return prediction

def userRatings(question, solvedBy, dataset):
	ratings = []
	for user in solvedBy:
		index = dataset.index[dataset['user_id'] == user]
		temp_user = dataset.loc[index]
		index2 = temp_user.index[temp_user['problem_id'] == question]
		final_user = temp_user.loc[index2]
		ratings.append(final_user['attempts_range'])
	return ratings

common_questions = [] #3shan ne7ot fih el common question beta3 kol user (if exists)

x = 0
while x < 300:
	#User targetted vars
	current_user = x
	problems_of_first_user = []
	mean_of_first_user = 0
	common_rating_first_user = 0
	changed = False
	threshold1 = 0.3
	threshold2 = 0.6
	sum_for_first_user_mean = 0
	x = x + 1
	index_for_first_user = data.index[data['user_id'] == x] 
	user1_table = data.loc[index_for_first_user]
	if user1_table.empty:
		# print(str(x) + " msh mawgood")
		common_questions.append(-1)
		# print(common_questions)
		continue

	#equation & similarity vars
	sum1_for_pearson = 0 
	sum2_for_pearson = 0
	sum3_for_pearson = 0
	similarity = 0

	#NN vars
	found_question = False
	question = 0
	closest_seven = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	closest_seven_ids = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	second_mean_count = 0 
	second_user_mean = 0
	second_mean_flag = False
	occurrence = 0 #represents the num of users who have a problem in common
	solvedBy = []
	solvedBy_similarity = []
	solvedBy_ratings = []
	best_question = 0
	# num_of_problems = 0 #still not used


	i = 0
	for index, row in data.iterrows(): #To get the mean of the 1st user and the problems he/she solved
	    if row['user_id'] > current_user and changed == False:
	    	current_user = row['user_id']
	    	changed = True
	    	sum_for_first_user_mean = sum_for_first_user_mean + row['attempts_range']
	    	i = i + 1
	    	problems_of_first_user.append(row['problem_id'])

	    elif row['user_id'] == current_user and changed == True:
	    	sum_for_first_user_mean = sum_for_first_user_mean + row['attempts_range']
	    	i = i + 1
	    	problems_of_first_user.append(row['problem_id'])
	    elif row['user_id'] > x:
	    	break

	mean_of_first_user = sum_for_first_user_mean/i
	second_user = 0

	i = 0
	j = 0

	for index, row in data.iterrows():

		if row['user_id'] == current_user:
			continue
		if row['user_id'] > second_user and second_mean_flag == False: #For when the 2nd user appears for the 1st time
			second_user = row['user_id']
			index_for_sec_user = data.index[data['user_id'] == row['user_id']] 
			temp_user = data.loc[index_for_sec_user] #Getting a sub dataset for the 2nd user
			for index2, row2 in temp_user.iterrows(): #Calculating the sum 
				second_mean_count = second_mean_count + row2['attempts_range']
				j = j + 1
			second_user_mean = second_mean_count/j
			j = 0
			second_mean_flag = True
			if row['problem_id'] in problems_of_first_user:
				index_for_first_user_prob = data.index[data['user_id'] == current_user]
				temp_user = data.loc[index_for_first_user_prob]
				real_index = temp_user.index[temp_user['problem_id'] == row['problem_id']]
				real_temp = temp_user.loc[real_index]
				for index2, row2 in real_temp.iterrows():
					common_rating_first_user = row2['attempts_range']
				sum1_for_pearson = sum1_for_pearson + ((common_rating_first_user - mean_of_first_user) * (row['attempts_range'] - second_user_mean))
				sum2_for_pearson = sum2_for_pearson + (common_rating_first_user - mean_of_first_user)**2
				sum3_for_pearson = sum3_for_pearson + (row['attempts_range'] - second_user_mean)**2

		elif row['user_id'] == second_user and second_mean_flag == True: #If the following entry is the same user
			second_user = row['user_id']
			if row['problem_id'] in problems_of_first_user:
				index_for_first_user_prob = data.index[data['user_id'] == current_user]
				temp_user = data.loc[index_for_first_user_prob]
				real_index = temp_user.index[temp_user['problem_id'] == row['problem_id']]
				real_temp = temp_user.loc[real_index]
				for index2, row2 in real_temp.iterrows():
					index = real_temp.index[real_temp['problem_id'] == row['problem_id']].tolist()
					common_rating_first_user = real_temp.at[index[0],'attempts_range']
				sum1_for_pearson = sum1_for_pearson + ((common_rating_first_user - mean_of_first_user) * (row['attempts_range'] - second_user_mean))
				sum2_for_pearson = sum2_for_pearson + (common_rating_first_user - mean_of_first_user)**2
				sum3_for_pearson = sum3_for_pearson + (row['attempts_range'] - second_user_mean)**2

		elif row['user_id'] > second_user and second_mean_flag == True: #When the 2nd user changes
			second_mean_flag = False
			similarity_check = math.sqrt(sum2_for_pearson)*math.sqrt(sum3_for_pearson)
			if(similarity_check != 0):
				similarity = sum1_for_pearson/(math.sqrt(sum2_for_pearson) * math.sqrt(sum3_for_pearson))
			else:
				similarity = 0

			smallest = 100.0
			for num in closest_seven:
				if abs(num) < abs(smallest):
					smallest = num		
			
			k = closest_seven.index(smallest)
			if  abs(similarity) > abs(smallest):
				if similarity < -1.0:
					similarity = -1.0
				elif similarity > 1.0:
					similarity = 1.0
				closest_seven[k] = similarity
				closest_seven_ids[k] = second_user
			
			sum1_for_pearson = 0
			sum2_for_pearson = 0
			sum3_for_pearson = 0
			second_mean_count = 0
			second_user_mean = 0
			j = 0
			second_user = row['user_id']
			index_for_sec_user = data.index[data['user_id'] == row['user_id']]
			temp_user = data.loc[index_for_sec_user]
			for index2, row2 in temp_user.iterrows():
				second_mean_count = second_mean_count + row2['attempts_range']
				j = j + 1
			second_mean_flag = True
			second_user_mean = second_mean_count/j
			j = 0
			if row['problem_id'] in problems_of_first_user:
				index_for_first_user_prob = data.index[data['user_id'] == current_user]
				temp_user = data.loc[index_for_first_user_prob]
				real_index = temp_user.index[temp_user['problem_id'] == row['problem_id']]
				real_temp = temp_user.loc[real_index]
				for row2, index2 in real_temp.iterrows():
					index = real_temp.index[real_temp['problem_id'] == row['problem_id']].tolist()
					common_rating_first_user = real_temp.at[index[0],'attempts_range']
				sum1_for_pearson = sum1_for_pearson + ((common_rating_first_user - mean_of_first_user) * (row['attempts_range'] - second_user_mean))
				sum2_for_pearson = sum2_for_pearson + (common_rating_first_user - mean_of_first_user)**2
				sum3_for_pearson = sum3_for_pearson + (row['attempts_range'] - second_user_mean)**2

	if closest_seven[0] == 0:
		# print("Mafeesh similar f " + str(x))
		common_questions.append(0)
		# print(common_questions)
		continue
	current_problem = 0
	current_number = 1
	for user in closest_seven_ids:
		
		number = 0
		if found_question == False:
			index = data.index[data['user_id'] == user]
			already_answered = False
			temp_user = data.loc[index]
			for index,row in temp_user.iterrows():
				for index2, row2 in user1_table.iterrows():
					if row['problem_id'] == row2['problem_id']:
						already_answered = True
						number = checkOtherUsers(row['problem_id'], closest_seven_ids, data)
						break
				
				if number > current_number & already_answered == True:
					current_number = number
					found_question = True
					question = row['problem_id']
				already_answered = False
		else:
			break
	i = 0
	if question == 0:
		# print("Ma7adesh 7all f " + str(x))
		common_questions.append(0)
		# print(common_questions)
		continue
	for user in closest_seven_ids:
			index = data.index[data['user_id'] == user]
			temp_user = data.loc[index]
			index2 = temp_user.index[temp_user['problem_id'] == question]
			real_temp = temp_user.loc[index2]
			if not real_temp.empty:
				solvedBy_similarity.append(closest_seven[i])
				solvedBy.append(user)
				for i2, r2 in real_temp.iterrows():
					solvedBy_ratings.append(r2['attempts_range'])
			i = i + 1

	common_questions.append(question)
	# print(common_questions)
	score = predictRating(solvedBy, solvedBy_similarity, solvedBy_ratings, mean_of_first_user)



