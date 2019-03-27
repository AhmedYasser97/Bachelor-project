#Beya5od el question wel closest users w yesheel el question men 3and el user w ye7seb el closest neighbors tany
import pandas as pd
import numpy as np
import math
# import testRecommender
# from testRecommender import *

data = pd.read_csv("sample.csv")
# print("3addeina " + str(len(testRecommender.common_questions)))
# print(testRecommender.common_questions)
#User targetted vars
common_questions = [164, 522, 0, 199, 53, 29, 199, 332, 53, 127, 29, 190, 164, 246, 0, 361, 522, 32, 81, 246, 522, 534,211, 127, 0, 133, 304, 127, -1, 332, 164, 121, 164, 0, 164,29, 29, 0, 332, 542, 0, -1, 304, 334, 137, 542, 522, 164, 421, 334, 0, 0, 522, 522, 0, 334, 137, 164, -1, 334, 454, 127, 127, 0, 334, -1, 388, 53, 334, 137, 332, -1, 574, 583, 0, 199, 164, 114, 137, 137, 520, -1, 334, 121, 164, 304, 332, 0, -1, 131, 0, 121, 574, 81, 332, 421, -1, 0, -1, 0, 0, -1, 81, 522, 361, 0, 0, 534, 32, 0, 137, 199, 0, -1, 83, 127, 332, 271, 211, 318, 334, 190, 137, 0, 127, 454, 137, 127, 522, 127, 334, 522, 81, 522, 131, 522, 332, 32, 334, -1, 334, 537, 421, 29, -1, 334, 246, -1, 95, 133, -1, 164, 522, -1, 454, 346, 0, 307, -1, 574, 246, 0, 304, 0, 53, 0, 127, 164, 29, 137, 164, 137, 334, 522, 164, -1, 137, 164, 246, 334, -1, 454, 75, 211, 336, 95, 164, 332, 81, 0, 0, 307, 522, 0, -1, 133, 81, 0, 0, 454, 282, 133, 137, 522, -1, 29, 522, 0, -1, 74, 421, 522, 164, 332, 137, -1, 137, 81, 318, 32, 522, 542, 0, 522, 164, 522, 307, 164, 32, -1, 32, 304, 0, 81, 0, 0, 304, 332, 0, 127, 318, 537, 332, 0, 0, 95, 70, 334, 29, 583, 121, 249, 0, 164, 542, 53, 164, 421, 522, 534, 95, 32, 332, 53, 304, -1, 334, 137,-1, 318, 332, 0, 164, 95, 133, 114, 53, -1, 522, 0, 332, 318, 346, 0, 542, 127, 522, 332, 127, 137, 332, 318, 127, -1, 53, 332, 0, -1, 164, 249]
non_existing_users = 0
no_similarity_users = 0
close_to_real = 0
far_from_real = 0
x = 0

while x < 300:
	# print(x)
	if common_questions[x] == -1:
		non_existing_users = non_existing_users + 1
		x = x + 1
		continue

	if common_questions[x] == 0:
		no_similarity_users = no_similarity_users + 1
		x = x + 1
		continue

	problems_of_first_user = []
	mean_of_first_user = 0
	common_rating_first_user = 0
	changed = False
	threshold1 = 0.3
	threshold2 = 0.6
	sum_for_first_user_mean = 0
	current_user = x
	index_for_first_user = data.index[data['user_id'] == x+1] 
	user1_table = data.loc[index_for_first_user]
	user1_table_size = user1_table.size//3 

	#equation & similarity vars
	sum1_for_pearson = 0 
	sum2_for_pearson = 0
	sum3_for_pearson = 0
	similarity = 0
	target_question = common_questions[x]

	#NN vars
	found_question = False
	question = 0
	closest_seventeen = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	closest_seventeen_ids = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	second_mean_count = 0 
	second_user_mean = 0
	second_mean_flag = False
	occurrence = 0 #represents the num of users who have a problem in common
	solvedBy = []
	solvedBy_similarity = []
	solvedBy_ratings = []
	best_question = 0
	i = 0
	for index, row in data.iterrows(): #To get the mean of the 1st user and the problems he/she solved
	    if (row['problem_id'] == target_question and row['user_id'] > current_user and changed == False) or (row['problem_id'] == target_question and row['user_id'] == current_user and changed == True):
	    	continue
	    elif row['user_id'] > current_user and changed == False:
	    	current_user = row['user_id']
	    	changed = True
	    	sum_for_first_user_mean = sum_for_first_user_mean + row['attempts_range']
	    	i = i + 1
	    	problems_of_first_user.append(row['problem_id'])

	    elif row['user_id'] == current_user and changed == True:
	    	sum_for_first_user_mean = sum_for_first_user_mean + row['attempts_range']
	    	i = i + 1
	    	problems_of_first_user.append(row['problem_id'])
	    elif row['user_id'] > x + 1:
	    	break
	print(problems_of_first_user)
	mean_of_first_user = sum_for_first_user_mean/i
	second_user = 0
	i = 0
	j = 0
	count_till_end_of_user = 0
	first_time = False #3shan law 5allas 2, ye7seb 3and 3 awel marra bas
	question_exists = False

	for index, row in data.iterrows():

		if row['user_id'] == current_user and first_time == True and count_till_end_of_user < user1_table_size:
			second_mean_flag = False
			count_till_end_of_user = count_till_end_of_user + 1
			if count_till_end_of_user == user1_table_size-1:
				first_time = False
			continue
		if row['user_id'] > second_user and second_mean_flag == False: #For when the 2nd user appears for the 1st time
			second_user = row['user_id']
			index_for_sec_user = data.index[data['user_id'] == row['user_id']] 
			temp_user = data.loc[index_for_sec_user] #Getting a sub dataset for the 2nd user
			for index2, row2 in temp_user.iterrows(): #Calculating the sum 
				if row2['problem_id'] == target_question:
					question_exists = True
				second_mean_count = second_mean_count + row2['attempts_range']
				j = j + 1
			second_user_mean = second_mean_count/j
			j = 0
			second_mean_flag = True
			if row['problem_id'] in problems_of_first_user and question_exists == True:
				index_for_first_user_prob = data.index[data['user_id'] == current_user]
				temp_user = data.loc[index_for_first_user_prob]
				real_index = temp_user.index[temp_user['problem_id'] == row['problem_id']]
				real_temp = temp_user.loc[real_index]
				for index2, row2 in real_temp.iterrows():
					common_rating_first_user = row2['attempts_range']
				sum1_for_pearson = sum1_for_pearson + ((common_rating_first_user - mean_of_first_user) * (row['attempts_range'] - second_user_mean))
				sum2_for_pearson = sum2_for_pearson + (common_rating_first_user - mean_of_first_user)**2
				sum3_for_pearson = sum3_for_pearson + (row['attempts_range'] - second_user_mean)**2


		elif row['user_id'] == second_user and second_mean_flag == True and question_exists == True: #If the following entry is the same user
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

		elif row['user_id'] > second_user and second_mean_flag == True and first_time == False : #When the 2nd user changes
			if(row['user_id'] == current_user):
				first_time = True
			second_mean_flag = False	
			similarity_check = math.sqrt(sum2_for_pearson)*math.sqrt(sum3_for_pearson)
			if(similarity_check != 0):
				similarity = sum1_for_pearson/(math.sqrt(sum2_for_pearson) * math.sqrt(sum3_for_pearson))
			else:
				similarity = 0
			smallest = 100.0
			for num in closest_seventeen:
				if abs(num) < abs(smallest):
					smallest = num		
			k = closest_seventeen.index(smallest)
			if  abs(similarity) > abs(smallest):
				if similarity < -1.0:
					similarity = -1.0
				elif similarity > 1.0:
					similarity = 1.0
				closest_seventeen[k] = similarity
				closest_seventeen_ids[k] = second_user
			question_exists = False
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
				if row2['problem_id'] == target_question:
					question_exists = True
				second_mean_count = second_mean_count + row2['attempts_range']
				j = j + 1
			second_mean_flag = True
			second_user_mean = second_mean_count/j
			j = 0
			if row['problem_id'] in problems_of_first_user and question_exists == True:
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
	# print(closest_seventeen)
	current_problem = 0
	l = 0
	numerator = 0
	denomenator = 0
	if closest_seventeen[0] == 0:
		no_similarity_users = no_similarity_users + 1
		x = x + 1
		continue
	for eachOne in closest_seventeen:
		if eachOne == 0:
			break
		index = data.index[data['user_id'] == closest_seventeen_ids[l]]
		temp_user = data.loc[index]
		real_index = temp_user.index[temp_user['problem_id'] == target_question]
		real_temp = temp_user.loc[real_index]
		for i, r in real_temp.iterrows():
			val = float(r['attempts_range'])
			numerator = numerator + (eachOne*val)
		denomenator = denomenator + eachOne
	prediction_test = mean_of_first_user + (numerator/denomenator)
	index_to_check = data.index[data['user_id'] == x + 1]
	temp_user = data.loc[index_to_check]
	real_index = temp_user.index[temp_user['problem_id'] == common_questions[x]]
	real_temp = temp_user.loc[real_index]
	real_value = 0
	for i, r in real_temp.iterrows():
			real_value = r['attempts_range']
	# print("prediction: " + str(prediction_test) + " & truth is: " + str(real_value))
	difference = abs(prediction_test - real_value)
	if difference > 1.2:
		far_from_real = far_from_real + 1
	else:
		close_to_real = close_to_real + 1
	x = x + 1

print("Close: " + str(close_to_real))
print("Far: " + str(far_from_real))
print("Don't exist " + str(non_existing_users))
print("Not similar " + str(no_similarity_users))
# def checkOtherUsers( current_problem, ids, dataset ):
# 	occurrence = 0
# 	for user in ids:
# 		index = dataset.index[dataset['user_id'] == user]
# 		temp_user = dataset.loc[index]
# 		index2 = temp_user.index[temp_user['problem_id'] == current_problem]
# 		real_temp = temp_user.loc[index2]
# 		if not real_temp.empty:
# 			occurrence = occurrence + 1

# 	return occurrence


# def findUsers(question, ids, dataset):
# 	solved = []
# 	for user in ids:
# 		index = dataset.index[dataset['user_id'] == user]
# 		temp_user = dataset.loc[index]
# 		index2 = temp_user.index[temp_user['problem_id'] == current_problem]
# 		real_temp = temp_user.loc[index2]
# 		if not real_temp.empty:
# 			solved.append(user)
# 	return solved

# def predictRating(users, similarities, ratings, mean):
# 	i = 0
# 	numerator = 0
# 	denomenator = 0
# 	for user in users:
# 		numerator = numerator + (similarities[i]*ratings[i])
# 		denomenator = denomenator + similarities[i]
# 	prediction = mean + (numerator/denomenator)
# 	return prediction

# for user in closest_seven_ids:
# 	current_number = 4
# 	# if found_question == False:
# 	index = data.index[data['user_id'] == user]
# 	already_answered = False
# 	temp_user = data.loc[index]
# 	for index,row in temp_user.iterrows():
# 		# for index2, row2 in user1_table.iterrows():
# 		# 	if row['problem_id'] == row2['problem_id']:
# 		# 		already_answered = True
# 		number = checkOtherUsers( row['problem_id'], closest_seven_ids, data )
# 		if number > current_number :
# 			print(number)
# 			current_number = number
# 			found_question = True
# 			print(row['problem_id'])
# 			question = row['problem_id']
# 			# break
# 		already_answered = False
# 	# else:
# 	# 	break
# print("question is " + str(question))
# i = 0
# for user in closest_seven_ids:
# 		index = data.index[data['user_id'] == user]
# 		temp_user = data.loc[index]
# 		index2 = temp_user.index[temp_user['problem_id'] == question]
# 		real_temp = temp_user.loc[index2]
# 		if not real_temp.empty:
# 			solvedBy_similarity.append(closest_seven[i])
# 			solvedBy.append(user)
# 			for i2, r2 in real_temp.iterrows():
# 				solvedBy_ratings.append(r2['attempts_range'])
# 		i = i + 1

# print(solvedBy_similarity)
# print(solvedBy)
# print(solvedBy_ratings)
# score = predictRating(solvedBy, solvedBy_similarity, solvedBy_ratings, mean_of_first_user)
# print(score)


# def userRatings(question, solvedBy, dataset):
# 	ratings = []
# 	for user in solvedBy:
# 		index = dataset.index[dataset['user_id'] == user]
# 		temp_user = dataset.loc[index]
# 		index2 = temp_user.index[temp_user['problem_id'] == question]
# 		final_user = temp_user.loc[index2]
# 		ratings.append(final_user['attempts_range'])
# 	return ratings

