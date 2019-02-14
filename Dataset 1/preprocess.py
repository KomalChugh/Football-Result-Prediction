import pandas as pd
import numpy as np
 #!/usr/bin/env python -W ignore::DeprecationWarning
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import math
import pickle
import os
from sklearn.utils import resample
	
def drop_col(data):
	data = data.drop( ['id'] , 1 )
	data = data.drop( ['team_fifa_api_id'] , 1 )
	data = data.drop( ['team_api_id'] , 1 )
	data = data.drop( ['date'] , 1 )
	data = data.drop( ['id:1'] , 1 )
	data = data.drop( ['date:1'] , 1 )
	data = data.drop( ['match_api_id'] , 1 )
	data = data.drop( ['home_team_api_id'] , 1 )
	data = data.drop( ['away_team_api_id'] , 1 )
	data = data.drop( ['home_player_X1'] , 1 )
	data = data.drop( ['home_player_X2'] , 1 )
	data = data.drop( ['home_player_X3'] , 1 )
	data = data.drop( ['home_player_X4'] , 1 )
	data = data.drop( ['home_player_X5'] , 1 )
	data = data.drop( ['home_player_X6'] , 1 )
	data = data.drop( ['home_player_X7'] , 1 )
	data = data.drop( ['home_player_X8'] , 1 )
	data = data.drop( ['home_player_X9'] , 1 )
	data = data.drop( ['home_player_X10'] , 1 )
	data = data.drop( ['home_player_X11'] , 1 )
	data = data.drop( ['away_player_X1'] , 1 )
	data = data.drop( ['away_player_X2'] , 1 )
	data = data.drop( ['away_player_X3'] , 1 )
	data = data.drop( ['away_player_X4'] , 1 )
	data = data.drop( ['away_player_X5'] , 1 )
	data = data.drop( ['away_player_X6'] , 1 )
	data = data.drop( ['away_player_X7'] , 1 )
	data = data.drop( ['away_player_X8'] , 1 )
	data = data.drop( ['away_player_X9'] , 1 )
	data = data.drop( ['away_player_X10'] , 1 )
	data = data.drop( ['away_player_X11'] , 1 )
	data = data.drop( ['home_player_Y1'] , 1 )
	data = data.drop( ['home_player_Y2'] , 1 )
	data = data.drop( ['home_player_Y3'] , 1 )
	data = data.drop( ['home_player_Y4'] , 1 )
	data = data.drop( ['home_player_Y5'] , 1 )
	data = data.drop( ['home_player_Y6'] , 1 )
	data = data.drop( ['home_player_Y7'] , 1 )
	data = data.drop( ['home_player_Y8'] , 1 )
	data = data.drop( ['home_player_Y9'] , 1 )
	data = data.drop( ['home_player_Y10'] , 1 )
	data = data.drop( ['home_player_Y11'] , 1 )
	data = data.drop( ['away_player_Y1'] , 1 )
	data = data.drop( ['away_player_Y2'] , 1 )
	data = data.drop( ['away_player_Y3'] , 1 )
	data = data.drop( ['away_player_Y4'] , 1 )
	data = data.drop( ['away_player_Y5'] , 1 )
	data = data.drop( ['away_player_Y6'] , 1 )
	data = data.drop( ['away_player_Y7'] , 1 )
	data = data.drop( ['away_player_Y8'] , 1 )
	data = data.drop( ['away_player_Y9'] , 1 )
	data = data.drop( ['away_player_Y10'] , 1 )
	data = data.drop( ['away_player_Y11'] , 1 )
	data = data.drop( ['home_player_1'] , 1 )
	data = data.drop( ['home_player_2'] , 1 )
	data = data.drop( ['home_player_3'] , 1 )
	data = data.drop( ['home_player_4'] , 1 )
	data = data.drop( ['home_player_5'] , 1 )
	data = data.drop( ['home_player_6'] , 1 )
	data = data.drop( ['home_player_7'] , 1 )
	data = data.drop( ['home_player_8'] , 1 )
	data = data.drop( ['home_player_9'] , 1 )
	data = data.drop( ['home_player_10'] , 1 )
	data = data.drop( ['home_player_11'] , 1 )
	data = data.drop( ['away_player_1'] , 1 )
	data = data.drop( ['away_player_2'] , 1 )
	data = data.drop( ['away_player_3'] , 1 )
	data = data.drop( ['away_player_4'] , 1 )
	data = data.drop( ['away_player_5'] , 1 )
	data = data.drop( ['away_player_6'] , 1 )
	data = data.drop( ['away_player_7'] , 1 )
	data = data.drop( ['away_player_8'] , 1 )
	data = data.drop( ['away_player_9'] , 1 )
	data = data.drop( ['away_player_10'] , 1 )
	data = data.drop( ['away_player_11'] , 1 )
	data = data.drop( ['team_fifa_api_id.1'] , 1 )
	data = data.drop( ['team_api_id.1'] , 1 )
	data = data.drop( ['date.1'] , 1 )
	data = data.drop( ['goal'] , 1 )
	data = data.drop( ['shoton'] , 1 )
	data = data.drop( ['shotoff'] , 1 )
	data = data.drop( ['foulcommit'] , 1 )
	data = data.drop( ['card'] , 1 )
	data = data.drop( ['cross'] , 1 )
	data = data.drop( ['corner'] , 1 )
	data = data.drop( ['possession'] , 1 )
	return data

def preprocess_features(X):
    # Preprocesses the football data and converts catagorical variables into dummy variables.
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)
    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
        # Collect the revised columns
        output = output.join(col_data)
    return output


# standardizes the test data with means and standard deviations obtained from the train data
def getStandardizedTestData(data , means , stds):
	for i in range(len(data[0])):
		mean = means[i]
		std = stds[i]
		data[:,i] = data[:,i] - mean
		if std!=0:
			data[:,i] = data[:,i] / std
	return data


# standardizes the train data
def getStandardizedTrainData(data):
	means = []
	stds = []
	for i in range(len(data[0])):
		mean = data[:,i].mean()
		means.append(mean)
		std = data[:,i].std()
		stds.append(std)
		data[:,i] = data[:,i] - mean
		if std!=0:
			data[:,i] = data[:,i] / std
	return data,means,stds


def downSample(data):
	df_majority = data[ data['home_team_goal'] - data['away_team_goal'] >= 0 ]
	df_minority = data[ data['home_team_goal'] - data['away_team_goal'] < 0 ]
	# Downsample majority class
	df_majority = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority),     # to match minority class
                                 random_state=12)
	data = pd.concat([df_majority, df_minority])
	return data

def preprocess():
	pd.options.mode.chained_assignment = None
	if os.path.exists("data_set_all.pkl"):
		save_file = open("data_set_all.pkl","rb") # load the existing model
		data = pickle.load(save_file)
	else:
		data = pd.read_csv('dataset1.csv' )
		data = drop_col(data)
		data = data.fillna(data.mean())
		save_file = open("data_set_all.pkl","wb") # save the model
		pickle.dump(data,save_file)
		save_file.close()
	
	# check if anything is nan
	for key in data:
		count=0
		for val in data[key]:
			if isinstance(val, float):
				if math.isnan(val):
					count+=1
		if count!=0:
			print(key,count)
	
	# solve the class imbalance problem
	data = downSample(data)
	
	y_all = []
	for i in data.index:
		if data['home_team_goal'][i] - data['away_team_goal'][i] > 0 :
			y_all.append(1)
		else:
			y_all.append(-1)
	
	y_all = np.array( y_all )
	X_all = data.drop( ['home_team_goal'] , 1 ) 
	X_all = X_all.drop( ['away_team_goal'] , 1 ) 
	X_all = X_all.drop( ['id.1'] , 1)
	n_homewins = len( y_all[y_all == 1 ]) 
	n_matches = X_all.shape[0]
	win_rate = (float(n_homewins) / (n_matches)) * 100
	print( "Total number of matches: {}".format(n_matches))
	print( "Number of matches won by home team: {}".format(n_homewins))
	print( "Win rate of home team: {:.2f}%".format(win_rate))
	print( "Initial feature columns ({} total features)".format(len(X_all.columns) ))
	X_all = preprocess_features(X_all)
	print( "Processed feature columns ({} total features)\n".format(len(X_all.columns) ))
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = .5,
                                                    random_state = 2,
                                                    stratify = y_all)
	
	X_train = X_train.values
	y_train = y_train.astype(np.float64)
	X_test = X_test.values
	y_test = y_test.astype(np.float64)
	X_train,means,stds = getStandardizedTrainData(X_train)
	X_test = getStandardizedTestData(X_test , means , stds)	
	y_train = y_train.reshape ( (len(y_train) ,1) )
	y_test = y_test.reshape ( (len(y_test) ,1) )
	return 	X_train, y_train , X_test, y_test
