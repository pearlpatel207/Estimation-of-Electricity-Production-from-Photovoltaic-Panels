#setup
import numpy as np
import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
import tensorflow as tf

#Loading the dataset

PATH = 'PATH_TO_INPUT_FILE' #change the path to location of file

df = pd.read_csv(PATH)
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df['year'] = df['TimeStamp'].dt.year
df['date'] = df['TimeStamp'].dt.date


#split dataset in training and testing datasets
train = df.loc[(df.year < 2016) & (df.year > 2012)]
train = train.reset_index(drop=True)

test = df[df.year == 2016]
test = test.reset_index(drop=True)


#scaling the data

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = train
test_scaled = test

train_scaled[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year','hour_of_day']] = scaler.fit_transform(train[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year','hour_of_day']])
test_scaled[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year','hour_of_day']] = scaler.transform(test[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year','hour_of_day']])

yscaler = MinMaxScaler(feature_range=(0, 1))
train_scaled[['dc_pow']] = yscaler.fit_transform(train[['dc_pow']])
test_scaled[['dc_pow']] = yscaler.transform(test[['dc_pow']])

Irrscaler = MinMaxScaler(feature_range=(0, 1))
train_scaled[['Irr']] = Irrscaler.fit_transform(train[['Irr']])
test_scaled[['Irr']] = Irrscaler.transform(test[['Irr']])


#separating features and labels for irradiance prediction

#training data
trainf = train_scaled[['uvIndex.1', 'cloudcover', 'humidity', 'tempC', 'visibility', 'day_of_year','hour_of_day',  'Irr']].copy()
traint = train_scaled[['Irr']]

train_dataset = trainf.values
train_target = traint.values

#testing data
testf = test_scaled[['uvIndex.1', 'cloudcover', 'humidity', 'tempC', 'visibility', 'day_of_year','hour_of_day', 'Irr']].copy()
testt = test_scaled[['Irr']]

test_dataset = testf.values
test_target = testt.values


#function for windowing the dataset for LSTM Model
def window_dataset(dataset, target, history_size,
                      target_size,):

	'''
		The LSTM model makes predictions (target) based on a window of consecutive samples from the data (dataset)
		history_size specifies the number of past samples to be considered for predictions
		target_size specifies the time offset between past sample and predictions

	'''
  data = []
  labels = []

  for i in range(history_size, len(dataset)-target_size):
    indices = range(i-history_size, i, 1)
    data.append(dataset[indices])

    labels.append(target[i+target_size])

  return np.array(data), np.array(labels)

HISTORY = 1
TARGET = 0

x_train, y_train = window_dataset(train_dataset, train_target, HISTORY, TARGET)
x_test, y_test = window_dataset(test_dataset, test_target, HISTORY, TARGET)


#convert the data into TensorFlow Dataset to feed it into TensorFlow Model

BATCH_SIZE = 256
BUFFER_SIZE = 10000

print ('Single window of past history : {}'.format(x_train[0].shape))
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_data = val_data.batch(BATCH_SIZE).repeat()


#neural network for irradiance prediction
EPOCHS = 300
es = EarlyStopping(monitor='val_loss', mode='min', patience=60, min_delta=0.0001, verbose=1,restore_best_weights=True)

Irr_model = tf.keras.models.Sequential()
Irr_model.add(tf.keras.layers.LSTM(32, input_shape=x_train.shape[-2:], return_sequences=True))
Irr_model.add(tf.keras.layers.LSTM(32, activation="relu"))
Irr_model.add(tf.keras.layers.Dense(16, activation="relu"))
Irr_model.add(tf.keras.layers.Dense(8, activation="relu"))
Irr_model.add(tf.keras.layers.Dense(1))

Irr_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

Irr_history = Irr_model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=200,
                                            validation_data=val_data,
                                            validation_steps=50,
                                            callbacks=[es]
                                        )


#add the predicted irradiace values to the dataset

y = Irr_model.predict(x_test)
Irr_cal = Irr_model.predict(x_train)

test_scaled.drop([0], inplace = True)
train_scaled.drop([0], inplace = True)

test_scaled['Irr_cal'] = y
train_scaled['Irr_cal'] = Irr_cal


# sampling data for hourly predictions

train_scaled = train_scaled.groupby(['date','hour_of_day']).first().reset_index()
test_scaled = test_scaled.groupby(['date','hour_of_day']).first().reset_index()


# separating features and labels for hourly power production

xtrain = train_scaled[['Irr_cal', 'uvIndex.1', 'tempC', 'cloudcover', 'humidity', 'day_of_year','hour_of_day', 'dc_pow']].copy()
ytrain = train_scaled[['dc_pow']].copy()

xtest = test_scaled[['Irr_cal', 'uvIndex.1', 'tempC', 'cloudcover', 'humidity', 'day_of_year','hour_of_day', 'dc_pow']].copy()
ytest = test_scaled[['dc_pow']].copy()

trainx = xtrain.values
trainy = ytrain.values

testx = xtest.values
testy = ytest.values


# windowing dataset

HISTORY = 1     #to include more historical data for better predictions, increase HISTORY value
TARGET = 0      #for predictions of power produced 3 hours later, set TARGET = 3

x_train, y_train = window_dataset(trainx, trainy, HISTORY, TARGET)
x_test, y_test = window_dataset(testx, testy, HISTORY, TARGET)


#convert the data into TensorFlow Dataset to feed it into TensorFlow Model

print ('Single window of past history : {}'.format(x_train[0].shape))
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_data = val_data.batch(BATCH_SIZE).repeat()


# LSTM based Neural Network for hourly power production

es = EarlyStopping(monitor='val_loss', mode='min', patience=60, min_delta=0.0001, verbose=1,restore_best_weights=True)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, input_shape=x_train.shape[-2:], return_sequences=True))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.LSTM(32, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(8, activation="tanh"))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

history = model.fit(train_data, epochs =    EPOCHS,
                                            steps_per_epoch=200,
                                            validation_data=val_data,
                                            validation_steps=50,
                                            callbacks=[es]
                                        )


#test the model

y = model.predict(x_test)
mae = mean_absolute_error(y_test, y)
rmse = np.sqrt(mean_squared_error(y_test, y))
print('Test MAE: %.3f' %mae)
print('Test RMSE: %.3f' %rmse)


#save the model for future use
model.save('weights.h5')