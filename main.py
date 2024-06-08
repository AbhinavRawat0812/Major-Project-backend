from __future__ import print_function
import matplotlib.pyplot as plt
import Functions as l4f
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import numpy as np
import pickle
from joblib import Parallel, delayed 
import joblib 

# import data
data = l4f.import_data()

# prepare data
data_train = data[data['id']>1]
data_test = data[data['id']<=1]
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)

plt.figure()
sns.countplot(x='activity',
              data=data,
              order = data.activity.value_counts().index)
plt.title("Records per activity")
plt.figure()
sns.countplot(x='id',
              data=data,
              palette=[sns.color_palette()[0]],
              order = data.id.value_counts().index
              )
plt.title("Records per user")

# prepare dataset
TIME_STEPS = 130
STEP = 14

X_train, y_train = l4f.create_dataset(
    data_train[['x','y','z']],
    data_train.activity,
    TIME_STEPS,
    STEP
)

X_test,y_test = l4f.create_dataset(
    data_test[['x','y','z']],
    data_test.activity,
    TIME_STEPS,
    STEP
)


# encoding of the classes - transform them into binary
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

# Train the RNN
model = l4f.create_model(X_train,y_train)

# fit a model
history = model.fit(
    X_train,y_train,
    epochs=20,
    validation_split=0.1,
)

# plt.figure()
# plt.plot(history.history['acc'],label='train')
# plt.plot(history.history['val_acc'], label='test')
# plt.ylabel('Accuracy [%]')
# plt.xlabel('Epochs []')
# plt.legend()


# Model Save Code
joblib.dump(model, 'model.pkl') 

fitness_band_model = joblib.load('model.pkl') 

y_pred = enc.inverse_transform(fitness_band_model.predict(X_test))

# Test accuracy
accuracy = model.evaluate(X_test, y_test)
print(X_test)
y_pred = enc.inverse_transform(model.predict(X_test))
print("Accuracy = ",accuracy[1])


y_test = enc.inverse_transform(y_test)
issame = y_pred == y_test
issame_sum = sum(issame)
acc = issame_sum/len(issame)

l4f.plot_cm(
    y_test,
    y_pred,
    enc.categories_[0]
)

