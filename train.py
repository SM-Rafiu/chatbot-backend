import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
import numpy as np
import pickle
import random
import json
  
lem = WordNetLemmatizer()
collected_tags = sorted(set([]))
groupings = []
list_of_words = []

intents = json.loads(open('intents.json').read())
for intent in intents['intents']:
    for pattern in intent['patterns']:
        list_of_words.extend(nltk.word_tokenize(pattern))
        groupings.append((nltk.word_tokenize(pattern), intent["tag"]))
        if intent["tag"] not in collected_tags:
            collected_tags.append(intent['tag'])

list_of_bases = sorted(set({lem.lemmatize(word) for word in list_of_words if word not in ['!', ',', ':', '?', '.']}))

pickle.dump(list_of_bases, open('bases.pickle', 'wb'))
pickle.dump(collected_tags, open('tags.pickle', 'wb'))

out_template = [0] * len(collected_tags)
xy_data = []
  
for group in groupings:
    bag = []
    base_patterns = [lem.lemmatize(word.lower()) for word in group[0]]
    for base in list_of_bases:
        bag.append(1 if base in base_patterns else 0)
    current_out = list(out_template)
    current_out[collected_tags.index(group[1])] = 1 
    xy_data.append([bag, current_out])

random.shuffle(xy_data)

# Convert xy_data to a NumPy array
xy_data = np.array(xy_data, dtype=object)

# Splitting the training list
x_train = np.array([item[0] for item in xy_data])
y_train = np.array([item[1] for item in xy_data])

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(len(x_train[0]),)))
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Uncomment the optimizer you want to use
# sgd = SGD(learning_rate=0.01)
adam = Adam(learning_rate=0.01)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

x = np.array(x_train)
y = np.array(y_train)
model.save('botModel.h5', model.fit(x, y, batch_size=10, epochs=400, verbose=2))


print("Training complete!")
model.summary()