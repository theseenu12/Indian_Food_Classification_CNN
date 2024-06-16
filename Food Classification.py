import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,multilabel_confusion_matrix
from pathlib import Path
import tensorflow_hub as hub
from PIL import Image
import os
import datetime
import seaborn as sns


print("Gpu Available" if tf.config.list_physical_devices('GPU') else "Not available")

# unique_foods = os.listdir('Food')

# print(unique_foods)

df = pd.read_csv('food_labels_shuffle.csv')

print(df)

df.rename(columns={'Features':'features','Labels':'labels'},inplace=True)

features = [x for x in df['features']]

labels = df['labels'].to_numpy()


print(features)
print(labels)

## get the unique labels of the food
print(np.sort(df['labels'].unique()))


## check the median value for each food variety
print(df['labels'].value_counts().median())

labels = df['labels'].to_numpy()

# features = df['features'].to_numpy()

# turn every label into boolean array

boolean_labels = []

unique_foods = np.unique(labels)

print(unique_foods)

print(labels[1] == unique_foods)
print(features[1])

## turn labels into booleans

for label in labels:
    boolean_labels.append(label == unique_foods)

print(boolean_labels[10])
print(features[10])

x = features
y = boolean_labels

boolean_labels  = np.array(boolean_labels,dtype=bool)


# print(boolean_labels[25])

## 6293 total images
x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42)

## test a random image and its label
plt.imshow(plt.imread(x_train[35]))
plt.axis('off')
plt.title(unique_foods[np.argmax(y_train[35])])

plt.show()

plt.imshow(plt.imread(x_valid[35]))
plt.axis('off')
plt.title(unique_foods[np.argmax(y_valid[35])])

plt.show()

## lets build a function to turn an image into tensor
IMG_SIZE = 224

def process_image(image_path,img_size=IMG_SIZE):
    ## takes an image file path and turns the image into tensor
    
    
    ## read in an image file and it return the tensor and dtype as string
    image = tf.io.read_file(image_path)
    
    ## one more method
    
    # image1 = tf.constant(plt.imread(image_path))

    ## turn the jpeg into numerical(int) tensor with 3 color channels
    image = tf.image.decode_jpeg(image,3)
    
    ## convert the color channel values from 0-255 to 0-1 values (This is called Normalization)

    image = tf.image.convert_image_dtype(image,tf.float32)
    
    ## resize the image to our desired shape
    image = tf.image.resize(image,size=[224,224])
    
    return image

## create a simple function to return a tuple

def get_image_label(image_path,label):
    
    """ takes an image file path name and label and returns the tuple of image and label"""

    image = process_image(image_path)
    
    return image,label

## create a function to turn data into batches

batch_size = 32

processed_image = process_image(x_train[45])

print(processed_image)
# plt.imshow(processed_image)

# plt.title(y_train[45])

# plt.show()

def create_data_batches(x,y=None,batch_size=batch_size,valid_data=False,test_data=False):
    
    '''
    create batches of data out of image x and label y pairs
    it shuffles the data if its training data,But  doesnt shuffle data if its validation data
    also accepts test data as input(thats why we have y=None in case)
    '''
    
    ## if the data is a test dataset we probably dont have labels
    if test_data:
        print("Creating Test data batches..")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))  ## only filepaths no labels
        data_batch = data.map(process_image).batch(batch_size=batch_size)
        
        return data_batch
    
    ## if the data is valid data we dont need to shuffle it

    elif valid_data:
        print("Creating valid Data Batches ......")
        
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        
        data_batch = data.map(get_image_label).batch(batch_size=batch_size)
        
        return data_batch
        
    else:
        print("Creating Training Data batches ...... ")
        ## turn filepaths and labels into tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        
        ## shuffling pathnames and labels before mapping image processor function is faster than shuffling images

        data = data.shuffle(buffer_size=len(x))
        
        ## create image and label tuples (this also turns the image path into image tensor with range between 0-1 instaed of 0-255)
        data_batch = data.map(get_image_label).batch(batch_size)

        return data_batch



train_data = create_data_batches(x_train,y_train)

val_data = create_data_batches(x_valid,y_valid,valid_data=True)


## setup input shape to the model

INPUT_SHAPE = [None,IMG_SIZE,IMG_SIZE,3]  # batch,height,width,colorchannels

## set up the output shape

OUTPUT_SHAPE = 20
## setup model URL From tensorflow_hub

MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4'
# https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4

MODEL_URL2 = "https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/130-224-classification/versions/2"

# create a function for the model

def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model_url=MODEL_URL2):
    print("Building model with :",model_url)
    
    ## setup the model layers

    model = tf.keras.Sequential([hub.KerasLayer(model_url),
                                 
                                 tf.keras.layers.Dense(units=output_shape,activation='softmax')## layer2  Output Layer
                                 ])
    
    
    ## compile the model

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    
    ## build the model
    model.build(input_shape)

    return model

def create_tensorboard_callback():
    ## create a log directory for storing tensorboard logs
    logdir = os.path.join('D:/Visual Studio Projects/Machine Learning Project Daniel Boruke/logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    return tf.keras.callbacks.TensorBoard(logdir)

early_stoppping  = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=4)

NUM_EPOCHS = 100

print(len(train_data))

print(len(val_data))
    
def train_model():
    '''Trains a given model and returns a trained version'''
    
    ## create model
    model = create_model()
    
    ## create a Tensorboard session everytime we train a model
    tensorboard = create_tensorboard_callback()
    
    ## fit the model to the data passing it the callbacks we created    
    model.fit(x=train_data,epochs=NUM_EPOCHS,validation_data=val_data,validation_freq=1,callbacks=[early_stoppping])
    
    ## return the fitted model

    return model

    
# model = train_model()

# print(model.summary())

## lets try and create a full model with full train database instead of x_train

# load_model = tf.keras.models.load_model('Food_Model')

# predictions = load_model.predict(val_data,verbose='auto')

## create a full model with full train dataset

# full_model = create_model()

# full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=3)

# full_data = create_data_batches(x,y)

# full_model.fit(full_data,epochs=NUM_EPOCHS,batch_size=batch_size,callbacks=[full_model_early_stopping])

# predictions = full_model.predict(val_data,verbose='auto')

# full_model.save('Food_Model_Full/')

full_load_model = tf.keras.models.load_model('Food_Model_Full')

predictions = full_load_model.predict(val_data,verbose='auto')

valid_images = []
valid_labels = []


for data,labels in val_data.unbatch().as_numpy_iterator():
    valid_images.append(data)
    valid_labels.append(labels)
    
random = np.random.randint(1,200,20)

# model.save('Food_Model/')

## create a function to plot predictions
def plot_predictions(predictions,actuals):
    for i,label in enumerate(random):
        plt.subplot(4,5,i+1)
        plt.imshow(valid_images[label])
        plt.title(f"Pred : {unique_foods[np.argmax(predictions[label])]} | act:{unique_foods[np.argmax(actuals[label])]} | prob:{np.max(predictions[label])*100:.2f}",color="green" if unique_foods[np.argmax(predictions[label])] == unique_foods[np.argmax(actuals[label])] else 'red',fontdict=dict(fontsize='small'))
        plt.axis('off')
        plt.tight_layout()
        
    plt.show()


plot_predictions(predictions,valid_labels)


test_food = ["Food_Test/" + x for x in os.listdir('Food_Test/')]

test_data_set = create_data_batches(test_food,test_data=True)

test_preds = full_load_model.predict(test_data_set,verbose='auto')

for i,label in enumerate(test_data_set.unbatch().as_numpy_iterator()):
    plt.subplot(2,7,i+1)
    plt.imshow(label)
    plt.title(f"Pred : {unique_foods[np.argmax(test_preds[i])]} | Prob : {np.max(test_preds[i])*100 :.1f}",color='blue',fontdict=dict(fontsize='small'))
    plt.axis('off')
   

plt.show()
