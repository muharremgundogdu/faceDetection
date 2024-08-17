import tensorflow as tf
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# tensorflow veri akisina (data pipeline) goruntuleri yukle

images = tf.data.Dataset.list_files('C:/Users/Muharrem/OpenCVProjects/ip-projects/facedetection/data/images/*.jpg' , shuffle = False)
# shuffle kapatarak rastgele yuklenmesini engelledik

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)
images.as_numpy_iterator().next()  # veri kumesinden sonraki eleman alinir ve numpy dizisine donusturup dondurur

image_generator = images.batch(4).as_numpy_iterator()  # batch ile 4 erli gruplandirdik
plot_images = image_generator.next()   # sonrakine geciyor

fig , ax = plt.subplots(ncols = 4 , figsize = (20,20))
for idx , image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()
# 4 erli sekilde gorsellestirdik

# --------------------------------

# MANUALLY SPLT DATA INTO TRAIN TEST AND VAL -> rastgele resimler secip train test ve val e atacagiz
# 90*.7 -> imagesdeki 63 tanesini train/images e kesip yapistirtik
# 90*.15 -> 14 tanesini teste and 13 tanesini val e kesip attik


# --------------------------------

# dongu icinde labellari da atacagiz
for folder in ['train','test','val']:  # bu dongu train test val klasorlerini dolasir
    for file in os.listdir(os.path.join('data', folder, 'images')):  # bu dongu her bir images klasorunde bulunan dosyalari tasir
        
        filename = file.split('.')[0]+'.json'  # goruntu dosyasinda jpg yi siler ve uzantiyi json yapar
        existing_filepath = os.path.join('data','labels', filename) # etiket dosyasinin tam yolunu olusturur
        if os.path.exists(existing_filepath): # etiket dosyasinin var olup olmadigini kontrol eder
            new_filepath = os.path.join('data',folder,'labels',filename)  # etiket dosyasinin yeni konumu
            os.replace(existing_filepath, new_filepath)   # mevcut etiket dosyasini yeni konuma tasir. 
            # Eger hedef konumda ayni isimde bir dosya varsa, bu islev mevcut dosyayi degistirir.


# --------------------------------
# Apply Image Augmentation on Images and Labels using Albumentations

import albumentations as alb

augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))
# veri arttirma islemi yaptik -> rastgele kirpma, yana cevirme ,rastgele parlaklik.. yaparak veri cesitliligini arttirdik
# augmentation -> veri arttirma


# ---------------------------------------------------------------------------------------------------------
# bir resim icin deneme yapalim
# Load a Test Image and Annotation with OpenCV and JSON  
img = cv2.imread(os.path.join('data','train', 'images','8b1c1c96-e125-11ee-a13d-2cf05dff42a9.jpg'))

with open(os.path.join('data', 'train', 'labels', '8b1c1c96-e125-11ee-a13d-2cf05dff42a9.json'), 'r') as f:
    label = json.load(f)
label['shapes'][0]['points']

# Extract Coordinates and Rescale to Match Image Resolution
# Etiketlerden nesnenin konumunu belirlemek icin kose noktalarinin koordinatlari alinir ve cozunurluge gore olcekle 
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords = list(np.divide(coords, [640,480,640,480]))


# Apply Augmentations and View Results. veri arttirma islemi uygula
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

cv2.rectangle(augmented['image'], 
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)), 
                    (255,0,0), 2)

plt.imshow(augmented['image'])

# ---------------------------------------------------------------------------------------------------------
# tum resimler icin yapalim

# Build and Run Augmentation Pipeline
# Run Augmentation Pipeline

for partition in ['train','test','val']:  # train test val klasorlerini dolasiyor
    for image in os.listdir(os.path.join('data',partition,'images')):   # bu klasorlerdeki tum resimleri al
        img = cv2.imread(os.path.join('data',partition,'images',image))
        
        coords = [0 , 0 , 0.00001 , 0.00001]   # etiket dosyasi yoksa default olarak bu koordinatlar kullanilacak
        label_path = os.path.join('data',partition,'labels',f'{image.split(".")[0]}.json')  # etiket dosyasinin yolu
        if os.path.exists(label_path):   # etiket dosyasi var mi diye kontrol ediliyor
            with open(label_path , 'r') as f:
                label = json.load(f)
                
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords , [640,480,640,480]))
        
        # resimler ve etiketler yuklendi
        
        try:
            for x in range(60):  # her bir resim icin 60 arttirilmis resim olusturulacak
                augmented = augmentor(image = img , bboxes = [coords] , class_labels = ['face'])  # arttirma islemi
                cv2.imwrite(os.path.join('aug_data',partition,'images',f'{images.split(".")[0]}.{x}.jpg'),augmented['image'])
                # once manuel sekilde aug_data klasoru, icine train test val, bunlarin da icine images,labels klasorleri
                # olusturduk sonra aug_data klasorune kaydettik
                
                # bos bir etiket sozlugu olustur ve bu sozluge goruntunun adini ekle
                annotation = {}
                annotation['image'] = image
                
                # eger etiket dosyasi varsa arttirilmis etiketleri al ve sozluge ekle. 
                # etiket yoksa varsayilan koordinatlari ve sinifi 0 yap
                if os.path.exist(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0
                    
                # arttirilmis etiketleri json dosyasina kaydet    
                with open(os.path.join('aug_data',partition,'labels',f'{image.split(".")[0]}.{x}.json'),'w') as f:
                    json.dump(annotation , f)
                    # json.dump -> sozluk veya liste gibi bir veri yapisini alir ve bunu belirtilen dosyaya
                    # json formaitinda yazar
                    
        except Exception as e:
            print(e)


# Load Augmented Images to Tensorflow Dataset  ->  Arttirilmis kumeyi tensorflow datasetine yukle
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg' , shuffle = False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x , (120,120)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg' , shuffle = False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x , (120,120)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg' , shuffle = False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x , (120,120)))
val_images = val_images.map(lambda x: x/255)

train_images.as_numpy_iterator().next()




# Labellari hazirla
# label yukleme fonksiyonu;
def load_labels(label_path):
    with open(label_path.numpy() , 'r' , encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']] , label['bbox']

# labellari tensorflow datasetine yukle
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

train_labels.as_numpy_iterator().next()



# Combine Label and Image Samples -> etiketleri ve labellari birlestir
# uzunluklari kontrol edelim
len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)

# Create Final Datasets (Images/Labels)
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)  
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)
# tf.data.Dataset.zip fonksiyonu goruntuler ve etiketler arasinda bir eslestirme yapar. 

# shuffle fonksiyonu veri setini karistirarak her bir ogenin siralanmis olmasini engeller. 
# bu modelin egitim sirasinda ogrenme surecini iyilestirebilir ve asiri uydurma riskini azaltabilir.
# 5000 karistirma islemi icin kullanilan oge sayisini belirtir. Yani bu durumda 5000 oge rastgele secilecektir.

# batch fonksiyonu veri setini kucuk toplu islemlere boler. 8 her bir toplu islemde bulunmasi gereken ornek sayisini belirtir.

# prefetch() fonksiyonu, model egitimi sirasinda veri setinin bir sonraki toplu islemi yuklenirken bellegi 
# doldurmayi saglar. 4, bir sonraki toplu islem icin yuklenmesi gereken ornek sayisini belirtir.

train.as_numpy_iterator().next()[1]


# View Images and Annotations
data_samples = train.as_numpy_iterator()

res = data_samples.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]
    
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(sample_image, 
                  tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                        (255,0,0), 2)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    ax[idx].imshow(sample_image)


# dataset artik hazir.



# ---------------------------------------------------------------------------------------------------------
# DERIN OGRENME MODELINI OLUSTUR

# Import Layers and Base Network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16


# Download VGG16
vgg = VGG16(include_top=False)

vgg.summary()  # summary ile bilgisine bakiyoruz


# Build instance of network
def build_model():
    input_layer = Input(shape=(120,120,3))
    
    vgg = VGG16(include_top=False)(input_layer)
    
    # classification model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048 , activation = 'relu')(f1)
    class2 = Dense(1 , activation = 'sigmoid')(class1)
    
    # bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048 , activation = 'relu')(f2)
    regress2 = Dense(4 , activation = 'sigmoid')(regress1)
    
    facetracker = Model(inputs=input_layer , outputs=[class2 , regress2])
    
    return facetracker
    
    

# test out neural network
facetracker = build_model()

facetracker.summary()

X, y = train.as_numpy_iterator().next()

X.shape

classes, coords = facetracker.predict(X)

classes, coords





# Define Losses and Optimizers
# Define Optimizer and LR
batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch

opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

# Create Localization Loss and Classification Loss
def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# Test out Loss Metrics
localization_loss(y[1], coords)

classloss(y[0], classes)

regressloss(y[1], coords)




# Train Neural Network
# Create Custom Model Class
class FaceTracker(Model):   # model sinifindan miras aliyor
    def __init__(self, eyetracker,  **kwargs):  # sinifin insa yontemidir. eyetracker argumani facetracker sinifinin bir
                                                # ozelligi olarak atanacaktir
        super().__init__(**kwargs)  # bu ust sinif olan model sinifinin insa yontemini cagirir ve tum ozellikleri alir
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):  # modelin derleme yontemidir
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt  # optimizasyon algoritmasini belirler
    
    def train_step(self, batch, **kwargs): # Bu, modelin egitim adimi yontemidir. Bu yontem, bir yigin veri alir, 
                                            # gradyanlari hesaplar ve modelin agirliklarini gunceller.
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): # modelin test adimi yontemidir
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs):  # modelin cagri yontemidir
        return self.model(X, **kwargs)


model = FaceTracker(facetracker)

model.compile(opt , classloss , regressloss)


# Train
logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
# burayi calistirinca log klasoru olustu




# Plot Performance
hist.history

fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()






# Make Predictions
# Make Predictions on Test Set
test_data = test.as_numpy_iterator()

test_sample = test_data.next()

yhat = facetracker.predict(test_sample[0]) # 0.index yani x i aldik

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]
    sample_image = cv2.cvtColor(test_sample[0][idx], cv2.COLOR_RGB2BGR) # uyumsuzluk oluyor diye cevirdim

    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image, 
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)
    
    ax[idx].imshow(sample_image)




# Save the Model
from tensorflow.keras.models import load_model

facetracker.save('facetracker.h5')

facetracker = load_model('facetracker.h5')  # artik bunu yazip farkli yerde kullanabiliriz



