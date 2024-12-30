import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Configurar o diretório de dados
data_dir = 'caminho_para_sua_pasta_de_dados'

# Preparar os dados
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Construir o modelo de Deep Learning usando VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_generator, epochs=10, steps_per_epoch=len(train_generator))

# Função para extrair características das imagens
feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)

def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array)
    return features.flatten()

# Extração de características para todas as imagens
features_dict = {}
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(root, file)
            features_dict[img_path] = extract_features(img_path)

# Função para medir a similaridade e encontrar imagens similares
def find_similar_images(query_img_path, top_n=5):
    query_features = extract_features(query_img_path)
    similarities = {}
    for img_path, features in features_dict.items():
        similarity = cosine_similarity([query_features], [features])
        similarities[img_path] = similarity[0][0]

    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[:top_n]

# Exemplo de uso para encontrar as imagens mais similares
query_image = 'caminho_para_sua_imagem_consulta.jpg'
similar_images = find_similar_images(query_image)

print("Imagens mais similares:")
for img_path, similarity in similar_images:
    print(f"Imagem: {img_path}, Similaridade: {similarity}")
