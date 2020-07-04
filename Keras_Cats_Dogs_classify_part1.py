# 고양이와 강아지를 분류해보자
# wget을 이용해 고양이와 강아지의 zip파일을 불러오자
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /content/cats_and_dogs_filtered.zip

# 불러온 zip파일을 압축을 풀어주자
import os
import zipfile
# 모델 구성을 위해 케라스의 레이어와 모델을 불러옴
from keras import layers
from keras import Model

local_zip = '/content/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

base_dir = '/content/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 훈련할 고양이와 강아지 사진 디렉토리를 각각 설정해 줌
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# 검증에 사용할 사진들을 각각 디렉토리를 설정해 줌
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


# os의 하위 디렉토리를 확인하는 것을 통해 이름들을 넣어준다
train_cat_fname = os.listdir(train_cats_dir)
train_dog_fname = os.listdir(train_dogs_dir)

# 각각 1000개씩 사진 데이터 이름이 있는 것을 확인할 수 있음
print(len(train_cat_fname))
print(len(train_dog_fname))

 가로세로 150픽셀이며, RGB 3가지 속성을 위해 3개 속성을 냅둠
img_input = layers.Input(shape=(150, 150, 3))

# 내가 참고한 강의 자료에 따르면 Sequential 모델을 두고,
# 거기에 add 해주는 방식이었는데 얘는 x 변수에 차곡차곡 쌓는 방식
# 2D 형태로 만들어 주어 MaxPooling2D로 크기를 축소해 효율성 챙김
x = layers.Conv2D(16, (3, 3), activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))

x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(2)(x)



# Flattern을 통해 차원을 1차원으로 만듦
x = layers.Flatten()(x)

# 오버피팅이 일어나기 전까지 계속 돌림(?)
x = layers.Dense(512, activation='relu')(x)

# 마지막 결과물은 하나로 만들어 sigmoid로 확인하자
output = layers.Dense(1, activation='sigmoid')(x)

# 강의와 다르게 여기서는 다 만들고 모델에 집어 넣음
model = Model(img_input, output)

# 지금까지 만든 모델 확인
model.summary()