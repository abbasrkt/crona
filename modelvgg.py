from tensorflow.python.keras.models import load_model
#test5

model = load_model('D:\cropImages - Covid\weights.00100-vl0.00.h5')
model.load_weights('D:\cropImages - Covid\weights.00100-vl0.00.h5')
