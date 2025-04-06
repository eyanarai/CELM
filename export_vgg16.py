import os
import tensorflow as tf
from vgg_model import build_vgg16

def export_model(model, model_name="model", export_dir="exports"):
    os.makedirs(export_dir, exist_ok=True)
    h5_path = os.path.join(export_dir, f"{model_name}.h5")
    model.save(h5_path)
    print(f"✅ Saved {model_name}.h5")

    sm_path = os.path.join(export_dir, f"{model_name}_savedmodel")
    model.save(sm_path, save_format='tf')
    print(f"✅ Saved {model_name}_savedmodel")

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
    tflite_model = converter.convert()
    tflite_path = os.path.join(export_dir, f"{model_name}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Exported {model_name}.tflite")

def run_export():
    model = build_vgg16(input_shape=(224, 224, 3))
    model.load_weights("results/VGG16/VGG16_best_weights.h5")
    export_model(model, model_name="VGG16", export_dir="exports/vgg16")

if __name__ == "__main__":
    run_export()
