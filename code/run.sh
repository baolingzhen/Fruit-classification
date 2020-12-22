# Train the model
#python main.py train --dataset ../fruits-360/Training/ --save-model-dir ../cache --cuda

# Evaluate the trained model
python main.py eval --dataset ../fruits-360/Test/ --model ../cache/epochs10_best.pth --cuda
