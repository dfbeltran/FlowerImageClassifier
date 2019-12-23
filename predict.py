# Imports
import argparse
import predict_functions as pf


parser = argparse.ArgumentParser(description='Use neural network to classify flower category on an image.')

parser.add_argument('image_path', action='store',
                    default = './flowers/test/10/image_07090.jpg',
                    help='Enter path to image.')

parser.add_argument('--load_check_dir', action='store',
                    dest='check_dir', default = 'checkpoint.pth',
                    help='Enter path where the model checkpoint is located.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name', default = 'cat_to_name.json',
                    help='Enter name of cat file.')

parser.add_argument('--cat_to_path', action='store',
                    dest='cat_path', default = './',
                    help='Enter path where cat file is located.')

parser.add_argument('--top_k', action='store',
                    type=int, default = 5,
                    help='Number of top most likely clsses to view, default is 5')

parser.add_argument('--gpu', action="store_true", 
                    default=False,
                    help='Turn GPU mode on or off, default is off.')


results = parser.parse_args()
img_path = results.image_path
check_path = results.check_dir
cat_name = results.cat_name
cat_path = results.cat_path
top_k = results.top_k
device = results.gpu


model=pf.load_model(check_path, device)
cat_to_name=pf.openCat(cat_path, cat_name)
prediction= pf.predict(img_path, model, top_k, cat_to_name, device)

for k, (i,v) in enumerate(prediction.items()):
    if k == 0:
        print(f"This flower is most likely to be a: '{i}' with a probability of {round(v*100,4)}% ")
    elif k == 1:
        print(f"also it could be a: '{i}' with a probability of {round(v*100,4)}% ")
    else:
        print(f"or it could be a: '{i}' with a probability of {round(v*100,4)}% ")
