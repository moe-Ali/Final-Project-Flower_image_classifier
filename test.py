import argparse
from operator import itemgetter
parser = argparse.ArgumentParser()
# Create command line arguments using add_argument() from ArguementParser method
parser.add_argument('--dir', type = str,action="store", default = 'flowers/', help = 'path to the folder of flower images')
arg=parser.parse_args()
arch=itemgetter('dir')(vars(arg))
dic={"train":1,"test":2}
y=dic["train"]
print(y)