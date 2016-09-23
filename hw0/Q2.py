from PIL import Image
import sys
im = Image.open(sys.argv[1])
im.rotate(180).save('ans2.png','PNG')