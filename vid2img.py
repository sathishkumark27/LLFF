import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="./")
parser.add_argument("--images", type=str, default="./")
parser.add_argument("--rate", type=int, default=1)
args = parser.parse_args()

vidcap = cv2.VideoCapture(args.video)

if (vidcap.isOpened()== False):
    print("Error opening video stream or file")
count = 0
total = 0
while (vidcap.isOpened()):
  success,image = vidcap.read()
  if success is not True:
    break
  if count%args.rate == 0:
    im_path = args.images + ("frame%d.png" % count)
    cv2.imwrite(im_path, image)     # save frame as JPEG file
    # if cv2.waitKey(10) == 27:     # exit if Escape is hit
    #     break
    total+=1
    print("count = ", count)
  count += 1
print("Total = ", total)
vidcap.release()
