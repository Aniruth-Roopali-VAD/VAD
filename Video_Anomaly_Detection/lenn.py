import os 
path_ = os.path.join(os.getcwd() , 'Videos' , 'Train')

l = []
import cv2


os.chdir(path_)
for i in os.listdir(path_):
    cap = cv2.VideoCapture(i)
    
    # Get the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames in the video: {}".format(num_frames))
    
    l.append(num_frames)

    # Release the video capture object
    cap.release()

print(l)
print(max(l))
