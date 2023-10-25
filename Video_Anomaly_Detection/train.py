# Import necessary libraries and modules
import torch
import torchvision
import cv2 

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from mmaction.models.backbones.swin_transformer import SwinTransformer3D

#from dam_vp import MetaDiversityModel  
from damvp.task_adapting.adapter import Adapter
from datasets.shanghai_tech import SHANGHAITECH  
from datasets.transforms import RemoveBackground
from datasets.transforms import ToFloatTensor3D
from datasets.shanghai_tech import VideoAnomalyDetectionDataset
from torch.utils.data import Dataset
import os 
from glob import glob 

class PromptEmbeddingHead(nn.Module):
    def __init__(self, input_channels, prompt_dim):
        super(PromptEmbeddingHead, self).__init()
        self.prompt_dim = prompt_dim
        self.fc = nn.Linear(input_channels, prompt_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x



#path_ = r"Final/video_swin_transformer/videos/Train"

#print(os.listdir(path_))

#print(os.getcwd())
path_ = os.path.join(os.getcwd() , 'videos' , 'Train')
#print(os.listdir(path_))

from torch.utils.data import DataLoader

class AVIDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.avi_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
        self.transform = transform

    def __len__(self):
        return len(self.avi_files)

    def __getitem__(self, idx):
        avi_file_path = os.path.join(folder_path, self.avi_files[idx])
        avi = cv2.VideoCapture(avi_file_path)
        frames = []

        while True:
            ret, frame = avi.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        avi.release()

        # Return a list of frames for this video
        return frames



from torch.utils.data import Dataset


class SHANGHAITECH(VideoAnomalyDetectionDataset):
    """
    Models ShanghaiTech dataset for video anomaly detection.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which ShanghaiTech is stored.
        """
        super(SHANGHAITECH, self).__init__()

        self.path = path

        # Test directory
        self.test_dir = os.path.join(path, 'testing')

        # Transform
        self.transform = transforms.Compose([RemoveBackground(threshold=128), ToFloatTensor3D(normalize=True)])
        
        self.num_frames_sample = 400 #self.get_num_of_frames()

        # Load all test ids
        self.test_ids = self.load_test_ids()

        # Other utilities
        self.cur_len = 0
        self.cur_video_id = None
        self.cur_video_frames = None
        self.cur_video_gt = None
        self.cur_background = None
   
    def get_num_of_frames(self):
    	cap = cv2.VideoCapture(self.path)
    	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    	cap.release()
    	return total_frames



    def load_test_ids(self):
        # type: () -> List[str]
        """
        Loads the set of all test video ids.

        :return: The list of test ids.
        """
        return sorted([basename(d) for d in glob(os.path.join(self.test_dir, 'frames', '**')) if isdir(d)])

    def load_test_sequence_frames(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads a test video in memory.

        :param video_id: the id of the test video to be loaded
        :return: the video in a np.ndarray, with shape (n_frames, h, w, c).
        """
        c, t, h, w = self.shape

        sequence_dir = os.path.join(self.test_dir,  'frames', video_id)
        img_list = sorted(glob(os.path.join(sequence_dir, '*.jpg')))
        test_clip = []
        for img_path in img_list:
            img = io.imread(img_path)
            img = resize(img, output_shape=(h, w), preserve_range=True)
            img = np.uint8(img)
            test_clip.append(img)
        test_clip = np.stack(test_clip)
        return test_clip

    def load_test_sequence_gt(self, video_id):
        # type: (str) -> np.ndarray
        """
        Loads the groundtruth of a test video in memory.

        :param video_id: the id of the test video for which the groundtruth has to be loaded.
        :return: the groundtruth of the video in a np.ndarray, with shape (n_frames,).
        """
        clip_gt = np.load(os.path.join(self.test_dir,  'test_frame_mask', f'{video_id}.npy'))
        return clip_gt

    def test(self, video_id):
        # type: (str) -> None
        """
        Sets the dataset in test mode.

        :param video_id: the id of the video to test.
        """
        c, t, h, w = self.shape

        self.cur_video_id = video_id
        self.cur_video_frames = self.load_test_sequence_frames(video_id)
        self.cur_video_gt = self.load_test_sequence_gt(video_id)
        self.cur_background = self.create_background(self.cur_video_frames)

        self.cur_len = len(self.cur_video_frames) - t + 1

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        """
        Returns the shape of examples being fed to the model.
        """
        return 16 , 3, self.num_frames_sample , 256, 512

    @property
    def test_videos(self):
        # type: () -> List[str]
        """
        Returns all available test videos.
        """
        return self.test_ids

    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.cur_len

    @staticmethod
    def create_background(video_frames):
        # type: (np.ndarray) -> np.ndarray
        """
        Create the background of a video via MOGs.

        :param video_frames: list of ordered frames (i.e., a video).
        :return: the estimated background of the video.
        """
        mog = cv2.createBackgroundSubtractorMOG2()
        for frame in video_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mog.apply(img)

        # Get background
        background = mog.getBackgroundImage()

        return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    def __getitem__(self, i):
    	print(self.shape)
    	b , c, t, h, w = self.shape
    	if self.cur_video_frames is None:
    	    placeholder_sample = torch.zeros((b , c, t, h, w), dtype=torch.float32)
    	    return placeholder_sample, placeholder_sample, self.cur_background
    	
    	clip = self.cur_video_frames[i:i+t]
    	
    	sample = clip, clip, self.cur_background
    	
    	if self.transform:
    		sample = self.transform(sample)
    	
    	return sample

    @property
    def collate_fn(self):
        """
        Returns a function that decides how to merge a list of examples in a batch.
        """
        return default_collate

    def _repr_(self):
        return f'ShanghaiTech (video id = {self.cur_video_id})'

# Define hyperparameters
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# Load and preprocess the Shanghai dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
shanghai_dataset = SHANGHAITECH(path_ )



print(len(shanghai_dataset))

#data_loader = torch.utils.data.DataLoader(shanghai_dataset, batch_size=batch_size, shuffle=True)

data_loader2 = torch.utils.data.Subset(shanghai_dataset  , indices = [i for i in range(0 , 2)] )

print(len(data_loader2))
video_transformer = SwinTransformer3D()

class AnomalyDetectionHead(nn.Module):
    def __init__(self, input_dim = 512 , hidden_dim=128):
        super(AnomalyDetectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
combined_model = nn.Sequential(
    video_transformer,
    )

#meta_diversity_model = MetaDiversityModel()

anomaly_detection_head = AnomalyDetectionHead(video_transformer.num_features)

criterion = nn.BCELoss()
optimizer = optim.Adam(anomaly_detection_head.parameters(), lr=learning_rate)

#print(data_loader)
args = {}
adapter = Adapter(args, combined_model)


# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(data_loader2):
        video_frames = batch[batch_idx]
        visual_features = combined_model(video_frames)
        #adapted_visual_features = adapter.get_prompted_image(visual_features )
        labels=0
        loss = criterion(anomaly_scores, labels.float() )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(data_loader2)}], Loss: {loss.item()}')

# After training, you can use the model for anomaly detection on new videos
