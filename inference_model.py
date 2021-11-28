import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine
from transformers import ViTFeatureExtractor, ViTModel

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
class LandmarkDataset(Dataset):

    def __init__(self, meta, root_dir, transform=None):
        self.meta = meta
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.meta['id'].iloc[idx]
        img_name = os.path.join(
            self.root_dir,
            image_id
        )
        image = Image.open(img_name).convert('RGB')
    
        if self.transform:
            image = self.transform(image=np.asarray(image))['image']
            image = Image.fromarray(image)

        image = feature_extractor(images=image, return_tensors="pt")

        return image

if __name__ == '__main__':
    #-----------------------Параметры-для-изменений--------------------------#
    dataset_path = "/workspace/test_data"
    answers_csv = "test_infer.csv"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # указать индекс гпу
    #------------------------------------------------------------------------#

    seed_everything()

    infer = pd.DataFrame()
    tmp_df = pd.DataFrame()
    tmp_df['id'] = os.listdir(dataset_path)
    infer = infer.append(tmp_df, ignore_index=True)

    df = pd.read_csv("./full_base_file.csv") 
    base = df.values.T

    model = 'google/vit-base-patch16-384'
    batch_size = 1

    feature_extractor = ViTFeatureExtractor.from_pretrained(model)

    infer_dataset = LandmarkDataset(meta=infer, root_dir=dataset_path, transform=None)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=True, num_workers=4)

    model = ViTModel.from_pretrained("./trained_metric_transformer/")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(infer_dataloader)):
            for i in batch:
                batch[i] = batch[i][:, 0].to(device)
            prediction = model(**batch).pooler_output
            prediction = prediction[0].cpu().detach().numpy()

            dist = []
            for emb in base:
                dist.append(cosine(emb, prediction))

            class_idx = np.argmin(np.array(dist)) 
            
            if dist[class_idx] < 0.42:
                infer.loc[idx, 'class'] = int(class_idx)
            else:
                infer.loc[idx, 'class'] = int(3)

    
    infer.to_csv(answers_csv, index=False)