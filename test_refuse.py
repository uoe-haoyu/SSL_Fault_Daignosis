import torch
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

class Test:
    def __init__(self, configs):
        self.t_time = 0.0
        self.t_sec = 0.0
        self.net = configs['netname']('_')
        self.test = configs['dataset']['test']
        self.val_dataloader = torch.utils.data.DataLoader(self.test,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pth = configs['pth_repo']
        self.sava_path = configs['test_path']
        self.print_staistaic_text = self.sava_path + 'print_staistaic_text.txt'


    def start(self):
        print("Loading .......   path:{}".format(self.pth))
        # Load model
        state = torch.load(self.pth)
        old_state_dict = state["state_dict"]
        new_state_dict = {}
        for key in old_state_dict:
            if key.startswith("fc.0"):
                new_key = "fc1.weight" if "weight" in key else "fc1.bias"
                new_state_dict[new_key] = old_state_dict[key]
            elif key.startswith("fc.2"):
                new_key = "fc2.weight" if "weight" in key else "fc2.bias"
                new_state_dict[new_key] = old_state_dict[key]

        self.net.load_state_dict(new_state_dict)
        self.net.to(self.device)

        accuracy = self.val_step(self.pth[-5],self.val_dataloader)

        return accuracy



    def val_step(self, epoch, dataset):
        print('-----------------start test--------------------')


        self.csv_onlylable = []

        self.net = self.net.eval()
        star_time = time.time()

        for i, data in enumerate(dataset):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)
            with torch.no_grad():
                prediction, feature = self.net(images)
                l1 = labels.to(self.device)


                temp_onlylable = torch.cat([l1.unsqueeze(1) , prediction, feature], dim=-1)
                self.csv_onlylable.append(temp_onlylable.cpu().detach().numpy().squeeze())


        duration = time.time() - star_time
        speed = 1 / (duration / len(dataset))
        print('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle = open(self.print_staistaic_text, mode='a')

        file_handle.write('-----------------start test--------------------')
        file_handle.write('\n')
        file_handle.write('avg_time:%.5f, sample_per_s:%.5f' %
                        (duration / len(dataset), speed))

        file_handle.write('\n')
        file_handle.close()

        self.net = self.net.train()
        accuracy = self.tocsv_onlylable(epoch)

        print('-----------------test over--------------------')

        return accuracy

    def tocsv_onlylable(self, epoch):
        np_data = np.array(self.csv_onlylable)
        label = np_data[:, :1]  # ground Truth
        pred_logits = np_data[:, 1:20]

        # Calculate softmax
        exp_logits = np.exp(pred_logits - np.max(pred_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Get predicted classes and maximum probabilities
        pred = np.argmax(pred_logits, axis=1)
        max_probs = np.max(probs, axis=1)
        features = np_data[:, 20:]
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        #########################################################################
        # Load class centers
        class_centers = np.load("class_centers.npy")

        # Stage 1: Confidence threshold
        confidence_threshold = 0.95

        # Stage 2: Distance threshold
        ood_threshold = 0.4

        # Create a mask to mark samples that need to go through the second-stage analysis
        low_confidence_mask = max_probs <= confidence_threshold

        # Only compute distances for low-confidence samples
        if np.any(low_confidence_mask):
            # Extract features of low-confidence samples
            low_conf_features = features[low_confidence_mask]

            # Compute the distance of these samples to each class center
            distances = cdist(low_conf_features, class_centers, metric='euclidean')

            # Get the minimum distance
            min_dists = np.min(distances, axis=1)

            # Create an out-of-distribution (OOD) mask
            ood_mask = min_dists > ood_threshold

            # Update predictions
            # For low-confidence samples:
            #   - If the distance is within the threshold, keep the original prediction
            #   - If the distance exceeds the threshold, mark it as unknown class (19)
            low_conf_preds = pred[low_confidence_mask]
            low_conf_preds[ood_mask] = 19  # Mark as unknown class
            pred[low_confidence_mask] = low_conf_preds

        # Metrics
        accuracy = accuracy_score(label, pred)

        # Print Results
        num_high_conf = np.sum(max_probs > confidence_threshold)
        num_low_conf = len(label) - num_high_conf
        num_rejected = np.sum(pred == 19)

        print(f"Two-Stage Rejection Analysis Results:")
        print(f"  High-confidence samples: {num_high_conf} ({num_high_conf / len(label) * 100:.2f}%)")
        print(f"  Low-confidence samples: {num_low_conf} ({num_low_conf / len(label) * 100:.2f}%)")
        print(f"  Final rejected samples: {num_rejected} ({num_rejected / len(label) * 100:.2f}%)")
        print(f"  Accuracy: {accuracy:.4f}")

        return accuracy
