import torch
import speechbrain as sb
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from parse_data import parse_to_json
from speechbrain.dataio.dataset import DynamicItemDataset

# HuggingFace model hub
model_hub_w2v2 = "facebook/wav2vec2-base-960h"

model_w2v2 = HuggingFaceWav2Vec2(model_hub_w2v2, save_path="./pretrained/")

print(model_w2v2)

# How to install the datasets from LibriSpeech
# Go to this link and copy the link for the data to install: https://www.openslr.org/12
# Then run the following command:
# wget ${LINK_NAME}
# tar -xvzf ${TAR_FILE_NAME}

parse_to_json("LibriSpeech/test-clean")
dataset = DynamicItemDataset.from_json("data.json")

class SimpleBrain(sb.Brain):
  def compute_forward(self, batch, stage):
    return self.modules.model(batch["input"])


  def compute_objectives(self, predictions, batch, stage):
    return torch.nn.functional.l1_loss(predictions, batch["target"])

model = torch.nn.Linear(in_features=10, out_features=10)
brain = SimpleBrain({"model": model}, opt_class=lambda x: torch.optim.SGD(x, 0.1))
data = [{"input": torch.rand(10, 10), "target": torch.rand(10, 10)}]
brain.fit(range(10), data)