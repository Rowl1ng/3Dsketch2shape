from models.encoder.pointnet2_cls_msg import get_model
import torch.nn as nn
import torch

class SketchEncoder(nn.Module):
	def __init__(self, config):
		super(SketchEncoder, self).__init__()
		self.model = get_model()
	def forward(self, pc, is_training=False):
		if is_training:	
			self.model.train()
			feat = self.model(pc)
		else:
			with torch.no_grad():
				self.model.eval()
				feat = self.model(pc)
		return feat