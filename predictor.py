import torch
from torch import nn

from networks.modules import MetaLinear, MetaConv2d, MetaSequential, MetaModule
from config import TEACHER

class PredictorModel(MetaModule):
	def __init__(self, args):
		super(PredictorModel, self).__init__()
		self.args = args
		self.input_type = args.input_type
		self.h_dim = args.h_dim
		self.proj_inp_dim = 0

		if 'S' in self.input_type: # function embedding for student (Functional)
			self.hidden_channels = args.hidden_channels
			self.out_channels = args.out_channels
			self.proj_inp_dim += self.out_channels
			self.func_encoder = FuncEncoder(self.args, self.hidden_channels, self.out_channels)

		if 'T' in self.input_type: # teacher embedding 
			self.hidden_channels = args.hidden_channels
			self.out_channels = args.out_channels
			self.proj_inp_dim += self.out_channels
			self.teacher_encoder = FuncEncoder(self.args, self.hidden_channels, self.out_channels, teacher=True)
		
		if 'A' in self.input_type: # student architecture encoding (Architecture)
			self.a_inp_dim = args.a_inp_dim
			self.a_out_dim = args.a_out_dim
			self.proj_inp_dim += self.a_out_dim
			self.arch_encoder = ArchEncoder(self.a_inp_dim, self.a_out_dim)

		if self.proj_inp_dim == 0: raise ValueError(self.proj_inp_dim)

		self.proj_layers = MetaSequential(
			MetaLinear(self.proj_inp_dim, self.h_dim),
			nn.Tanh(),
			MetaLinear(self.h_dim, 1)
		)

	def forward(self, D=None, F=None, A=None, pred_inp=None, tcfunc_enc=None, n=None, params=None):
		input_vec = []
		if 'S' in self.input_type:
			input_vec.append(self.func_encoder(F, D, pred_inp, tcfunc_enc, n, params=self.get_subdict(params, 'func_encoder')))
		if 'A' in self.input_type:
			input_vec.append(self.arch_encoder(A, pred_inp, params=self.get_subdict(params, 'arch_encoder')))
		if 'T' in self.input_type:
			input_vec.append(self.teacher_encoder(F, D, pred_inp, tcfunc_enc, n, params=self.get_subdict(params, 'teacher_encoder')))
		
		input_vec = torch.cat(input_vec, dim=1)
		return self.proj_layers(input_vec, params=self.get_subdict(params, 'proj_layers'))


class FuncEncoder(MetaModule):
	def __init__(self, args, hidden_channels=512, out_channels=256, teacher=False):
		super(FuncEncoder, self).__init__()    
		self.teacher = teacher
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		self.tc_stage_cws = [int(TEACHER['cw_mul'] * w) for w in TEACHER['tc_stage_default_cw']]
		self.in_channels = self.tc_stage_cws[-1]
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels

		self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = MetaConv2d(
					self.in_channels,
					self.hidden_channels,
					kernel_size=3,
					stride=2,
					padding=1,
					groups=1,
					bias=False,
					dilation=1,
					)
		
		self.conv2 = MetaConv2d(
                    self.hidden_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=1,
                    bias=False,
                    dilation=1,
                    )
	
	def forward(self, F, D, pred_inp, tcfunc_enc, n, params=None):
	
		if self.teacher:
			inp = tcfunc_enc.to(self.device)
			out = self.conv1(inp, params=self.get_subdict(params, 'conv1'))
			out = self.relu(out)
			out = self.conv2(out, params=self.get_subdict(params, 'conv2'))
			out = self.relu(out)
			out = self.avgpool(out)
			out = out.view(-1)
			out = out.repeat(n, 1)
		else:
			x = torch.stack(pred_inp['func_enc'], dim=0)
			inp = x.squeeze()
			inp = inp.to(self.device) # (1, 1, 256, 8, 8) --> (1, 256, 8, 8) // (256, 8, 8)
			out = self.conv1(inp, params=self.get_subdict(params, 'conv1'))
			out = self.relu(out)
			out = self.conv2(out, params=self.get_subdict(params, 'conv2'))
			out = self.relu(out)
			out = self.avgpool(out)
			out = out.view(x.size(0), -1)
		return out


class ArchEncoder(MetaModule):
	def __init__(self, a_inp_dim, a_out_dim):
		super(ArchEncoder, self).__init__()    
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.a_inp_dim = a_inp_dim
		self.a_out_dim = a_out_dim
		self.fc = MetaLinear(self.a_inp_dim, self.a_out_dim)
		
	def forward(self, A, pred_inp, params=None):
		arch_encs = torch.stack(pred_inp['arch_enc'], dim=0).to(self.device)
		out = self.fc(arch_encs, params=self.get_subdict(params, 'fc'))
		return out

