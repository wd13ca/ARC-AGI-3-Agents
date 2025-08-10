


#########################################################################################################################################################################

AGENT_MAX_ACTIONS = 50000
AGENT_LOOP_SLEEP = 0.1
AGENT_E = 0.5
RWEIGHT_MIN = 0.1
RWEIGHT_RANK_DISCOUNT = 0.5
RWEIGHT_NO_DISCOUNT = 0.5
MODEL_LR = 1e-4
MODEL_BATCH_SIZE = 32
MODEL_NUM_EPOCHS = 10
MODEL_SCORE_MAG = 1
MODEL_MAX_TRAIN_TIME = 15



#########################################################################################################################################################################

from .agent import Agent
from .structs import FrameData, GameAction, GameState

from collections import deque
import random
import sys
import time

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import models as torchvision_models

sys.setrecursionlimit(2000)



#########################################################################################################################################################################

class BlindSquirrel(Agent):

	MAX_ACTIONS: int = AGENT_MAX_ACTIONS

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def process_latest_frame(self,latest_frame):
		time.sleep(AGENT_LOOP_SLEEP)
		if latest_frame.state is GameState.NOT_PLAYED:
			assert latest_frame.frame==[]
			self.prev_state = 'NOT_PLAYED'
			return
		if latest_frame.state is GameState.GAME_OVER:
			return
		assert latest_frame.state in (GameState.NOT_FINISHED,GameState.WIN)
		if self.prev_state=='NOT_PLAYED':
			self.game_counter = 0
			self.level_counter = 0
			self.game_id = latest_frame.game_id
			self.graph = StateGraph()
			self.current_state = self.graph.get_state(latest_frame)
			self.graph.add_init_state(self.current_state)
			return
		self.current_state = self.graph.get_state(latest_frame)
		self.game_counter+=1
		if self.current_state.score>self.prev_state.score:
			self.level_counter=0
		else:
			self.level_counter+=1
		self.graph.update(self.prev_state,self.prev_action,self.current_state)

	def is_done(self,frames,latest_frame):
		self.process_latest_frame(latest_frame)
		if latest_frame.state is GameState.WIN:
			return True
		return False

	def choose_action(self,frames,latest_frame):
		if latest_frame.state in (GameState.NOT_PLAYED,GameState.GAME_OVER):
			return GameAction.RESET
		if AGENT_E<random.random() and latest_frame.score>0 and len(self.current_state.future_states)>0:
			action = self.get_model_action()
		else:
			action = self.get_rweights_action()
		action_obj = self.current_state.get_action_obj(action)
		self.prev_state = self.current_state
		self.prev_action = action
		return action_obj

	def get_model_action(self):
		game_id = self.current_state.game_id
		score = self.current_state.score
		model = self.graph.action_model
		device = next(model.parameters()).device
		model.eval()
		model_values = {}
		with torch.no_grad():
			for action,rweight in self.current_state.action_rweights.items():
				if rweight==0:
					continue
				x_s = torch.as_tensor(self.current_state.frame,dtype=torch.long).unsqueeze(0).to(device)
				x_a = self.current_state.get_action_tensor(action).unsqueeze(0).to(device)
				value = model(x_s,x_a).item()
				if rweight is None:
					value = value*self._rweight_calc(game_id,score,action)
				model_values[action] = value
		if not model_values:
			print('Warning: No Actions',self.game_id)
			return random.randint(0,self.current_state.num_actions-1)
		return max(model_values,key=model_values.get)

	def get_rweights_action(self):
		game_id = self.current_state.game_id
		score = self.current_state.score
		actions = []
		weights = []
		for action,rweight in self.current_state.action_rweights.items():
			if rweight==0:
				continue
			if rweight is None:
				weight = self._rweight_calc(game_id,score,action)
			else:
				assert rweight==1
				weight = 1
			actions.append(action)
			weights.append(weight)
		if not actions:
			print('Warning: No Actions',self.game_id)
			return random.randint(0,self.current_state.num_actions-1)
		return random.choices(actions,weights=weights,k=1)[0]

	def _rweight_calc(self,game_id,score,action):
		no,yes = self.graph.action_counter.get((game_id,score,action),[0,0])
		if yes>0:
			weight = max(RWEIGHT_MIN,yes/(no+yes))
		else:
			weight = max(RWEIGHT_MIN,(1 if action<5 else RWEIGHT_RANK_DISCOUNT**(action-5))*(RWEIGHT_NO_DISCOUNT**no))
		return weight



#########################################################################################################################################################################

class State:

	def __init__(self,latest_frame):
		assert latest_frame.state in (GameState.NOT_FINISHED,GameState.WIN)
		self.latest_frame = latest_frame
		self.game_id = latest_frame.game_id
		self.score = latest_frame.score
		if latest_frame.state is GameState.NOT_FINISHED:
			self.frame = tuple(tuple(inner) for inner in latest_frame.frame[-1])
		else:
			self.frame = 'WIN'
		self.future_states = {}
		self.prior_states = []
		self.get_object_data()
		self.num_actions = len(self.object_data)+5
		self.action_rweights = {i:None for i in range(self.num_actions)}
		if GameAction.ACTION1 not in latest_frame.available_actions:
			self.action_rweights[0] = 0
		if GameAction.ACTION2 not in latest_frame.available_actions:
			self.action_rweights[1] = 0
		if GameAction.ACTION3 not in latest_frame.available_actions:
			self.action_rweights[2] = 0
		if GameAction.ACTION4 not in latest_frame.available_actions:
			self.action_rweights[3] = 0
		if GameAction.ACTION5 not in latest_frame.available_actions:
			self.action_rweights[4] = 0
		if GameAction.ACTION6 not in latest_frame.available_actions:
			for i in range(len(self.object_data)):
				self.action_rweights[i+5] = 0

	def __eq__(self,other):
		if not isinstance(other,State):
			return NotImplemented
		return (self.game_id,self.score,self.frame)==(other.game_id,other.score,other.frame)

	def __hash__(self):
		return hash((self.game_id,self.score,self.frame))

	def get_object_data(self):
		grid = np.array(self.frame)
		self.object_data = []
		orig_idx = 0
		for colour in range(16):
			raw_labeled,num_features = scipy.ndimage.label((grid==colour))
			slices = scipy.ndimage.find_objects(raw_labeled)
			for i,slc in enumerate(slices):
				if slc is None:
					continue
				mask = (raw_labeled[slc]==(i+1))
				area = np.sum(mask)
				h = slc[0].stop-slc[0].start
				w = slc[1].stop-slc[1].start
				bbox_area = h*w
				size = h*w/(64*64)
				regularity = area/bbox_area
				ys, xs = np.nonzero(mask)
				y_centroid = ys.mean()+slc[0].start
				x_centroid = xs.mean()+slc[1].start
				self.object_data.append({
					"orig_idx": orig_idx,
					"colour": colour,
					"slice": slc,
					"mask": mask,
					"area": area,
					"bbox_area": bbox_area,
					"size": size,
					"regularity":regularity,
					"y_centroid":y_centroid,
					"x_centroid":x_centroid
				})
				orig_idx += 1
		self.object_data.sort(key=lambda obj:(-obj["regularity"],-obj["area"],-obj["colour"],obj["orig_idx"]))

	def get_action_tensor(self,action):
		action_type = torch.zeros(6)
		colour = torch.zeros(16)
		regularity = torch.zeros(1)
		size = torch.zeros(1)
		y_centroid = torch.zeros(1)
		x_centroid = torch.zeros(1)
		if action<=4:
			action_type[action] = 1
			regularity[0] = 1
			size[0] = 1
			y_centroid[0] = -1
			x_centroid[0] = -1
		else:
			action_obj = self.object_data[action-5]
			action_type[5] = 1
			colour[action_obj['colour']] = 1
			regularity[0] = action_obj['regularity']
			size[0] = action_obj['size']
			y_centroid[0] = action_obj['y_centroid']
			x_centroid[0] = action_obj['x_centroid']
		combined = torch.cat([action_type,colour,regularity,size,y_centroid,x_centroid])
		return combined

	def get_action_obj(self,action):
		if action==0:
			return GameAction.ACTION1
		if action==1:
			return GameAction.ACTION2
		if action==2:
			return GameAction.ACTION3
		if action==3:
			return GameAction.ACTION4
		if action==4:
			return GameAction.ACTION5
		else:
			return self.get_click_action_obj(action)

	def get_click_action_obj(self,action):
		obj = self.object_data[action-5]
		slc = obj["slice"]
		mask = obj["mask"]
		local_coords = np.argwhere(mask)
		idx = np.random.choice(len(local_coords))
		local_y, local_x = local_coords[idx]
		global_y = slc[0].start + local_y
		global_x = slc[1].start + local_x
		new_action = GameAction.ACTION6
		new_action.set_data({"x":global_x,"y":global_y})
		return new_action

	def zero_back(self):
		if all(v==0 for v in self.action_rweights.values()):
			for state,action in self.prior_states:
				if state.action_rweights[action]==1:
					state.action_rweights[action] = 0
					state.zero_back()



#########################################################################################################################################################################

class StateGraph:

	def __init__(self):
		self.init_state = None
		self.milestones = {}
		self.states = set()
		self.action_counter = {}
		self.game_id = None

	def get_state(self,latest_frame):
		new_obj = State(latest_frame)
		existing_obj = next((s for s in self.states if s==new_obj),None)
		if existing_obj:
			return existing_obj
		self.states.add(new_obj)
		return new_obj

	def update(self,prev_state,action,new_state):
		game_id = prev_state.game_id
		score = prev_state.score
		if action in prev_state.future_states:
			if prev_state.future_states[action]==new_state:
				return
			else:
				print('Warning: Markov Violation',game_id)
		assert prev_state in self.states
		prev_state.future_states[action] = new_state
		new_state.prior_states.append((prev_state,action))
		if (game_id,score,action) not in self.action_counter:
			self.action_counter[(game_id,score,action)] = [0,0]
		if new_state==prev_state:
			self.action_counter[(game_id,score,action)][0]+=1
			prev_state.action_rweights[action] = 0
			prev_state.zero_back()
			print('Warning: Bad Action',game_id)
		elif new_state==self.milestones[(game_id,score)]:
			self.action_counter[(game_id,score,action)][1]+=1
			prev_state.action_rweights[action] = 0
			prev_state.zero_back()
		elif new_state.score>prev_state.score:
			self.action_counter[(game_id,score,action)][1]+=1
			prev_state.action_rweights[action] = 1
			self.add_milestone(new_state)
			if AGENT_E<1:
				self.train_model(game_id,score+1)
		else:
			self.action_counter[(game_id,score,action)][1]+=1
			prev_state.action_rweights[action] = 1

	def add_milestone(self,state):
		if (state.game_id,state.score) in self.milestones:
			assert self.milestones[(state.game_id,state.score)]==state
		else:
			self.milestones[(state.game_id,state.score)] = state

	def add_init_state(self,state):
		self.init_state = state
		self.add_milestone(state)
		self.game_id = state.game_id

	def get_level_training_data(self,old_milestone,new_milestone):
		final_frame = tuple(tuple(inner) for inner in new_milestone.latest_frame.frame[0])
		state_data = {new_milestone:{'distance':0,'frame':final_frame}}
		max_d = 0
		queue = deque([new_milestone])
		while queue:
			state = queue.popleft()
			current_distance = state_data[state]['distance']
			for prev_state,action in state.prior_states:
				if prev_state.score!=old_milestone.score:
					continue
				if prev_state not in state_data:
					state_data[prev_state] = {'frame':prev_state.frame,'distance':current_distance+1}
					if current_distance+1>max_d:
						max_d = current_distance+1
					queue.append(prev_state)
		queue = deque([old_milestone])
		while queue:
			state = queue.popleft()
			current_distance = state_data[state]['distance']
			for action,future_state in state.future_states.items():
				if future_state.score!=old_milestone.score:
					continue
				if future_state not in state_data:
					state_data[future_state] = {'frame':prev_state.frame,'distance':current_distance+1}
					if current_distance+1>max_d:
						max_d = current_distance+1
					queue.append(future_state)
		final_data = []
		for state in self.states:
			if state.score!=old_milestone.score:
				continue
			for action,future_state in state.future_states.items():
				state_tensor = torch.tensor(state_data[state]['frame'],dtype=torch.long)
				action_tensor = state.get_action_tensor(action)
				if state.action_rweights[action]==0:
					score = torch.tensor(-MODEL_SCORE_MAG,dtype=torch.float32).unsqueeze(0)
				else:
					state_distance = state_data[state]['distance']
					future_state_distance = state_data[future_state]['distance']
					score = torch.tensor(MODEL_SCORE_MAG*(state_distance-future_state_distance)/max_d,dtype=torch.float32).unsqueeze(0)
				data_i = {'state':state_tensor,'action':action_tensor,'score':score}
				final_data.append(data_i)
		return final_data

	def train_model(self,game_id,max_score,verbose=True):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.action_model = ActionModel(game_id).to(device)
		criterion = nn.MSELoss()
		optimizer = optim.Adam(self.action_model.parameters(),lr=MODEL_LR)
		data = []
		for score in range(0,max_score):
			old_milestone = self.milestones[(game_id,score)]
			new_milestone = self.milestones[(game_id,score+1)]
			data_i = self.get_level_training_data(old_milestone,new_milestone)
			data = data+data_i
		dataset = ActionModelDataset(data)
		dataloader = DataLoader(dataset,batch_size=MODEL_BATCH_SIZE,collate_fn=dataset.collate,shuffle=True,num_workers=0,pin_memory=True)
		start_time = time.time()
		for epoch in range(MODEL_NUM_EPOCHS):
			self.action_model.train()
			running_loss = 0.0
			for batch in dataloader:
				state_b = batch['state']
				action_b = batch['action']
				score_b = batch['score']
				optimizer.zero_grad(set_to_none=True)
				preds = self.action_model(state_b,action_b)
				loss = criterion(preds,score_b)
				loss.backward()
				optimizer.step()
				running_loss += loss.item()*score_b.size(0)
			epoch_loss = running_loss/len(dataloader.dataset)
			if verbose:
				print(f"Model Training | {game_id} - level {max_score} | {len(dataloader.dataset)} frames | Epoch {epoch+1}/{MODEL_NUM_EPOCHS} | Loss: {epoch_loss:.4f}")
			if time.time()-start_time>MODEL_MAX_TRAIN_TIME*60:
				print(f'Warning: Reached maximum train time of {MODEL_MAX_TRAIN_TIME} minutes',game_id)



#########################################################################################################################################################################

class ActionModel(nn.Module):

	def __init__(self,game_id):
		super().__init__()
		self.game_id = game_id
		self.grid_symbol_embedding = nn.Embedding(16,16)
		self.stem = nn.Sequential(
			nn.Conv2d(16,64,kernel_size=3,stride=1,padding=1,bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True))
		weights = torchvision_models.ResNet18_Weights.IMAGENET1K_V1
		backbone = torchvision_models.resnet18(weights=weights)
		backbone.conv1 = nn.Identity()
		backbone.bn1 = nn.Identity()
		backbone.relu = nn.Identity()
		backbone.maxpool = nn.Identity()
		self.layer1 = backbone.layer1
		self.layer2 = backbone.layer2
		self.layer3 = backbone.layer3
		self.layer4 = backbone.layer4
		self.avgpool = backbone.avgpool
		self.state_fc = nn.Sequential(
			nn.Linear(backbone.fc.in_features, 64),
			nn.ReLU(inplace=True)
		)
		self.action_fc = nn.Sequential(
			nn.Linear(26, 64),
			nn.ReLU(inplace=True)
		)
		self.head_fc = nn.Sequential(
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 1)
		)

	def forward(self,state,action):
		x = self.grid_symbol_embedding(state)
		x = x.permute(0, 3, 1, 2)
		x = self.stem(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.state_fc(x)
		x_a = self.action_fc(action)
		x = torch.cat([x,x_a],dim=1)
		x = self.head_fc(x)
		return x



#########################################################################################################################################################################

class ActionModelDataset(Dataset):

	def __init__(self,examples):
		self.examples = examples

	def __len__(self):
		return len(self.examples)

	def __getitem__(self,idx):
		return self.examples[idx]

	def collate(self,batch):
		state = torch.stack([b["state"] for b in batch],dim=0)
		action = torch.stack([b["action"] for b in batch],dim=0)
		score = torch.stack([b["score"] for b in batch],dim=0)
		return {"state":state,"action":action,"score":score}


