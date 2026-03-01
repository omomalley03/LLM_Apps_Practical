import logging
import operator
import pathlib
import sys

import click
import time

import torch
from omegaconf import OmegaConf, DictConfig
import hydra
import wandb
import os

from tqdm import tqdm
from pathlib import Path
from dstdataset import TrainDataset, Vocabulary
from utils import set_seed, save_checkpoint, load_checkpoint, get_data_version, get_path, setup_wandb, log_level_sort

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import (
	DataLoader,
	RandomSampler,
	SequentialSampler,
)

from torch.utils.tensorboard import SummaryWriter
from transformers import (
	AutoTokenizer,
	AdamW,
	get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def get_dataloader(args, tokenizer, filename, sampler, data_size=-1):
	dataset = TrainDataset(args, tokenizer, filename, data_size)
	dataloader = DataLoader(
		dataset,
		sampler=sampler(dataset),
		batch_size=args.batch_size,
		collate_fn=dataset.collate_fn
	)
	return dataloader


def score_dev(args, dataloader, model):
	loss_total = 0
	num_batches = 0
	model.eval()
	t0 = time.time()
	for batch in tqdm(dataloader, desc="Dev", disable=args.verbose.disable_display):
		num_batches += 1
		with torch.no_grad():
			inputs = batch['input_ids'].to(DEVICE)
			labels = batch['label_ids'].to(DEVICE)
			output = model(
				input_ids=inputs,
				attention_mask=batch['attention_mask'].to(DEVICE),
				labels=labels,
			)
		loss_total += output.loss.item()
	return loss_total / num_batches , time.time() - t0


def train(args, tokenizer, model, initial_step=0):
	log_to_wandb = True if wandb.run else False
	train_dev_args = args
	dev_args, train_args = args.dev, args.train
	log_dir = Path().resolve().joinpath(f'runs/{train_args.experiment_name}')
	writer =  SummaryWriter(
		log_dir=str(log_dir),
	)
	logger.info(f"Tensorboard logs saved at: {log_dir}")
	train_dataloader = get_dataloader(
		train_args,
		tokenizer,
		train_args.dst_train_path,
		sampler=RandomSampler,
		data_size=train_args.data_size
	)
	optimizer = AdamW(model.parameters(), lr=train_args.learning_rate, eps=train_args.adam_eps)
	scheduler = None
	if train_args.use_scheduler:
		t_total = len(train_dataloader) // train_args.gradient_accumulation_steps * train_args.epochs
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=train_args.warmup_steps,
			num_training_steps=t_total
		)
	dev_args.model_type = train_args.model_type
	dev_dataloader = get_dataloader(
		dev_args,
		tokenizer,
		dev_args.dst_dev_path,
		sampler=SequentialSampler,
		data_size=dev_args.data_size
	)
	eval_step = dev_args.eval_interval // train_args.batch_size
	gstep = initial_step // train_args.batch_size
	loss_dev, d = score_dev(dev_args, dev_dataloader, model)
	logger.info(f'Epoch: {gstep} | Batch: 0 | Train Step: 0 | dev loss: {loss_dev:.8f} | time: {d:.3f}')
	if log_to_wandb:
		wandb.log({"Train Step": 0, "Batch": 0, "Train/Dev Loss": loss_dev})
	if gstep > 0:  # we can't actually read the plot if we log that value
		writer.add_scalar('loss/dev', loss_dev, global_step=gstep * train_args.batch_size)
	dev_loss_curve = [(loss_dev, d, 0)]
	logger.info('Start training!')
	for epoch in range(train_args.epochs):
		# initialize for each epoch training
		t0 = time.time()
		disp_loss = 0
                # following had multiplied by 10 after first epoch; but too much evaluation of dev set in first epoch
		if epoch == 1 and initial_step == 0:
			eval_step *= 1
		elif epoch < 1 and initial_step > 0:
			eval_step *= 1
		model.train()
		model.zero_grad()

		iterator = enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=train_args.verbose.disable_display))
		for local_step, batch in iterator:
			output = model(
				input_ids=batch['input_ids'].to(DEVICE),
				attention_mask=batch['attention_mask'].to(DEVICE),
				labels=batch['label_ids'].to(DEVICE)
			)
			loss = output.loss
			disp_loss += output.loss.item()
			gstep += 1
			global_step = gstep * train_args.batch_size
			# update model
			if loss.item() != 0:
				loss = loss / train_args.gradient_accumulation_steps
				loss.backward()
			if gstep % train_args.gradient_accumulation_steps == 0:
				norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)
				optimizer.step()
				if train_args.use_scheduler:
					scheduler.step()
				optimizer.zero_grad()
			if eval_step > 0 and gstep % eval_step == 0:
				loss_dev, d = score_dev(dev_args, dev_dataloader, model)
				dev_loss_curve.append((loss_dev, d, gstep))
				model.train()
				logger.info(f'Epoch: {epoch} | Batch: {gstep} | Train Step: {global_step} | devel loss: {loss_dev:.8f} | time: {d:.3f}')
				if log_to_wandb:
					wandb.log({"Epoch": epoch, "Batch": gstep, "Train Step": global_step, "Train/Dev Loss": loss_dev})
				writer.add_scalar('loss/dev', loss_dev, global_step=gstep * train_args.batch_size)
				save_checkpoint(train_dev_args, tokenizer, model, gstep * train_args.batch_size)
		disp_loss /= (local_step+1)
		logger.info(f'Epoch: {epoch} | Batch: {gstep} | Train Step: {global_step} | train loss: {disp_loss:.8f} | time: {time.time() - t0:.3f}')
		if log_to_wandb:
			wandb.log({"Epoch": epoch, "Batch": gstep, "Train Step": global_step, "Train/Train Loss": disp_loss})
		writer.add_scalar('loss/train', disp_loss, global_step=gstep * train_args.batch_size)
		loss_dev, d = score_dev(dev_args, dev_dataloader, model)
		logger.info(f'Epoch: {epoch} | Batch: {gstep} | Train Step: {global_step} | devel loss: {loss_dev:.8f} | time: {d:.3f}')
		if log_to_wandb:
			wandb.log({"Epoch": epoch, "Batch": gstep, "Train Step": global_step, "Train/Dev Loss": loss_dev})
		writer.add_scalar('loss/dev', loss_dev, global_step=gstep * train_args.batch_size)
		if eval_step < 0: # Only save checkpoint at the end of the epoch if eval_step is negative
			# i.e. you aren't evaluating the dev set while training within an epoch
			save_checkpoint(train_dev_args, tokenizer, model, gstep * train_args.batch_size)
	dev_loss_curve.sort(key=operator.itemgetter(0))
	logger.info(
		f"lowest dev loss: {dev_loss_curve[0][0]} | step: {dev_loss_curve[0][2]} | time: {dev_loss_curve[0][1]}."
	)
	if log_to_wandb:
		wandb.run.summary["Train/Lowest Dev Loss"] = dev_loss_curve[0][0]
		wandb.run.summary["Train/Lowest Dev Loss Step"] = dev_loss_curve[0][2]



def set_model(args, compile: bool = True):
	""" Initiate config, tokenizer and model"""
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) # NOTE: by default, unk_token is set to <|endoftext|>
	special_tokens = OmegaConf.to_object(args.vocab_special_tokens)
	# So that this is only contains normal python objects and no omegaconf objects
	if special_tokens is None:
		raise ValueError("special_tokens must be specified in the config file.")
	vocabulary = Vocabulary(special_tokens=special_tokens)
	# Adding special tokens from the dataset to the tokenizer
	vocabulary.add_special_tokens(args.special_tokens)
	tokenizer.add_special_tokens(vocabulary.special_tokens)
	# Check here
	# Check input tensors to the models!
	# Check with subsample of data
	# If not the input to the model, it will be something else!
	model = hydra.utils.call(args.model_loader, args.model_name_or_path)
	model_config = model.config
	model.resize_token_embeddings(len(tokenizer))
	model.to(DEVICE)
	if compile:
		model = torch.compile(model)
	return model_config, tokenizer, model


@hydra.main(version_base=None, config_path="config", config_name="train_conf")
def main(args: DictConfig):
	# Set up logging
	logger.setLevel(log_level_sort(args.log_level)) 

	try:
		torch.set_float32_matmul_precision(args.torch.f32_matmul_precision)
		logger.info(f"Set torch f32 matmul precision to {args.torch.f32_matmul_precision}.")
	except Exception as e:
		logger.info(f"Could not set torch f32 matmul precision to {args.torch.f32_matmul_precision} due to {e}.")
	
	# Have to do this to convert to a dictionaty and back into a config object
	# to ensure that there are no problems with accesing keys that are not present in subsequent lines
	args_dict = OmegaConf.to_container(args, resolve=True)
	args  = OmegaConf.create(args_dict)
	args_copy = OmegaConf.create(args_dict)
	
	# Set up paths
	train_path = get_path(args.train.train_path, "Train")
	dev_path = get_path(args.dev.dev_path, "Dev")
	
	# Set up checkpoint
	if args.train.checkpoint:
		ckpt_path = get_path(args.train.checkpoint, "Checkpoint")
		logger.info(f"Restarting training from checkpoint: {args.train.checkpoint}")
		initial_step = int(ckpt_path.suffix[1:])
	else:
		ckpt_path = None
		initial_step = 0
	
	model_input_config = OmegaConf.load(f"{train_path.parent.joinpath('preprocessing_config.yaml')}")
	args.data.processing = model_input_config.data.processing
	args.data.version = get_data_version(train_path)
	logger.info(OmegaConf.to_yaml(args))
	logger.info(f"Training on: {'GPU' if 'cuda' in DEVICE.type else 'CPU'}")
	set_seed(args.reproduce)
	args.train.dst_train_path = str(train_path)
	args.dev.dst_dev_path = str(dev_path)
	args.train.special_tokens = model_input_config.data.processing.sequence_format.separators
	
	setup_wandb(args_copy) # Setup wandb with the user provided config
	
	# Initialize model as specified in config
	if ckpt_path:
		config, tokenizer, model = load_checkpoint(args.train, device=DEVICE, compile=args.torch.compile)
	else:
		config, tokenizer, model = set_model(args.train, compile=args.torch.compile)

	# Call Main Training Loop
	train(args, tokenizer, model, initial_step=initial_step)
	logger.info('This is GPT2 for MultiWOZ DST!')
	
	# Finish wandb run
	if wandb.run:
		wandb.finish()

if __name__ == "__main__":
	main()
    

# TODO: ADD default special tokens to config so we don't have to hardcode them in the parser!!!
