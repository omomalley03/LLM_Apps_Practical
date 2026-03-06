import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer

class MappingNetwork(nn.Module):
    """
    Simple MLP to project image embeddings into the LLM's embedding space.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super().__init__()
        self.model = nn.Sequential(
            
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class VisionLanguageModel(nn.Module):
    """
    A simple VLM that combines a pre-trained vision encoder with an LLM.
    """
    def __init__(self, vision_encoder_name="openai/clip-vit-large-patch14", 
                 llm_name="Qwen/Qwen2-1.5B-Instruct"):
        super().__init__()
        
        # Initialize the vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name, 
                                                              torch_dtype=torch.bfloat16, 
                                                              )
        
        # Initialize the LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, 
                                                        torch_dtype=torch.bfloat16, 
                                                       )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        
        # Add special token for image
        self.image_token = "<image>"
        special_tokens_dict = {"additional_special_tokens": [self.image_token]}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_tokens > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # Get dimensions for mapping network
        vision_dim = self.vision_encoder.config.hidden_size
        llm_dim = self.llm.get_input_embeddings().weight.shape[1]
        
        # Initialize the mapping network
        self.mapping_network = MappingNetwork(vision_dim, llm_dim).to(torch.bfloat16)
        
        # Save model configuration for later loading
        self.vision_encoder_name = vision_encoder_name
        self.llm_name = llm_name
    
    @classmethod
    def from_pretrained(cls, model_dir):
        """
        Load a pre-trained VisionLanguageModel from a directory.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            A VisionLanguageModel instance
        """
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model with the saved configuration
        model = cls(
            vision_encoder_name=config["vision_encoder_name"],
            llm_name=config["llm_name"]
        )
        
        # Load mapping network weights
        mapping_path = os.path.join(model_dir, "mapping_network.pt")
        model.mapping_network.load_state_dict(torch.load(mapping_path))
        model.mapping_network = model.mapping_network.to(torch.bfloat16)
        
        # Load LLM and tokenizer if they were saved separately
        llm_path = os.path.join(model_dir, "llm")
        if os.path.exists(llm_path):
            print(f"Loading LLM from {llm_path} for modified weights.")
            model.llm = AutoModelForCausalLM.from_pretrained(llm_path, 
                                                            torch_dtype=torch.bfloat16)
            model.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        else:
            print("No LLM directory found, using original model weights")
        
        # Ensure tokenizer has the image token
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            # Re-add special tokens if needed
            if model.image_token not in model.tokenizer.get_vocab():
                special_tokens_dict = {"additional_special_tokens": [model.image_token]}
                model.tokenizer.add_special_tokens(special_tokens_dict)
                model.llm.resize_token_embeddings(len(model.tokenizer))
            model.image_token_id = model.tokenizer.convert_tokens_to_ids(model.image_token)
        
        return model
    
    def save_pretrained(self, output_dir):
        """
        Save the model to a directory.
        
        Args:
            output_dir: Directory to save the model
        """
        import os
        import json
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model configuration
        config = {
            "vision_encoder_name": self.vision_encoder_name,
            "llm_name": self.llm_name
        }
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Save mapping network weights
        mapping_path = os.path.join(output_dir, "mapping_network.pt")
        torch.save(self.mapping_network.state_dict(), mapping_path)
        
        # Save LLM and tokenizer
        llm_path = os.path.join(output_dir, "llm")
        self.llm.save_pretrained(llm_path)
        self.tokenizer.save_pretrained(llm_path)
    
    def freeze_vision_encoder(self):
        """Freeze the vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def freeze_llm(self):
        """Freeze the LLM parameters."""
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def unfreeze_llm(self):
        """Unfreeze the LLM parameters."""
        for param in self.llm.parameters():
            param.requires_grad = True


    def freeze_mapping_network(self):
        """Freeze the Mapping Network parameters."""
        for param in self.mapping_network.parameters():
            param.requires_grad = False

    def unfreeze_mapping_network(self):
        """Unfreeze the Mapping Network parameters."""
        for param in self.mapping_network.parameters():
            param.requires_grad = True

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass through the VLM.
        Image embedding is inserted after the <image> token.
        """
        
        if pixel_values is not None and input_ids is not None:
            image_features = self.vision_encoder(pixel_values).last_hidden_state
            # Use CLS token feature as the global image representation
            image_embeds = image_features[:, 0, :] 
            projected_image_embeds = self.mapping_network(image_embeds) # (batch_size, llm_hidden_size)
            projected_image_embeds = projected_image_embeds.unsqueeze(1) # (batch_size, 1, llm_hidden_size)

            input_embeds = self.llm.get_input_embeddings()(input_ids) # (batch_size, seq_len, llm_hidden_size)
            
            new_input_embeds = []
            new_attention_mask = []
            new_labels = [] if labels is not None else None

            for i in range(input_ids.shape[0]):
                image_token_indices = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
                
                if len(image_token_indices) == 0: # No image token found, proceed as text-only
                    new_input_embeds.append(input_embeds[i])
                    if attention_mask is not None:
                        new_attention_mask.append(attention_mask[i])
                    if labels is not None:
                        new_labels.append(labels[i])
                    continue

                image_token_idx = image_token_indices[0]

                # Concatenate parts: before <image> | projected_image_embed | after <image>
                # The <image> token itself is replaced by the projected image embedding.
                current_input_embeds = torch.cat([
                    input_embeds[i, :image_token_idx],
                    projected_image_embeds[i], # Insert the image embedding
                    input_embeds[i, image_token_idx + 1:]
                ], dim=0)
                new_input_embeds.append(current_input_embeds)

                if attention_mask is not None:
                    # The new sequence length is the same as the original if we replace the <image> token.
                    # If we consider the image embedding as one "token" in terms of attention.
                    current_attention_mask = attention_mask[i] # This might need adjustment if seq length changes
                    new_attention_mask.append(current_attention_mask)

                if labels is not None:
                    # Shift labels for the inserted image embedding.
                    # The <image> token's label should be -100.
                    # All subsequent labels are for the original tokens.
                    current_labels = labels[i]
                    # If the <image> token itself was supposed to be predicted (it shouldn't), mask it.
                    # No change in label length as we are replacing one token's embedding.
                    new_labels.append(current_labels)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            if attention_mask is not None:
                attention_mask = torch.stack(new_attention_mask, dim=0)
            if labels is not None:
                labels = torch.stack(new_labels, dim=0)

            outputs = self.llm(inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask,
                               labels=labels)
        elif input_ids is not None: # Text-only forward pass
            outputs = self.llm(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)
        else:
            raise ValueError("Either pixel_values and input_ids or only input_ids must be provided.")
            
        return outputs

    def generate(self, pixel_values=None, input_ids=None, attention_mask=None, max_new_tokens=None, do_sample=None, temperature=None, top_k=None, top_p=None, **kwargs):
        """
        Generate text based on image and/or text inputs.
        Image embedding is inserted after the <image> token.
        """
        if pixel_values is not None and input_ids is not None:
            image_features = self.vision_encoder(pixel_values).last_hidden_state
            image_embeds = image_features[:, 0, :] # CLS token
            projected_image_embeds = self.mapping_network(image_embeds) # (batch_size, llm_hidden_size)
            projected_image_embeds = projected_image_embeds.unsqueeze(1) # (batch_size, 1, llm_hidden_size)

            input_embeds = self.llm.get_input_embeddings()(input_ids)

            new_input_embeds_list = []
            for i in range(input_ids.shape[0]):
                image_token_indices = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
                if len(image_token_indices) == 0: # No image token
                    new_input_embeds_list.append(input_embeds[i])
                    continue
                
                image_token_idx = image_token_indices[0]
                
                # Replace <image> token's embedding with the projected one
                current_input_embeds = torch.cat([
                    input_embeds[i, :image_token_idx],
                    projected_image_embeds[i],
                    input_embeds[i, image_token_idx + 1:]
                ], dim=0)
                new_input_embeds_list.append(current_input_embeds)
            
            inputs_embeds = torch.stack(new_input_embeds_list, dim=0)
            
            # Adjust attention_mask if sequence length changed. Here, it does not.
            # The original <image> token is replaced by one embedding vector.
            
            # The generate function of LLM expects `inputs_embeds`
            # Max length for generation should account for the input prompt length
            # `max_length` in generate usually means total length (prompt + new tokens)
            # `max_new_tokens` is often preferred.
            
            # Ensure attention_mask has the same sequence length as inputs_embeds
            # If input_ids had shape (batch, seq_len) and inputs_embeds has (batch, seq_len, embed_dim)
            # attention_mask should be (batch, seq_len)
            
            outputs = self.llm.generate(inputs_embeds=inputs_embeds,
                                        attention_mask=attention_mask,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p,
                                        **kwargs)
        elif input_ids is not None: # Text-only generation
            outputs = self.llm.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=do_sample,
                                        temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p,
                                        **kwargs)
        else:
            raise ValueError("Either pixel_values and input_ids or only input_ids must be provided for generation.")
            
        return outputs
