import json
import torch
import pandas as pd
import torch.nn as nn

from typing import Optional, Tuple

from .ctnet import CTNetStep
from .language_model import LanguageModel

def json_to_tensor(path: str, device=torch.device):
    with open(path) as f:
        thresholds_map = json.load(f)["thresholds"]
    values = [thresholds_map[c] for c in thresholds_map.keys()]
    thresholds = torch.tensor(values, device=device)
    return thresholds

class ReportGenerationModel(nn.Module):
    def __init__(self, args, mode="train"):
        super().__init__()
        self.args = args

        # thresholds to get selected abnormalities
        self.thresholds = json_to_tensor(args.path_thresholds, args.device)
        
        # Initialize visual encoder, load pretrained weights and freeze parameters
        self.visual_encoder = CTNetStep(args)
        if mode == "train":
            self.visual_encoder.load()
        self.visual_encoder.freeze()

        # Initialize Language Model
        self.language_model = LanguageModel(args)

    def freeze(
        self,
    ):
        for params in self.parameters():
            params.requires_grad = False

    def forward(
        self,
        images_id, labels, images, reports_ids, reports_masks,
        return_loss: bool = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
    ):

        embeddings, predictions, _ = self.visual_encoder(images, labels, return_embeddings=True)
        class_detected = 1*(predictions > self.thresholds)

        valid_input_ids, valid_attention_mask, valid_region_features = self.get_valid_decoder_input_for_training(
            class_detected, labels, reports_ids, reports_masks, embeddings
        )

        # case with no abnormalities detected
        if valid_input_ids.shape[0] == 0:
            return -1

        language_model_loss = self.language_model(
            valid_input_ids,
            valid_attention_mask,
            valid_region_features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache,
        )

        return language_model_loss


    def get_valid_features_handmade_for_training(
            self,
            embeddings,
            valid
    ):
        
        # extract indices of positive labels correctly detected
        valid_indices = torch.where(valid == True)

        # extract embeddings of positive labels correctly detected
        valid_embeddings = embeddings[valid]

        # extract number of valid labels, total number of labels and dimension of latent space
        nb_valid, nb_label, dim = valid_embeddings.size(0), embeddings.size(1), embeddings.size(2)

        # create with empty components where it does not correspond
        zeros_embeddings = torch.zeros(nb_valid, nb_label*dim, device=embeddings.device, dtype=embeddings.dtype)

        # fill this vector with correct components
        for i, (idx_sample, idx_label) in enumerate(zip(valid_indices[0], valid_indices[1])):
            zeros_embeddings[i, idx_label * dim:(idx_label + 1) * dim] = valid_embeddings[i, :]  

        return zeros_embeddings
    
    def get_valid_features_handmade_for_generation(
            self,
            class_detected,
            embeddings
    ):
        
        # extract indices of positive labels correctly detected
        valid_indices = torch.where(class_detected == True)[0]

        # extract embeddings of positive labels correctly detected
        valid_embeddings = embeddings[:, valid_indices]

        # dimension of latent space
        dim = embeddings.size(2)

        # create with empty components where it does not correspond
        zeros_embeddings = torch.zeros(valid_embeddings.size(1), embeddings.size(1)*embeddings.size(2), device=embeddings.device, dtype=embeddings.dtype)

        # fill this vector with correct components
        for i, idx in enumerate(valid_indices):
            zeros_embeddings[i, idx * dim:(idx + 1) * dim] = valid_embeddings[:, i]  

        return zeros_embeddings
    
    def get_valid_decoder_input_for_training(
        self,
        class_detected,
        region_has_sentence, 
        input_ids,  
        attention_mask,  
        region_features,  
    ):
        valid                 = torch.logical_and(class_detected, region_has_sentence)
        valid_input_ids       = input_ids[valid]
        valid_attention_mask  = attention_mask[valid]
        valid_region_features = self.get_valid_features_handmade_for_training(region_features, valid)

        return valid_input_ids, valid_attention_mask, valid_region_features

    def get_valid_decoder_input_for_evaluation(
        self,
        selected_regions,
        input_ids,
        attention_mask
    ):
        selected_regions = selected_regions.reshape(-1)

        valid_input_ids = input_ids[selected_regions]
        valid_attention_mask = attention_mask[selected_regions]

        return valid_input_ids, valid_attention_mask

    def get_valid_decoder_input_for_generation(
        self,
        class_detected: torch.tensor,
        region_features: torch.tensor
    ):
        selected_regions = class_detected.reshape(-1).bool()

        valid_region_features = self.get_valid_features_handmade_for_generation(selected_regions, region_features)

        return valid_region_features

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        volumes: torch.FloatTensor,
        max_length: int = None,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
    ):

        # list that will stock generated sentences
        sentences = []

        # extract embeddings and predicted classes from classifier
        embeddings, predictions = self.visual_encoder.extract_features(volumes)

        # extract detected classes as binary values
        class_detected = 1*(predictions > self.thresholds)

        # case with no abnormalities detected
        if torch.max(class_detected) == 0:
            return "No abnormalities detected."
        
        # get valid data decoder input for generation
        valid_region_features = self.get_valid_decoder_input_for_generation(class_detected=class_detected, region_features=embeddings)

        # iterate over indice of detected label : one sentence to generate per label
        for idx_to_generate in range(valid_region_features.size(0)):

            # extract hidden states corresponding to the current label to generate
            image_hidden_states = valid_region_features[idx_to_generate, :].repeat(num_beams, 1)

            # output_ids of shape (num_regions_selected_in_batch x longest_generated_sequence_length)
            output_ids = self.language_model.generate(
                image_hidden_states,
                max_length,
                num_beams,
                num_beam_groups,
                do_sample,
                num_return_sequences,
                early_stopping,
            )

            # decode and extract the sentence from the report
            generated_sents_for_selected_regions = tokenizer.batch_decode(output_ids, 
                                                                        skip_special_tokens=True, 
                                                                        clean_up_tokenization_spaces=True)

            # extract the sentence as string
            sentence_str = generated_sents_for_selected_regions[0]

            # save result as a single string
            sentences.append(sentence_str)

        generated_report = ' '.join(sentences).replace('"', '')

        return generated_report
