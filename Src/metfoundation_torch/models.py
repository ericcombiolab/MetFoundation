import torch 
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Union
from .mask_utils import _generate_mask_matrix, _generate_mask_matrix_VocabFree
from einops import repeat


SIZE_TEXT_EMBS = 1536    
SIZE_MOL_EMBS = 3200


###### Basic Modules


class MetFoundation_EmbeddingLayer(nn.Module):
    def __init__(self, Model_conf:dict, EmbeddingModule_conf:dict, b:int=100):
        
        super(MetFoundation_EmbeddingLayer, self).__init__()
        
        # metabolite symbol embeddings
        ##### integrating prior knowledge: debug only
        if "prior_embs" in EmbeddingModule_conf.keys():
            self.mol_embs = EmbeddingModule_conf["prior_embs"]['mol_embs']
            self.text_embs = EmbeddingModule_conf["prior_embs"]['text_embs']
            
            self.embs_text_layer = nn.Linear(SIZE_TEXT_EMBS, Model_conf["d_model"])       # --these two layers are used to transform the dimension to the model;  
            self.embs_mol_layer = nn.Linear(SIZE_MOL_EMBS, Model_conf["d_model"])         # --discard these layers after model training, save the trained-embeddings of each metabolites to be retrieved in inference
            self.embs_mol_txt_fusion_layer = nn.Linear(int(2*Model_conf["d_model"]), Model_conf["d_model"]) 
            
            self.integrate_metabo_prior = True
            
        else:
            self.embs_ident_layer = nn.Embedding(num_embeddings=EmbeddingModule_conf["n_vocabs"]["identifier"],embedding_dim=Model_conf["d_model"])

            self.integrate_metabo_prior = False
            
    
        # special tokens
        self.cls_emb = nn.Parameter(torch.randn(1, 1, Model_conf["d_model"]))       
        self.pad_emb = nn.Parameter(torch.randn(Model_conf["d_model"]))
        self.mask_emb = nn.Parameter(torch.randn(Model_conf["d_model"])) 


        # continous concentration values to value embeddings
        # ref to '8.Embedding module' of the supplementary file in scFoundation: https://www.nature.com/articles/s41592-024-02305-7
        self.lookup = nn.Linear(b, Model_conf["d_model"], bias=False)   # concentration values: loop-up table for value transformation              
        self.w1 = nn.Linear(1, b, bias=False)                           
        self.w2 = nn.Linear(b, b, bias=False)
        self.alpha = nn.Parameter(torch.randn(b)) 
        self.leakyRelu = nn.LeakyReLU()   
        self.softmax = nn.Softmax(dim=-1)



        
    def forward(self, inputs, add_cls:bool=True):
        
        ############ concentration value embedding ############
        x_conc = torch.unsqueeze(inputs['input_ids']['concentration'], dim=-1) 
        x_conc = torch.where(torch.isnan(x_conc), torch.tensor(0.0, dtype=x_conc.dtype), x_conc) # avoid error caused by NaN
    
        v1 = self.leakyRelu(self.w1(x_conc))     
        v2 = self.w2(v1) + self.alpha*v1
        v3 = self.softmax(v2)
        x_conc = self.lookup(v3)

        mask_indices = inputs['masking_mask'] == 1                                               # replace mask tokens with mask embedding
        x_conc = torch.where(mask_indices.unsqueeze(-1), self.mask_emb, x_conc)


        ############ metabolite symbol embedding ############
        if self.integrate_metabo_prior == True:
            x_mol_emb = self.embs_mol_layer( self.mol_embs[inputs['input_ids']['emb_id']] )
            x_text_emb = self.embs_text_layer( self.text_embs[inputs['input_ids']['emb_id']] )
            x_metabo = self.embs_mol_txt_fusion_layer( torch.concat((x_mol_emb, x_text_emb), dim=-1) )
        else:
            x_metabo = self.embs_ident_layer(inputs['input_ids']['identifier'])


        ############ final embedding: model input ############
        merged_embs = torch.mean( torch.stack((x_metabo, x_conc), dim=-1) , dim=-1)               # merge concentration & metabolite embeddings  


        mask_indices = inputs['padding_mask'] == 1                                                # replace pad tokens with pad embedding
        merged_embs = torch.where(mask_indices.unsqueeze(-1), self.pad_emb, merged_embs)

    
        batch_size = inputs['input_ids']['concentration'].size(0)                                 # create a tensor for CLS embeddings: 
        if add_cls == True:                                                                       # ref to ViT: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
            cls_embs = repeat(self.cls_emb, '1 1 d -> b 1 d', b = batch_size)           
            merged_embs = torch.cat((cls_embs, merged_embs), dim=1)


        # embs_batchID = None
        # if self.with_batchID==True:
        #     embs_batchID = self.embs_batchID_layer(inputs['batchID'])

  
        return merged_embs#, embs_batchID





class MetFoundation_Block(nn.Module):
    def __init__(self, d_ff:int=256, d_model:int=256, n_heads:int=4, dropout:float=0.3,
                 need_weights:bool=False, average_attn_weights:bool=False,
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU()):
        super(MetFoundation_Block, self).__init__()
        '''
        transformer block
        refer to https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
        '''

        # model configuration
        self.d_ff = d_ff
        self.d_model= d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
    
        # layers
        self.MaskedMHA = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, bias=False)
        self.LayerNorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.fc_ff_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.fc_ff_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
  


    def forward(self, x:Tensor, attn_mask: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None) -> Tensor:
        '''
        attn_mask (Optional[Tensor]): 
            If specified, a 2D or 3D mask preventing attention to certain positions.
            A 2D mask will be broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch. 
            Binary and float masks are supported. For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
            For a float mask, the mask values will be added to the attention weight. If both attn_mask and key_padding_mask are supplied, their types should match.
        '''

        x_residue = x
        x, attn = self._sa_block(x, attn_mask, padding_mask)
        x = self.LayerNorm(x_residue + x)
        x = self.LayerNorm(x + self._ff_block(x))
        return x, attn


    # self-attention block
    def _sa_block(self, x:Tensor, attn_mask:Optional[Tensor], key_padding_mask:Optional[Tensor], is_causal:bool=False) -> Tensor:
        x, attn = self.MaskedMHA(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=self.need_weights,
                           average_attn_weights=self.average_attn_weights)
        # print('The output after multi-head self-attention:\n',x.shape) # debug
        return self.dropout(x), attn



    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.fc_ff_2(self.dropout(self.activation(self.fc_ff_1(x))))
        return self.dropout(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")



class PredictionHeadTransform(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    

class ConcentrationPredictionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.transform = PredictionHeadTransform(d_model)
        self.decoder = nn.Linear(d_model, 1)

        # # concentration values should be positive
        # self.softplus = nn.Softplus()

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        logit = torch.squeeze(hidden_states, dim=-1)
        
        # return self.softplus(logit)
        
        return logit



class AgeEmbeddingFusion(nn.Module):
    """
    Age embedding fusion module for survival analysis.
    Projects scalar age value to embedding dimension and fuses with pretrained embeddings.
    
    Args:
        d_model: Dimension of the embeddings
        fusion_mode: How to combine age embedding with input embeddings
                     - 'add': element-wise addition (default)
                     - 'concat': concatenation along feature dimension
    """
    def __init__(self, d_model, fusion_mode='add'):
        super().__init__()
        self.d_model = d_model
        self.fusion_mode = fusion_mode
        
        # Age embedding layer: project scalar to embedding dimension
        self.age_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear( d_model // 2, d_model))
        
        
    def forward(self, embeddings, age):
        """
        Args:
            embeddings: Input embeddings, shape [batch_size, d_model]
            age: Age values, shape [batch_size, 1] or [batch_size]
        
        Returns:
            Fused embeddings with age information
        """
        # Ensure age has shape [batch_size, 1]
        if age.dim() == 1:
            age = age.unsqueeze(1)
        
        # Project age to embedding dimension
        age_emb = self.age_projection(age)
        
        # Fuse embeddings based on mode
        if self.fusion_mode == 'add':
            return embeddings + age_emb
        elif self.fusion_mode == 'concat':
            return torch.cat([embeddings, age_emb], dim=1)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")


class RiskPredictionHead(nn.Module):
    """
    Complete model combining age embedding fusion and prediction head.
    This module takes raw embeddings and age as input, fuses them, and outputs predictions.
    
    Args:
        d_model: Dimension of the embeddings
        num_classes: Number of output classes (1 for survival risk)
        fusion_mode: How to combine age embedding ('add' or 'concat')
        activation: Activation function for output ('softplus', 'sigmoid', or None)
        transform_layer: Whether to use transform layer before prediction
    """
    def __init__(self, d_model, num_classes=1, fusion_mode='add', activation=None, transform_layer=False):
        super().__init__()
        
        # Age embedding fusion module
        self.age_fusion = AgeEmbeddingFusion(d_model=d_model, fusion_mode=fusion_mode)
        
        # Determine input dimension for prediction head
        if fusion_mode == 'concat':
            pred_input_dim = d_model * 2
        else:
            pred_input_dim = d_model
        
        # Prediction head
        self.prediction_head = CustomPredictionHead(
            d_model=pred_input_dim,
            num_classes=num_classes,
            activation=activation,
            transform_layer=transform_layer
        )
    
    def forward(self, embeddings, age):
        """
        Args:
            embeddings: Raw pretrained embeddings, shape [batch_size, d_model]
            age: Age values, shape [batch_size] or [batch_size, 1]
        
        Returns:
            logit: Predictions, shape [batch_size, num_classes]
            fused_embs: Fused embeddings after age fusion, shape [batch_size, d_model or d_model*2]
        """
        # Fuse age embedding with raw embeddings
        fused_embs = self.age_fusion(embeddings, age)
        
        # Make predictions
        logit, _ = self.prediction_head(fused_embs)
        
        return logit, fused_embs


class CustomPredictionHead(nn.Module):
    def __init__(self, d_model, num_classes, activation=None, transform_layer=False):
        super().__init__()
        self.num_classes = num_classes
        self.transform_layer = transform_layer
        
        if transform_layer == True:
            self.transform = PredictionHeadTransform(d_model)
            
        self.decoder = nn.Linear(d_model, num_classes)
        
        self.activation = activation
        if activation == 'softplus':
            self.f_act = nn.Softplus()
        elif activation== 'sigmoid':
            self.f_act = nn.Sigmoid()
        elif activation== 'softmax':
            self.f_act = nn.Softmax(dim=-1)
        
    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        if self.transform_layer == True:
            hidden_states = self.transform(hidden_states)
        
        specific_embs = hidden_states
        
        logit = self.decoder(hidden_states)
        
        
        if isinstance(self.activation, str):  # with activation output
            logit = self.f_act(logit)
            
        return logit, specific_embs


    
    


        
###### Models
class MetFoundation_Model(nn.Module):
    def __init__(self,EmbeddingModule_conf, Model_conf, max_input_tokens:int=512):
        super(MetFoundation_Model, self).__init__()
        '''
        Basic architecture of MetFoundation model
        '''      
        self.d_model = Model_conf["d_model"]
        self.n_blocks = Model_conf["n_blocks"]
        self.attn_mode = Model_conf["attn_mode"]
        self.max_input_tokens = max_input_tokens

        self.emb_layer = MetFoundation_EmbeddingLayer(Model_conf,
                                                 EmbeddingModule_conf,
                                                 b=100                                             # b=100: pre-defined in scFoundation
                                                )

        self.blocks = nn.ModuleList([MetFoundation_Block(d_ff=Model_conf["d_ff"],
                                                        d_model=Model_conf["d_model"],
                                                        n_heads=Model_conf["n_heads"],
                                                        dropout=Model_conf["dropout"],
                                                        activation=Model_conf["activation"],
                                                        need_weights=Model_conf["need_weights"],
                                                        average_attn_weights=Model_conf["average_attn_weights"])
                                            for _ in range(Model_conf["n_blocks"])])
        

    def forward(self, inputs):
        # limited the number of input tokens
        if inputs['padding_mask'].shape[1] > self.max_input_tokens:
            raise ValueError(f'the maximum number of input tokens is {str(self.max_input_tokens)} while got {str(inputs['padding_mask'].shape[1])} tokens')
        
        # mask
        padding_mask = inputs['padding_mask']
        # if self.vocab_free==True:
        padding_mask = torch.cat((torch.zeros(padding_mask.size(0), 1).to(padding_mask.device), padding_mask), dim=1) # add zeros at the first column to represent <CLS>; 0:unmasked
        padding_mask = padding_mask.bool()                                                               # convert 0/1 to boolean type to satisfy the requirement of nn.MultiheadAttention()
        
        if self.attn_mode == 'mixdirect_mask':                                                      # 1:masked, 0:unmasked
            attn_mask = inputs['mixdirect_mask'].bool()                     
        elif self.attn_mode == 'bidirect_mask':
            attn_mask = torch.zeros(inputs['mixdirect_mask'].shape).bool().to(padding_mask.device)  # bi-directional attention
        else:
            raise TypeError(f"attn_mode not support {self.attn_mode}")

        # embedding
        x = self.emb_layer(inputs)

        # transformer block forward
        attn_each_block = []
        for i in range(self.n_blocks):
            x, attn = self.blocks[i](x, attn_mask=attn_mask, padding_mask=padding_mask)             # 'is_causal' set to False to allow MHA to accept 'attn_mask'
            attn_each_block.append(attn)

        return x, (attn_each_block,None)



class MetFoundation_ForPreTrain(nn.Module):        

    def __init__(self, EmbeddingModule_conf, Model_conf):
        super(MetFoundation_ForPreTrain, self).__init__()
        '''
        Metabolomic Foundation Model for pre-training
        '''
        self.n_heads=Model_conf['n_heads']
        self.d_model=Model_conf['d_model']
        self.metfoundation_model = MetFoundation_Model(EmbeddingModule_conf, Model_conf)
        self.conc_predictor =  ConcentrationPredictionHead(Model_conf['d_model']) 
        
        # multi-task learning
        self.multi_task = False
            
            
    def set_multitask_heads(self, config:dict): 
        buffer_dict ={}
        for head_name, head_config in config.items():
            buffer_dict[head_name] = CustomPredictionHead(self.d_model, num_classes=head_config[0], activation=head_config[1], transform_layer=head_config[4])  
        
        self.multi_task_heads =  nn.ModuleDict(buffer_dict)
        self.multi_task = True

        
    def forward(self, inputs):
        ouputs = {}                                                     # output object

        inputs = self.generate_mixdirect_mask(inputs)                   # mixdirect mask
        h, _ = self.metfoundation_model(inputs)                         # metfoundation forward

        h_sample = h[:,0,:]                                             # sample embedding
        ouputs['embs'] = h_sample

        if self.multi_task == True:                                     # multi-task
            for name, prediction_head in self.multi_task_heads.items(): 
               logit_buffer, emb_buffer = prediction_head(h_sample)
               ouputs[f"logit_{name}"] = logit_buffer
               ouputs[f"emb_{name}"] = emb_buffer


        logit = self.conc_predictor(h)                                  # concentration prediction  
        ouputs['logit_conc'] = logit

        return ouputs
    


    def generate_mixdirect_mask(self, inputs) -> Tensor:
        # masked tokens can only 'see' the known tokens and itself in the self-attention matrix
        batch_size = inputs['input_ids']['concentration'].size(0)
        mask_matrix =inputs['masking_mask']
        mask_matrix = torch.cat((torch.zeros(batch_size, 1).to(mask_matrix.device), mask_matrix), dim=1) # add zeros at the first column to represent <CLS>; 0:unmasked
        mask_matrix = _generate_mask_matrix_VocabFree(inputs['input_ids']['concentration'],
                                                        mask_matrix=mask_matrix)

        # expand mask matrix for multi-head
        # ref to: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html -> attn_mask
        N = inputs['padding_mask'].shape[0]
        # seq_len = inputs['padding_mask'].shape[1]
        seq_len = mask_matrix.shape[1]
        mask_matrix = mask_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mask_matrix = mask_matrix.reshape(N * self.n_heads, seq_len, seq_len) 

        inputs['mixdirect_mask'] = mask_matrix
        return inputs


    def save_pretrained(self, save_path):
        if self.multi_task == True:
            params_multi_task = self.multi_task_heads.state_dict()
        else:
            params_multi_task = {}
         
        torch.save({'METFOUNDATION':self.metfoundation_model.state_dict(),
                    'CONCENTRATION_PREDICTOR':self.conc_predictor.state_dict(), 
                    'MULTITASK_HEADS':params_multi_task}, 
                    save_path) 
               

    def from_pretrained(self, model_path):
        # Automatically map to CPU if CUDA is not available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights = torch.load(model_path, map_location=device)
        self.metabolm_model.load_state_dict(model_weights['METABOLM'])
        self.conc_predictor.load_state_dict(model_weights['CONCENTRATION_PREDICTOR'])
        if self.multi_task == True:
            self.multi_task_heads.load_state_dict(model_weights['MULTITASK_HEADS'])







## pre-trained model - for embedding extraction
class MetFoundation_Pretrained(nn.Module):        

    def __init__(self, EmbeddingModule_conf, Model_conf, Model_path):
        super(MetFoundation_Pretrained, self).__init__()
        '''
        Metabolomic Foundation Model
        '''
        self.n_heads=Model_conf['n_heads']
        self.d_model=Model_conf['d_model']

        self.metfoundation_model = MetFoundation_Model(EmbeddingModule_conf, Model_conf)
        self.conc_predictor =  ConcentrationPredictionHead(Model_conf['d_model']) 
        
        # multi-task learning
        if 'multitask_config' in Model_conf.keys():
            self.multitask_config = Model_conf['multitask_config']
            self.set_multitask_heads(self.multitask_config)   
            self.post_calibration = False
        else:
            self.multi_task = False

        # load pre-trained weights
        self.from_pretrained(Model_path)


    def set_multitask_heads(self, config:dict): 
        buffer_dict ={}
        for head_name, head_config in config.items():
            buffer_dict[head_name] = CustomPredictionHead(self.d_model, num_classes=head_config[0], activation=head_config[1], transform_layer=head_config[4]) 
        
        self.multi_task_heads =  nn.ModuleDict(buffer_dict)
        self.multi_task = True


        
    def forward(self, inputs):
        ouputs = {}                                                     # output object

        inputs = self.generate_mixdirect_mask(inputs)                   # mixdirect mask

        h, attn_ = self.metfoundation_model(inputs)                     # metfoundation forward
        attn = attn_[0]                                                 # attention from encoder part
        
        h_sample = h[:,0,:]                                             # sample embedding
        ouputs['embs'] = h_sample

        if self.multi_task == True:                                     # multi-task
            for name, prediction_head in self.multi_task_heads.items(): 
               logit_buffer, emb_buffer = prediction_head(h_sample)
               ouputs[f"logit_{name}"] = logit_buffer
               ouputs[f"emb_{name}"] = emb_buffer

        logit = self.conc_predictor(h)                                  # concentration prediction  
        ouputs['logit_conc'] = logit
        ouputs['attn'] = attn
        
        ouputs['metabolite embs'] = h[:,1:,:] 

        return ouputs
    

    def generate_mixdirect_mask(self, inputs) -> Tensor:
        # masked tokens can only 'see' the known tokens and itself in the self-attention matrix
        batch_size = inputs['input_ids']['concentration'].size(0)
        mask_matrix =inputs['masking_mask']
        mask_matrix = torch.cat((torch.zeros(batch_size, 1).to(mask_matrix.device), mask_matrix), dim=1) # add zeros at the first column to represent <CLS>; 0:unmasked
        mask_matrix = _generate_mask_matrix_VocabFree(inputs['input_ids']['concentration'],
                                                        mask_matrix=mask_matrix)

        # expand mask matrix for multi-head
        # ref to: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html -> attn_mask
        N = inputs['padding_mask'].shape[0]
        # seq_len = inputs['padding_mask'].shape[1]
        seq_len = mask_matrix.shape[1]
        mask_matrix = mask_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mask_matrix = mask_matrix.reshape(N * self.n_heads, seq_len, seq_len) 

        inputs['mixdirect_mask'] = mask_matrix
        return inputs


    def averge_attn(self, attn):
        '''
        averging attention scores across blocks and all tokens
        '''
        return torch.mean( torch.mean(torch.stack(attn,dim=0), dim=0), dim=-2)


    def from_pretrained(self, model_path):
        # Automatically map to CPU if CUDA is not available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights = torch.load(model_path, map_location=device)
        self.metfoundation_model.load_state_dict(model_weights['METABOLM'])        
        self.conc_predictor.load_state_dict(model_weights['CONCENTRATION_PREDICTOR'])
        if self.multi_task == True:
            self.multi_task_heads.load_state_dict(model_weights['MULTITASK_HEADS'])

  
class MetFoundation_Survival(nn.Module):        

    def __init__(self, EmbeddingModule_conf, Model_conf, Model_path):
        super(MetFoundation_Survival, self).__init__()
        '''
        Mortality Risk Model
        '''
        self.n_heads=Model_conf['n_heads']
        self.d_model=Model_conf['d_model']

        self.metfoundation_model = MetFoundation_Model(EmbeddingModule_conf, Model_conf)
        self.risk_head = RiskPredictionHead(d_model=Model_conf['d_model'],
                                            num_classes=1,
                                            fusion_mode='add',
                                            activation=None,
                                            transform_layer=False
                                        )
        # load pre-trained weights
        self.from_pretrained(Model_path)


    def _load_risk_head_weights(self, model_path):
        """
        Load risk_head parameters
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = torch.load(model_path, map_location=device)
        self.risk_head.load_state_dict(weights)
            
            
    def forward(self, x, age):
        outputs = {}
        x = self.generate_mixdirect_mask(x)                   # mixdirect mask

        h, _ = self.metfoundation_model(x)                     # metfoundation forward

        logit, _ = self.risk_head(h[:,0,:], age)
        outputs['logit_risk'] = logit

        return outputs    
    
    
    def generate_mixdirect_mask(self, inputs) -> Tensor:
        # masked tokens can only 'see' the known tokens and itself in the self-attention matrix
        batch_size = inputs['input_ids']['concentration'].size(0)
        mask_matrix =inputs['masking_mask']
        mask_matrix = torch.cat((torch.zeros(batch_size, 1).to(mask_matrix.device), mask_matrix), dim=1) # add zeros at the first column to represent <CLS>; 0:unmasked
        mask_matrix = _generate_mask_matrix_VocabFree(inputs['input_ids']['concentration'],
                                                        mask_matrix=mask_matrix)

        # expand mask matrix for multi-head
        # ref to: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html -> attn_mask
        N = inputs['padding_mask'].shape[0]
        # seq_len = inputs['padding_mask'].shape[1]
        seq_len = mask_matrix.shape[1]
        mask_matrix = mask_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        mask_matrix = mask_matrix.reshape(N * self.n_heads, seq_len, seq_len) 

        inputs['mixdirect_mask'] = mask_matrix
        return inputs   
     
    def from_pretrained(self, model_path):
        # Automatically map to CPU if CUDA is not available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights = torch.load(model_path, map_location=device)
        self.metfoundation_model.load_state_dict(model_weights['METABOLM']) # METABOLM: the name of the key for MetFoundation_Model weights in the saved checkpoint for compatibility in elsewhere

############################################################     
############### Lightweight Models       
############################################################  

## Main Model   
class MetFoundation_Lightweight(nn.Module):        
    def __init__(self, Model_conf):
        super(MetFoundation_Lightweight, self).__init__()
        in_dim = Model_conf['n_features']
        out_dim = Model_conf['d_model']
        p = Model_conf['dropout']
        
        hidden_dim = max(128, in_dim * 2)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        # If input dimension equals hidden_dim, use direct residual; otherwise use linear transformation
        if in_dim != hidden_dim:
            self.res_fc = nn.Linear(in_dim, hidden_dim)
        else:
            self.res_fc = nn.Identity()

    def forward(self, x):

        residual = self.res_fc(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = x + residual  # Residual connection
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * 2),  # Expand first
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),  # Then compress
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
        # Learnable scaling parameter
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        return x + self.alpha * self.block(x)



# With risk prediction head    
class MetFoundation_Lightweight_Survival(nn.Module):  
    def __init__(self, model_conf):
        super(MetFoundation_Lightweight_Survival, self).__init__()

        self.d_model = model_conf['d_model']  # Fix: define d_model first

        # model configuration
        self.lightweight_model = MetFoundation_Lightweight(model_conf)
    
        self.risk_head = RiskPredictionHead(d_model=self.d_model,
                                            num_classes=1,
                                            fusion_mode='add',
                                            activation=None,
                                            transform_layer=False
                                        )
        
        if 'n_subtypes' in model_conf.keys():                      
            self.subtype_head =  CustomPredictionHead(
                                    d_model=self.d_model,
                                    num_classes=model_conf['n_subtypes'],
                                    activation=None,
                                    transform_layer=False
                                )    
        else:
            self.subtype_head = None                    
        
    def _load_risk_head_weights(self, model_path):
        """
        Load risk_head parameters
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = torch.load(model_path, map_location=device)
        self.risk_head.load_state_dict(weights)
    

        
    def forward(self, x, age):
        outputs = {}

        x = self.lightweight_model(x)
        outputs['embs'] = x

        logit, _ = self.risk_head(x, age)
        outputs['logit_risk'] = logit

        if self.subtype_head is not None:
            subtype_logit, _ = self.subtype_head(x)
            outputs['logit_subtype'] = subtype_logit
            
        return outputs


    def save_distilled(self, save_path):
        """Save the entire model"""
        torch.save({'DISTILLED': self.state_dict()}, save_path)

    def from_distilled(self, save_path):
        """Load the entire model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights = torch.load(save_path, map_location=device)
        self.load_state_dict(model_weights['DISTILLED'])







if __name__ == '__main__':
    ###### module
    print('MetFoundation models')

