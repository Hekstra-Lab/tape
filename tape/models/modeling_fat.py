from fat.layers import FatBlock,FatOutput,Linear,FeedForward
import torch
import torch.nn as nn
from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceToSequenceClassificationHead
from tape.registry import registry

class FatEmbedding(torch.nn.Embedding):
    def reset_parameters(self) -> None:
        torch.nn.init.eye_(self.weight)

    def forward(self, data : torch.Tensor) -> torch.Tensor:
        mask = torch.eye(*self.weight.shape, device=self.weight.device, dtype=torch.bool)
        weight = torch.where(mask, 1., self.weight)
        out = torch.nn.functional.embedding(data, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out

class FatEmbedding(torch.nn.Embedding):
    def forward(self, data : torch.Tensor) -> torch.Tensor:
        logits = torch.concat((
            torch.ones_like(self.weight[...,:1]),
            self.weight[...,1:],
        ), axis=-1)
        weight = torch.nn.functional.softmax(logits, dim=-1)
        out = torch.nn.functional.embedding(data, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out

class FatConfig(ProteinConfig):
    """ 
    The config class for Fourier Attention Transformer model.
    """

    def __init__(self,
                 vocab_size: int = 30,
                 dmodel: int = 128,
                 num_layers: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dmodel = dmodel
        self.num_layers = num_layers


class FatAbstractModel(ProteinModel):
    """ 
    The Fourier Attention Transformer model.
    """
    config_class = FatConfig
    base_model_prefix = 'Fat'


class FatModel(FatAbstractModel):
    """ 
    The Fourier Attention Transformer model.
    """
    # init expects only a single argument - the config
    def __init__(self, config: FatConfig):
        super().__init__(config)
        self.embedding = FatEmbedding(config.vocab_size, config.dmodel)
        layers = []
        for i in range(config.num_layers):
            layers.append(FatBlock(config.dmodel))
        layers.append(FatOutput(config.dmodel))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input_ids, input_mask=None):
        """ Runs the forward model pass

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x protein_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid

        Returns:
            sequence_embedding (Tensor[float]):
                Embedded sequence of shape [batch_size x protein_length x hidden_size]
            pooled_embedding (Tensor[float]):
                Pooled representation of the entire sequence of size [batch_size x hidden_size]
        """

        # Embed the input_ids
        out = self.embedding(input_ids)
        for layer in self.layers:
            out = layer(out)
        pooled = out.mean(axis=-2)

        outputs = (out, pooled)
        return outputs

class FatDecoder(torch.nn.Module):
    def __init__(self, dmodel, vocab_size):
        super().__init__()
        self.linear = nn.Linear(2*dmodel, vocab_size-1)

    def forward(self, embedding):
        logits = self.linear(embedding)
        logits = torch.concat((
            torch.zeros_like(logits[...,:1]),
            logits,
        ), axis=-1)
        return logits

@registry.register_task_model('masked_language_modeling', 'fat')
class FatForMaskedLM(FatAbstractModel):

    def __init__(self, config: FatConfig):
        super().__init__(config)
        self.fat = FatModel(config)
        self.decoder = FatDecoder(config.dmodel, config.vocab_size)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        embedding,pooled = self.fat(input_ids, input_mask=input_mask)
        logits = self.decoder(embedding)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                logits.view(-1, self.config.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            loss_and_metrics = (masked_lm_loss, metrics)
            outputs = (loss_and_metrics,) + (logits, embedding, pooled)
        return outputs


@registry.register_task_model('secondary_structure', 'fat')
class FatForSequenceToSequenceClassification(FatAbstractModel):

    def __init__(self, config: FatConfig):
        super().__init__(config)
        # the name of this variable *must* match the base_model_prefix
        self.fat = FatModel(config)
        # The seq2seq classification head. First argument must match the
        # output embedding size of the SimpleConvModel. The second argument
        # is present in every config (it's an argument of ProteinConfig)
        # and is used for classification tasks.
        self.classify = SequenceToSequenceClassificationHead(
            config.dmodel, config.num_labels)

    def forward(self, input_ids, input_mask=None, targets=None):
        """ Runs the forward model pass and may compute the loss if targets
            is present. Note that this does expect the third argument to be named
            `targets`. You can look at the different defined models to see
            what different tasks expect the label name to be.

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x protein_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid
            targets (Tensor[long], optional):
                Tensor of output target labels of shape [batch_size x protein_length]
        """
        outputs = self.fat(input_ids, input_mask)
        sequence_embedding = outputs[0]

        prediction = self.classify(sequence_embedding)[0]

        outputs = (prediction,)

        if targets is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-1)(
                prediction.view(-1, prediction.size(2)), targets.view(-1))
            # cast to float b/c float16 does not have argmax support
            is_correct = prediction.float().argmax(-1) == targets
            is_valid_position = targets != -1

            # cast to float b/c otherwise torch does integer division
            num_correct = torch.sum(is_correct * is_valid_position).float()
            accuracy = num_correct / torch.sum(is_valid_position).float()
            metrics = {'acc': accuracy}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction


if __name__ == '__main__':
    """ To actually run the model, you can do one of two things. You can
    simply import the appropriate run function from tape.main. The
    possible functions are `run_train`, `run_train_distributed`, `run_eval`,
    and `run_embed`. Alternatively, you can simply place this file inside
    the `tape/models` directory, where it will be auto-imported
    into tape.
    """
    from tape.main import run_train
    run_train()
