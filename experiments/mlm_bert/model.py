# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch as T
from utils import print_rank
import logging
import copy
from typing import (Dict, 
                    List, 
                    Optional, 
                    Tuple, 
                    Union)

from experiments.mlm_bert.utils.trainer_pt_utils import (
    LabelSmoother,
    DistributedTensorGatherer,
    nested_concat,
    nested_detach,
    nested_numpify,
)

from experiments.mlm_bert.utils.trainer_utils import (
    EvalPrediction,
    ComputeMetrics)

from transformers import (
                    MODEL_FOR_MASKED_LM_MAPPING,
                    AutoConfig,
                    AutoModelForMaskedLM,
                    AutoTokenizer,
                    set_seed,
)
from utils.utils import to_device
from core.model import BaseModel

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class BERT(BaseModel):
    def __init__(self, model_config, **kwargs):
        super(BERT, self).__init__()
        """
            from transformers import RobertaConfig
            config = RobertaConfig(
                        vocab_size=52_000,
                        max_position_embeddings=514,
                        num_attention_heads=12,
                        num_hidden_layers=6,
                        type_vocab_size=1,
            )

            from transformers import RobertaTokenizerFast
            tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

            from transformers import RobertaForMaskedLM
            model = RobertaForMaskedLM(config=config)
        """

        # Extracting model_config['BERT']
        args = model_config['BERT']
        # Split data to smaller configuration parameters
        model_args, training_args = args['model'], args['training']

        # Set seed before initializing model.
        set_seed(training_args['seed'])

        self.gradient_accumulation_steps =  model_args.get('gradient_accumulation_steps', 1)
        self.past_index = model_args.get('past_index', -1)
        self.prediction_loss_only = model_args.get('prediction_loss_only', True)
        self.eval_accumulation_steps = model_args.get('eval_accumulation_steps', None)
        self.label_names = model_args.get('label_names', None)
        self.batch_size= training_args['batch_size']
        self.model_name=model_args['model_name']

        if 'model_name_or_path' not in model_args:
            model_args['model_name_or_path']=self.model_name

        # Label smoothing
        if training_args['label_smoothing_factor'] != 0:
            self.label_smoother = LabelSmoother(epsilon=training_args['label_smoothing_factor'])
        else:
            self.label_smoother = None
        self.label_names = ( ["labels"]) if self.label_names is None else self.label_names

        config_kwargs = {
                        "cache_dir": model_args['cache_dir'],
                        "revision": None,
                        "use_auth_token": None,
                    }

        if 'config_name' in model_args:
            config = AutoConfig.from_pretrained(model_args['config_name'], **config_kwargs)
        elif 'model_name_or_path' in model_args:
            config = AutoConfig.from_pretrained(model_args['model_name_or_path'], **config_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new configuration from scratch. This is not supported by this script."
            )


        tokenizer_kwargs = {
                            "cache_dir": model_args['cache_dir'],
                            "use_fast": model_args['use_fast_tokenizer'],
                            "use_auth_token":  None,
                        }
        if 'tokenizer_name' in model_args:
            tokenizer = AutoTokenizer.from_pretrained(model_args['tokenizer_name'], **tokenizer_kwargs)
        elif 'model_name_or_path' in model_args:
            print('Loading Tokenizer from Pretrained: {}'.format(model_args['model_name_or_path']) )
            tokenizer = AutoTokenizer.from_pretrained(model_args['model_name_or_path'], **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            )
        self.output_layer_size=len(tokenizer)

        if 'model_name_or_path' in model_args:
            print('Loading Model from Pretrained: {}'.format(model_args['model_name_or_path']) )
            self.model = AutoModelForMaskedLM.from_pretrained(
                                                    model_args['model_name_or_path'],
                                                    from_tf=False,
                                                    config=config,
                                                    cache_dir=model_args['cache_dir'],
                                                    use_auth_token=None,
                                                )
            if 'adapter' in model_args:
                if model_args['adapter']:
                    self.model.add_adapter("FLUTE")

                    #Activate the adapter
                    self.model.train_adapter("FLUTE")

        else:
            raise ValueError(
                "You are instantiating a new model from scratch. This is not supported by this script."
            )
        self.model.resize_token_embeddings(self.output_layer_size) 
        total_params = 0
        trainable_params = 0

        for p in self.model.parameters():
            total_params += p.numel()
            if p.requires_grad: 
                trainable_params += p.numel()

        print_rank(f"Total parameters count: {total_params}", loglevel=logging.DEBUG) # ~109M
        print_rank(f"Trainable parameters count: {trainable_params}", loglevel=logging.DEBUG) # ~1M
        print_rank(f"Original Bert parameters count: {total_params-trainable_params}", loglevel=logging.DEBUG) # ~1M
        

    def copy_state_dict(self, state_dict):
        self.model.state_dict=state_dict.clone()

    def get_model(self):
        return self.model


    def _prepare_inputs(self, inputs):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, T.Tensor):
                inputs[k] = to_device(v)
        if self.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs


    def forward(self, inputs):
        inputs = self._prepare_inputs(inputs)
        return self.model(**inputs)


    def loss(self, inputs):
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`T.Tensor`: The tensor with training loss on this batch.
        """
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(inputs)
        loss = loss / self.gradient_accumulation_steps

        return loss


    def compute_loss(self, inputs_orig, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.

        inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        """
        # Copy a local copy of the data
        inputs=copy.deepcopy(inputs_orig)

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs["labels"].detach().cpu()
        else:
            labels = None

        # The following fields need to be removed for Roberta
        if 'roberta'  in self.model_name:
            #print("here")
            if 'attention_mask' in inputs:
                inputs.pop('attention_mask')
            if 'special_tokens_mask' in inputs:
                inputs.pop('special_tokens_mask')


        # Forward pass for the transformer
        outputs = self.model(**inputs)

        if self.past_index >= 0:
            self._past = outputs[self.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss




    def inference(
            self, inputs, ignore_keys: Optional[List[str]] = [], metric_key_prefix: str = "eval"
    ) -> List[float]:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.
        Args:
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.
                            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        .. note::
            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.
        Returns: `NamedTuple` A namedtuple with the following keys:
            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """


        output, batch_size = self.prediction_loop(
                                            inputs,
                                            description="Evaluation",
                                            ignore_keys=ignore_keys,
                                            metric_key_prefix=metric_key_prefix)
        return {'output':output['eval_loss'], 'acc': output['eval_acc'], 'batch_size': batch_size[0]}



    def prediction_loop(
                    self,
                    inputs,
                    description: str,
                    ignore_keys: Optional[List[str]] = None,
                    metric_key_prefix: str = "eval",
            ) -> Union[Dict, List[int]]:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """

        out_label_ids=None
        if 'labels' in inputs:
            out_label_ids = inputs['labels'].detach().cpu()

        if 'attention_mask' in inputs:
            attention_mask= inputs['attention_mask'].detach().cpu()

        losses_host = None
        preds_host  = None
        labels_host = None

        world_size = 1
        num_hosts  = 1
        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_hosts, make_multiple_of=self.batch_size)
        if not self.prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_hosts)
            labels_gatherer = DistributedTensorGatherer(world_size, num_hosts)

        self.model.eval()
        if self.past_index >= 0:
            self._past = None

        loss, logits, _ = self.prediction_step(inputs, ignore_keys=ignore_keys, has_labels=True)
        if loss is not None:
            losses = loss.repeat(self.batch_size).cpu()
            losses_host = losses if losses_host is None else T.cat((losses_host, losses), dim=0)
        if logits is not None:
            preds_host = logits.detach().cpu() if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
        if out_label_ids is not None:
            labels_host = out_label_ids if labels_host is None else nested_concat(labels_host, out_label_ids, padding_index=-100)

        # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
        if self.eval_accumulation_steps is not None :
            eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
            if not self.prediction_loss_only:
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

            # Set back to None to begin a new accumulation
            losses_host, preds_host, labels_host = None, None, None

        if self.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if num_hosts>1:
            eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"), want_masked=True)
            if not self.prediction_loss_only:
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

            eval_loss = eval_losses_gatherer.finalize()
            preds = preds_gatherer.finalize() if not self.prediction_loss_only else None
            label_ids = labels_gatherer.finalize() if not self.prediction_loss_only else None
        else:
            eval_loss= losses_host
            preds    = preds_host
            label_ids= labels_host

        if preds is not None and label_ids is not None:
            metrics = ComputeMetrics.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids), attention_mask)
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key).item()
        return metrics, preds.size()


    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        return nested_numpify(tensors)


    def prediction_step(
            self,
            inputs,
            ignore_keys: Optional[List[str]] = None, has_labels: bool = None
    ) -> Tuple[Optional[float], Optional[T.Tensor], Optional[T.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[float], Optional[T.Tensor], Optional[T.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """


        inputs = self._prepare_inputs(inputs)

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            #labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            labels = inputs["labels"].detach().cpu()
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with T.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = self.model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                if self.past_index >= 0:
                    self._past = outputs[self.past_index - 1]

        if self.prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


    def floating_point_ops(self, inputs):
        """
        For models that inherit from :class:`~transformers.PreTrainedModel`, uses that method to compute the number of
        floating point operations for every backward + forward pass. If using another model, either implement such a
        method in the model or subclass and override this method.
        Args:
            inputs (:obj:`Dict[str, Union[T.Tensor, Any]]`):
                The inputs and targets of the model.
        Returns:
            :obj:`int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0



    def set_eval(self):
        """
        Bring the model into evaluation mode
        """
        self.model.eval()


    def set_train(self):
        """
        Bring the model into train mode
        """
        self.model.train()
