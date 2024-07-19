import torch
import torch.utils.hooks

import style_bert_vits2.models.models as models
import style_bert_vits2.models.models_jp_extra as models_jp_extra

class HookManager:
    def __init__(self, difficulty: list[float] | None):
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.difficulty = difficulty
        self.duration = None
        self.stochastic_duration = None

    def read_duration(
            self,
            module: models.DurationPredictor | models_jp_extra.DurationPredictor, 
            input: torch.Any, 
            output: torch.Tensor
        ):
        self.duration = output

    def read_stochastic_duration(
            self,
            module: models.StochasticDurationPredictor | models_jp_extra.StochasticDurationPredictor,
            input: torch.Any,
            output: torch.Tensor
        ):
        self.stochastic_duration = output

    def write_duration(
            self, 
            module: models.DurationPredictor | models_jp_extra.DurationPredictor, 
            input: torch.Any, 
            output: torch.Tensor
        ):
        if self.duration is not None:
            return self.duration
        
    def write_stochastic_duration(
            self, 
            module: models.StochasticDurationPredictor | models_jp_extra.StochasticDurationPredictor, 
            input: torch.Any, 
            output: torch.Tensor
        ):
        if self.stochastic_duration is not None:
            return self.stochastic_duration

    def modify_embedding(self, module: torch.nn.Embedding, input: torch.Any, output: torch.Tensor):
        if self.difficulty is not None:
            y = output
            difficulty = y.new_tensor(self.difficulty[:-1]).unsqueeze(-1)
            y[:, 1:-2:2] = y[:, 1:-2:2] * (1-difficulty) + y[:, 3::2] * difficulty
            return y
    
    def register_hooks1(self, model: models.SynthesizerTrn | models_jp_extra.SynthesizerTrn | None):
        if model is not None:
            self.handles.append(model.dp.register_forward_hook(self.read_duration))
            self.handles.append(model.sdp.register_forward_hook(self.read_stochastic_duration))

    def register_hooks2(self, model: models.SynthesizerTrn | models_jp_extra.SynthesizerTrn | None):
        if model is not None:
            self.handles.append(model.dp.register_forward_hook(self.write_duration))
            self.handles.append(model.sdp.register_forward_hook(self.write_stochastic_duration))
            self.handles.append(model.enc_p.emb.register_forward_hook(self.modify_embedding))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
