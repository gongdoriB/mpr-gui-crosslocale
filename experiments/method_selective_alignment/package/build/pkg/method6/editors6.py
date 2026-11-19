"""Self-edit text-only editors for the 6-language method run: the evaluated model itself is the editor.

The method branch's TextEditor (editor_matrix_backend.py) loads only Qwen3-VL. This module keeps its prompts, stage limits, greedy settings,
cache-key layout, record fields and the no-visual-input checks, and changes only how the evaluated model is loaded and asked to generate:
  qwen3_vl   (Qwen3-VL-8B, Qwen3-VL-4B)  -> TextEditor unchanged (the method as developed; the 4B dev run also used the 4B as its own editor)
  qwen2_5_vl (UI-TARS-1.5-7B)            -> TextEditor generation path; model class Qwen2_5_VLForConditionalGeneration (SDPA, as the solver config)
  internvl   (InternVL3.5-8B)            -> the model's own conversation template (conv_template, system message = method system prompt),
                                            text-only InternVLChatModel.generate(pixel_values=None) (eager attention, as the solver config)
Every editor call is text only: no image is passed, and a forward hook on the vision module raises if it runs.
"""
import json
import time
from copy import deepcopy
from pathlib import Path

FAMILIES = ('qwen3_vl', 'qwen2_5_vl', 'internvl')


def make_editor(family, config, repo, cache, mock):
    from followup_rel_4lang.method_dev.editor_matrix_backend import TextEditor
    from followup_rel_4lang.method_dev.editor_matrix import LIMITS
    from mpr_crosslocale.methods.selective_alignment.schemas import digest
    if family not in FAMILIES:
        raise ValueError('unknown editor family ' + family)
    if family == 'qwen3_vl' or mock:
        ed = TextEditor(config, repo, cache, 'V', mock)
        ed.family = family
        return ed

    class SelfEditor(TextEditor):
        def _load(self, role):
            if self.loaded == role:
                return 0
            import torch
            from transformers import set_seed
            free, _ = torch.cuda.mem_get_info()
            if free < self.config['min_free_gpu_gib'] * 2 ** 30:
                raise RuntimeError('Insufficient free GPU memory before model load')
            started = time.perf_counter(); set_seed(self.config['seed']); torch.set_num_threads(3)
            spec = self.config['models'][role]
            if family == 'qwen2_5_vl':
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(spec['path'], use_fast=True, min_pixels=262144, max_pixels=2097152, local_files_only=True)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    spec['path'], dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda:0', local_files_only=True).eval()
            else:
                from transformers import AutoModel
                from mpr_crosslocale.models.internvl25 import load_internvl_tokenizer, ensure_generation_mixin
                self.tokenizer = load_internvl_tokenizer(spec['model_id'], revision=spec['revision'], trust_remote_code=True)
                self.model = AutoModel.from_pretrained(spec['path'], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
                                                       use_flash_attn=False, device_map='cuda:0', local_files_only=True).eval()
                ensure_generation_mixin(self.model)
            self.loaded = role; torch.cuda.synchronize()
            return (time.perf_counter() - started) * 1000

        def __call__(self, stage, role, system, user, image=None):
            if family == 'qwen2_5_vl':
                rec = super().__call__(stage, role, system, user, image)
                rec['editor_family'] = family
                return rec
            if role != 'text' or image is not None or stage not in LIMITS:
                raise ValueError('Text-only allowlist violated')
            spec = self.config['models']['editor_vlm']
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
            generation = {'do_sample': False, 'num_beams': 1, 'max_new_tokens': LIMITS[stage], 'use_cache': True}
            key = digest({'messages': messages, 'model': spec, 'generation': generation, 'input_limit': 4096, 'seed': self.config['seed'],
                          'kind': 'V', 'family': family, 'backend_sha': digest(Path(__file__).read_text()),
                          'software': self.software, 'hardware': self.hardware, 'mock': self.mock})
            path = self.cache / (key + '.json')
            hit = path.exists()
            if hit:
                record = json.loads(path.read_text())
            else:
                import torch
                load_ms = self.ensure_loaded()
                start = time.perf_counter()
                template = deepcopy(self.model.conv_template)
                template.system_message = system
                template.append_message(template.roles[0], user)
                template.append_message(template.roles[1], None)
                rendered = template.get_prompt()
                if '<image>' in rendered or '<IMG_CONTEXT>' in rendered:
                    raise RuntimeError('image placeholder in a text-only editor prompt')
                enc = self.tokenizer(rendered, return_tensors='pt')
                tokens = int(enc['input_ids'].shape[1])
                if tokens > 4096:
                    raise RuntimeError(f'Input budget exceeded: {tokens}; no silent truncation')
                sep = template.sep.strip()
                stop = {self.tokenizer.convert_tokens_to_ids(sep)}
                if self.tokenizer.eos_token_id is not None:
                    stop.add(int(self.tokenizer.eos_token_id))
                eot = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')     # as the harness InternVL free-form generation
                if isinstance(eot, int) and eot != self.tokenizer.unk_token_id:
                    stop.add(eot)
                self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')   # InternVL generate() asserts it is set
                visual = [0]
                h = self.model.vision_model.register_forward_pre_hook(lambda m, a: visual.__setitem__(0, visual[0] + 1))
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                try:
                    with torch.inference_mode():
                        out = self.model.generate(pixel_values=None, input_ids=enc['input_ids'].to('cuda:0'),
                                                  attention_mask=enc['attention_mask'].to('cuda:0'), eos_token_id=sorted(stop),
                                                  pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else sorted(stop)[0],
                                                  **{k: v for k, v in generation.items() if k != 'use_cache'})   # InternVL generate() passes use_cache itself
                    torch.cuda.synchronize()
                finally:
                    h.remove()
                if visual[0]:
                    raise RuntimeError('Visual forward unexpectedly executed')
                generation_ms = (time.perf_counter() - t) * 1000
                ids = out[0].detach().cpu().tolist()       # remote-code generate passes inputs_embeds: output = new tokens only
                cut = next((i for i, x in enumerate(ids) if x in stop), None)
                kept = ids[:cut] if cut is not None else ids
                raw = self.tokenizer.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(sep, 1)[0].strip()
                record = {'raw_output': raw, 'rendered_prompt': rendered, 'input_tokens': tokens, 'output_tokens': len(kept) + (cut is not None),
                          'runtime_ms': (time.perf_counter() - start) * 1000, 'generation_ms': generation_ms, 'visual_forward_calls': 0,
                          'vision_tokens': 0, 'model_load_ms': load_ms, 'output_limit_reached': cut is None and len(ids) >= LIMITS[stage],
                          'ended_with_eos': cut is not None, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'execution_mode': 'gpu'}
            record = {**record, 'stage': stage, 'role': role, 'editor_kind': 'V', 'editor_family': family, 'messages': messages,
                      'system_prompt': system, 'user_prompt': user, 'model_id': spec['model_id'], 'revision': spec['revision'],
                      'generation': generation, 'cache_key': key, 'cache_hit': hit, 'prompt_sha256': digest(messages),
                      'software': self.software, 'hardware': self.hardware}
            if not hit:
                self.cache.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(record, ensure_ascii=False) + '\n'); tmp.replace(path)
            self.calls.append(record)
            return record

    ed = SelfEditor(config, repo, cache, 'V', mock)
    ed.family = family
    return ed
