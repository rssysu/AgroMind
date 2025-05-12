##############################################################################
#Config
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals




from utils.attr_dict import AttrDict


__C = AttrDict()
cfg = __C

__C.OPENAI = AttrDict()
__C.SINGLE = AttrDict()

__C.OPENAI.URL = ""
__C.OPENAI.KEY = {"GPT-4o":"",
                  "Gemini-1.5-Flash":"",
                  "Claude-3.5-Sonnet":""}
__C.SINGLE.MODEL = ["TinyLLaVA", "LLaVA-NeXT-7B-Mistral", "LLaVA-NeXT-7B-Vicuna", "LLaVA-NeXT-8B",
                    "LLaVA-NeXT-13B", "LLaVA-NeXT-34B", "XComposler2-4KHD", "InstructBLIP-Vicuna-7B",
                    "Idefics-2-8b"]