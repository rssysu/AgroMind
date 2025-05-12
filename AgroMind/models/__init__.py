from models.open_model import OpenAIClient
from models.idefics import IdeficsClient
from models.InstructBLIP import InstructBLIPClient
from models.InternVL import InternVLClient
from models.LLaVA_NeXT_Interleave import LLaVANextInterleaveClient
from models.LLaVA_NeXT import LLaVANeXTClient
from models.Mantis_Idefics2 import MantisIdefics2Client
from models.Mantis import MantisClient
from models.TinyLLaVA import TinyLLaVAClient
from models.XComposer import XComposerClient
from models.random_model import RandomClient
from models.deepseek import DeepseekVL2Client
from models.deepseek_small import DeepseekVL2ClientSmall
from models.discriminator import SentenceBERT
from models.mapping import model_path
from config import cfg


def get_model(args):
    model_name = model_path.get(args.model, None)
    if args.model in ["GPT-4o", "Gemini-1.5-Flash", "Claude-3.5-Sonnet"]:
        model = OpenAIClient(
            base_url=cfg.OPENAI.URL,
            api_key=cfg.OPENAI.KEY.get(args.model),
            prompt=args.prompt,
            model=model_name,
            temperature=args.temperature,
        )
    elif args.model in ["TinyLLaVA"]:
        model = TinyLLaVAClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["InternVL2-2B", "InternVL2-4B", "InternVL2-8B", "InternVL2-26b"]:
        model = InternVLClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["XComposer2-4KHD"]:
        model = XComposerClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["LLaVA-NeXT-7B-Mistral", "LLaVA-NeXT-7B-Vicuna", "LLaVA-NeXT-8B", "LLaVA-NeXT-13B", "LLaVA-NeXT-34B"]:
        model = LLaVANeXTClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["InstructBLIP-Vicuna-7B"]:
        model = InstructBLIPClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["LLaVA-NeXT-Interleave-7B"]:
        model = LLaVANextInterleaveClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["Mantis-LLaMA3-SigLIP"]:
        model = MantisClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["Mantis-Idefics2"]:
        model = MantisIdefics2Client(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["Idefics-2-8b"]:
        model = IdeficsClient(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["deepseek-vl2-tiny", "deepseek-vl2-small", "deepseek-vl2"]:
        model = DeepseekVL2Client(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["deepseek-vl2-small"]:
        model = DeepseekVL2ClientSmall(
            model_name=model_name,
            prompt=args.prompt
        )
    elif args.model in ["Random"]:
        model = RandomClient()
    else:
        raise ValueError(f"Model {args.model} not supported.")
    return model

def get_discriminator(args):
    model = SentenceBERT(args.discriminator_name, args.discriminator_threshold)
    return model