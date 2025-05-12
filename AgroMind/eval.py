import argparse
import os
import json
from tqdm import tqdm
from config import cfg
from utils.process_func import get_answer
from utils.check_func import check_answer


def remove_none_answer_items(dataset):
    return [item for item in dataset if item.get('ai_answer', True) is not None]

def save_results(output_path, dataset):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

def check_done(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    for item in dataset:
        if not item.get('is_processed', False):
            return False
    return True

def main(args, model, discriminator):
    
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    processed_count = 0
    total_count = len(dataset)
    correct_count = 0
    progress_bar = tqdm(dataset, desc="Processing items")
    for i, item in enumerate(progress_bar):
        if item.get('is_processed', False):
            if item.get('is_true', False):
                correct_count += 1
            continue
        ai_answer = get_answer(args, item, model)
        if ai_answer is None:
            item['ai_answer'] = None
            continue

        is_true = check_answer(args, item, ai_answer, discriminator)
        if is_true:
            correct_count += 1

        item['ai_answer'] = ai_answer
        item['is_true'] = is_true
        item['is_processed'] = True
        processed_count += 1

        progress_bar.set_postfix(accuracy=f"{correct_count}/{i + 1} ({correct_count/(i + 1):.2%})")

        if processed_count % args.save_every == 0:
            save_results(args.output_path, dataset)
    
    dataset = remove_none_answer_items(dataset)
    save_results(args.output_path, dataset)
    print(f"Final results saved to {args.output_path}")
    print(f"Total correct answers: {correct_count}/{total_count}")

    return correct_count, total_count
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "--model",
        type=str,
        default="GPT-4o",
        help="The model to use for evaluation.",
    )
    parser.add_argument(
        "--discriminator_name",
        type=str,
        default="paraphrase-MiniLM-L6-v2",
        help="The name of the discriminator model.",
    )
    parser.add_argument(
        "--discriminator_threshold",
        type=float,
        default=0.60,
        help="The threshold for the discriminator model.",
    )
    parser.add_argument(
        "--test_dir",
        action="store_true",
        help="Whether to test the directory.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./json",
        help="The directory containing the dataset.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./images",
        help="The directory containing the images.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./json/4_6.json",
        help="The path to the dataset for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The path to save the evaluation results.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to use for evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="The GPU ID to use for evaluation.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="The number of steps to refresh the evaluation results.",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from models import get_model, get_discriminator
    os.makedirs(os.path.join(args.output_dir,args.model), exist_ok=True)
    if args.model in cfg.SINGLE.MODEL:
        args.single_image = True
    else:
        args.single_image = False

    model = get_model(args)
    discriminator = get_discriminator(args)

    logging_file = os.path.join(args.output_dir, args.model, "log.txt")
    if not os.path.exists(logging_file):
        with open(logging_file, "w") as f:
            f.write(f"Model: {args.model}\n")
    if args.test_dir:
        for file in sorted(os.listdir(args.dataset_dir)):
            if file.endswith(".json"):
                args.dataset_path = os.path.join(args.dataset_dir, file)
                args.output_path = os.path.join(args.output_dir, args.model, file)
                if os.path.exists(args.output_path):
                    args.dataset_path = args.output_path
                if not check_done(args.dataset_path):
                    correct, total = main(args, model, discriminator)
                    with open(logging_file, "a") as f:
                        f.write(f"{args.dataset_path}: {correct}/{total}, {correct/total:.2%}\n")
    else:
        args.output_path = os.path.join(args.output_dir, args.model, args.dataset_path.split("/")[-1])
        if os.path.exists(args.output_path):
            args.dataset_path = args.output_path
        if not check_done(args.dataset_path):
            correct,total = main(args, model, discriminator)
            with open(logging_file, "a") as f:
                f.write(f"{args.dataset_path}: {correct}/{total}, {correct/total:.2%}\n")
    