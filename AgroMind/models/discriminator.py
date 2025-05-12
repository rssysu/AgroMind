from sentence_transformers import SentenceTransformer, util

class SentenceBERT:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', threshold=0.85):
        self.model = SentenceTransformer(model_name)
        self.model.cuda().eval()
        self.threshold = threshold
    
    def __call__(self, text1, text2):
        embeddings1 = self.model.encode(text1, convert_to_tensor=True).unsqueeze(0).cuda()
        embeddings2 = self.model.encode(text2, convert_to_tensor=True).unsqueeze(0).cuda()
        
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        print(f"Cosine Similarity: {cosine_scores.item()}")
        
        return cosine_scores.item() > self.threshold
    
if __name__ == "__main__":
    model = SentenceBERT()
    text1 = "This is a test sentence."
    text2 = "This is a test sentence."
    
    result = model(text1, text2)
    print(f"Are the sentences similar? {result}")