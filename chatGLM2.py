import torch
from transformers import AutoTokenizer
from chatglm.configuration_chatglm import ChatGLMConfig
from chatglm.modeling_chatglm import ChatGLMModel
import pandas as pd
from feature_extractor import ExternalFinancialKnowledgeModel, TextFeatureExtractor
from tqdm import tqdm

def get_model_memory_usage(model):
    """ 모델이 차지하는 GPU 메모리 크기 출력 (GB 단위) """
    model_parameters = sum(p.numel() for p in model.parameters())  # 총 파라미터 개수
    model_memory_bytes = model_parameters * 4  # float32(4 bytes) 기준
    model_memory_gb = model_memory_bytes / (1024 ** 3)  # GB 변환

    print(f"Model Parameters: {model_parameters:,}")  # 1000 단위 콤마 추가
    print(f"Model GPU Memory Usage: {model_memory_gb:.6f} GB")

def print_gpu_memory_usage():
    """ 현재 할당된 모델의 GPU 메모리 사용량 출력 """
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB 단위 변환
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB 단위 변환

    print(f" 현재 할당된 GPU 메모리: {allocated:.4f} GB") 
    print(f" 현재 예약된 GPU 메모리: {reserved:.4f} GB")

class ChatGLMEncoderOnly:
    def __init__(self, model_name_or_path="THUDM/chatglm2-6b", device="cpu"):
        self.device = device

        # Load config and tokenizer
        self.config = ChatGLMConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Load model and extract encoder onlyw
        self.model = ChatGLMModel(self.config).eval().half() #.to(self.device)
        self.encoder = self.model.encoder  # GLMTransformer
        self.embedding = self.model.embedding  # Token embedding
        self.rotary_pos_emb = self.model.rotary_pos_emb

    def encode(self, text):
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(self.device)
        input_ids = inputs["input_ids"]
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # print(tokens)

        batch_size, seq_len = input_ids.shape

        print("Input IDs shape:", input_ids.shape)

        # Generate embeddings
        inputs_embeds = self.embedding(input_ids)  # [seq_len, batch, hidden]
        attention_mask = self.model.get_masks(input_ids, past_key_values=None)

        rotary_pos = self.rotary_pos_emb(self.config.seq_length)[:seq_len].contiguous()

        # Encode
        with torch.no_grad():
            outputs = self.encoder(
                hidden_states=inputs_embeds,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos,
                use_cache=False,
                output_hidden_states=True
            )
        return outputs  # (last_hidden_state, past_key_values, all_hidden_states, attentions)

# Example usage
if __name__ == "__main__":
    
    # df_text = pd.read_csv("output/split_audio/utterance.csv")
    # encoder = ChatGLMEncoderOnly()
    model = TextFeatureExtractor()
    RC = ExternalFinancialKnowledgeModel()

    df_findkg = RC.df_findkg

    pt_list = []
    for i in tqdm(range(len(df_findkg))):
        source = RC.id_to_entity(df_findkg["subject"][i].values[0])
        relation = RC.id_to_relation(df_findkg["relation"][i].values[0])
        destination = RC.id_to_entity(df_findkg["object"][i].values[0])

        text = " ".join([source,relation,destination])

        result = model.encode(text)
        last_hidden = result[0].squeeze()
        pt_list.append(torch.mean(last_hidden,dim=0))

    pt_embedding = torch.stack(pt_list)

    print(pt_embedding.shape)

    # Save the embeddings
    torch.save(pt_embedding, "output/relation_emb.pt")


    # get_model_memory_usage(encoder.model)
    # # print_gpu_memory_usage()

    # pt_list = []

    # for text in tqdm(df_text["Text"]):
    #     result = encoder.encode(text)
    #     last_hidden = result[0].squeeze()
    #     pt_list.append(torch.mean(last_hidden,dim=0))

    # pt_embedding = torch.stack(pt_list)

    # print(pt_embedding.shape)

    # # Save the embeddings
    # torch.save(pt_embedding, "output/split_audio/utterance.pt")

