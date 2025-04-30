import pandas as pd
import torch
import numpy as np

# # CSV 파일 로드
# df = pd.read_csv("data_annotation.csv")

# df['label'] = [1 if i %2 == 0 else 0 for i in range(len(df))]


# # 결과 저장
# df.to_csv("data_annotation_label.csv", index=False)

df_annotation = pd.read_csv("/media/user/data/polymarket/data/data_annotation_politics_economy.csv")

df_label = pd.read_csv("volatility_label.csv")

df_merge = pd.merge(df_annotation, df_label, on=['title','url','matching_questions'], how='inner')

print(df_merge.info())

label = df_merge['threshold_label'].values
label = [1 if i == "Impact" else 0 for i in label]

print(np.mean(label))

df_merge[['title','url','matching_questions','audio_file','video_file','question_file','text_file','threshold_label']].to_csv("data_annotation_politics_economy_label.csv", index=False)

# df_cnn = pd.read_csv("cnn_with_labels_01_over60per.csv")

# df_fox = pd.read_csv("fox_with_labels_01_over60per.csv")

# df_concat = pd.concat([df_cnn, df_fox], axis=0)

# print(df_concat.shape)
# print(df_annotation.shape)

# df_merged = pd.merge(df_concat, df_annotation, on=['title','url','matching_questions'], how='inner')

# print(df_merged['change_1d'].mean())

# df_merged[['title','url','matching_questions','emb_file','graph_file','change_1d','change_3d','change_5d']].to_csv("data_annotation_label.csv", index=False)


# df_cnn = pd.read_csv("matching_questions_politics_economy_cnn.csv")
# df_fox = pd.read_csv("matching_questions_politics_economy_fox.csv")
# df_concat = pd.concat([df_cnn, df_fox], ignore_index=True)

# df_concat.to_csv("matching_questions_politics_economy_concat.csv", index=False)