from textblob import TextBlob
from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.pipeline import ExtractiveQAPipeline
import os
from flask import Flask, request, jsonify, Response, render_template
import glob
app = Flask(__name__)


def Find_answer(text_file_path, data_folder_path, symbol, question):
  document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
  
  with open(text_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
  for i,line in enumerate(data.split(symbol)):
    with open(f'{data_folder_path}/data{i+1}.txt', 'w') as f:
      print(f'writing file no.{i+1}')
      f.write(line)

  test_dicts = convert_files_to_dicts(
      dir_path=data_folder_path, clean_func=clean_wiki_text, split_paragraphs=True)
  document_store.write_documents(test_dicts)
  retriever = DensePassageRetriever(document_store=document_store,
                                    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                    max_seq_len_query=64,
                                    max_seq_len_passage=256,
                                    batch_size=16,
                                    use_gpu=True,
                                    embed_title=True,
                                    use_fast_tokenizers=True)

  document_store.update_embeddings(retriever)

  reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                      use_gpu=True, context_window_size=300)
  
  pipe = ExtractiveQAPipeline(reader, retriever)

  prediction = pipe.run(query=question, top_k_retriever=10,top_k_reader=3)

  doc_with_ans = []
  for i in range(len(prediction['answers'])):
    if prediction['answers'][i]['context'] not in doc_with_ans:
      doc_with_ans.append(prediction['answers'][i]['context'])

  answer = ' '.join(doc_with_ans)

  return answer


@app.route('get_answer',methods = ['POST'])
def get_answer_from_doc():
    data = request.json
    text = data["content"]
    question = data["question"]
    path_content = str(os.getcwd()) + "\\content\\" 
    path_content = path_content.replace("\\","/")
    path_data = str(os.getcwd()) + "\\content\\" 
    path_data = path_data.replace("\\","/")
    f = open(path_content +"/tempfile.txt", "w")
    f.write(text)
    f.close()
    answers = Find_answer(path_content +"/tempfile.txt",path_data,"#",question)
    files = glob.glob(path_data)
    for f in files:
        os.remove(f)
    return jsonify(answers)

if __name__ == "__main__":
    app.run()


