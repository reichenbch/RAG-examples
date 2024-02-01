# RAG-implementation

## Information
Example of Retrieval Augmented Generation with a private dataset.

We have demonstrated three different ways to utilise RAG Implementations over the document for Question/Answering and Parsing.

1 - Original MetaAI RAG Paper Implementation for user dataset.

2 - Llama-Index, LangChain and OpenAI RAG Implementation for user dataset.

3 - Minimilistic Implementation of LaBSE + OpenAI RAG Implementation for user dataset.

**Note:** 
1 - Implementation 2 and Implementation 3 has metrics result for the test dataset on WER (Word Error Rate used for Transcription) and RoUGE Scores (used for Translation) \
2 - These also have implementation of Multilingual Input/Output because of LID from NLLB (MetaAI) with ChatGPT and LID + LaBSE with ChatGPT.

## Setting Up

### Notebooks

All the notebook implementations are mostly self-explanatory.\
It consists of usually 5 sections -
1. Package Installation and Setup
2. Data Cleaning and Preprocessing
3. Main CodeBlock with Indexing and Usage.
4. Inference Example and WrapUp for Productionizing.
5. Evaluation Metrics.

I mostly used the evaluation metrics which made sense to me and usually towards established and statistical metrics. \
But for further work, 
1. ``BertScore`` - https://huggingface.co/spaces/evaluate-metric/bertscore
2. ``ColBERT`` - https://github.com/stanford-futuredata/ColBERT
3. ``RAGAS`` - https://github.com/explodinggradients/ragas

### Gradio Application
Gradio Application is an Inference Code for Implementation 2 - LlamaIndex/LangChain/OpenAI. 

It uses gpt-3.5-turbo model with temperature 0 (zero) for answer generation.

You need put OpenAI Key in `line 22` for Gradio Application and similarly in the notebook instance.

``
os.environ['OPENAI_API_KEY'] = <openai-api-key>
``

Then spin up the gradio application with given configuration, change question examples if using for different dataset.

``
python gradio_app.py
``


## Future Work

Audio Implementation can also be added to these implementations via ``OpenAI/Whisper`` and ``Microsoft/speecht5_tts`` but has not been implemented. 

**Can be added later based on request.**
