# RAG-implementation

Example of Retrieval Augmented Generation with a private dataset.

We have demonstrated three different ways to utilise RAG Implementations over the document for Question/Answering and Parsing.

1 - Original MetaAI RAG Paper Implementation for user dataset.
2 - Llama-Index, LangChain and OpenAI RAG Implementation for user dataset.
3 - Minimilistic Implementation of LaBSE + OpenAI RAG Implementation for user dataset.

Implementation 2 and Implementation 3 has metrics result for the test dataset on WER (Word Error Rate used for Transcription) and RoUGE Scores (used for Translation); these also have implementation of Multilingual Input/Output because of LID from NLLB (MetaAI) with ChatGPT and LID + LaBSE with ChatGPT.

Audio Implementation can also be added to these implementations via OpenAI/Whisper and Microsoft/speecht5_tts but has not been implemented.
