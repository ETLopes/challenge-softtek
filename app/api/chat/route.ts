import { StreamingTextResponse, Message } from "ai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { WebPDFLoader } from "langchain/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BytesOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import type { BaseMessage } from "@langchain/core/messages";
import { RunnableBranch } from "@langchain/core/runnables";

import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";

import fs from "node:fs";

export async function POST(req: Request) {
  const { messages } = await req.json();

  console.log("messages", messages);
  const chatModel = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: "llama3",
  });

  const loader = new WebPDFLoader(
    new Blob([fs.readFileSync("./public/test.pdf")], {
      type: "application/pdf",
    })
  );
  const splitter = new RecursiveCharacterTextSplitter();
  const docs = await loader.load();
  const splitDocs = await splitter.splitDocuments(docs);
  console.log("splitDocs");
  const embeddings = new OllamaEmbeddings({
    model: "mxbai-embed-large",
    maxConcurrency: 5,
    requestOptions: {
      useMMap: false,
      numThread: 6,
      numGpu: 1,
    },
  });
  console.log("embeddings");

  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  console.log("vectorstore");

  const retriever = vectorstore.asRetriever(4);

  console.log("retriever");

  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);
  console.log("historyAwarePrompt");

  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });

  console.log("historyAwareRetrieverChain");

  const chatHistory: Array<HumanMessage | AIMessage> = (
    messages as Message[]
  ).map((m) =>
    m.role == "user" ? new HumanMessage(m.content) : new AIMessage(m.content)
  );

  const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `Answer the user's questions based on the below context:\n\n{context} 
      \n\n You don't need to apologize from anything.
      You just need to answer the user's question.
      And your answer should be relevant to the conversation.
      Start with "The answer is" and then write your answer.`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "Based on the {context} answer this question: {input}"],
  ]);

  const historyAwareCombineDocsChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: historyAwareRetrievalPrompt,
  });
  const parser = new BytesOutputParser();

  const result2 = await historyAwareCombineDocsChain.pipe(parser).stream({
    messages: (messages as Message[]).map((m) =>
      m.role == "user" ? new HumanMessage(m.content) : new AIMessage(m.content)
    ),
    context: splitDocs,
    chat_history: chatHistory,
    input: messages.slice(-1).content,
  });

  // const parseRetrieverInput = (params: { messages: BaseMessage[] }) => {
  //   const transformedResult = { messages: params as unknown as BaseMessage[] };
  //   console.log("transformedResult", transformedResult);
  //   // @ts-ignore - this is a hack to get around the fact that the type of the return value is not the same as the input
  //   return transformedResult.messages.content;
  // };

  // console.log("parseRetrieverInput");

  // const retrievalChain = RunnablePassthrough.assign({
  //   context: RunnableSequence.from([parseRetrieverInput, retriever]),
  // }).assign({
  //   answer: historyAwareCombineDocsChain,
  // });

  // console.log("retrievalChain");

  // const queryTransformPrompt = ChatPromptTemplate.fromMessages([
  //   new MessagesPlaceholder("messages"),
  //   [
  //     "user",
  //     "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
  //   ],
  // ]);
  // const parser = new BytesOutputParser();
  // const queryTransformationChain = queryTransformPrompt
  //   .pipe(chatModel)
  //   .pipe(retrievalChain)
  //   .pipe(parser);

  // console.log("queryTransformationChain");

  // const result = await queryTransformationChain.stream({
  //   messages: (messages as Message[]).map((m) =>
  //     m.role == "user" ? new HumanMessage(m.content) : new AIMessage(m.content)
  //   ),
  // });

  console.log("result", result2);

  return new StreamingTextResponse(result2);
}
