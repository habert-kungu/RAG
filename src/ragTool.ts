import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { config } from "./config";
import { Document } from "langchain/document";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";

export async function queryData(query: string) {
  const client = new MongoClient(config.mongodb.uri);
  await client.connect();

  const collection = client
    .db(config.mongodb.dbName)
    .collection(config.mongodb.collectionName);

  const vectorStore = new MongoDBAtlasVectorSearch(
    new GoogleGenerativeAIEmbeddings({ apiKey: config.llm.apiKey }),
    {
      collection: collection,
      indexName: config.vectorStore.indexName,
    }
  );

  const retriever = vectorStore.asRetriever({
    k: 5,
  });

  const llm = new ChatGoogleGenerativeAI({
    apiKey: config.llm.apiKey,
    temperature: 0,
    model: config.llm.model,
  });

  const prompt = PromptTemplate.fromTemplate(
    `Answer the user's question based on the following context:
    Context: {context}
    Question: {question}`
  );

  const formatDocs = (docs: Document[]) => {
    return docs.map((doc) => doc.pageContent).join("\n\n");
  };

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocs),
      question: (input) => input,
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  const result = await chain.invoke(query);

  await client.close();
  return result;
}