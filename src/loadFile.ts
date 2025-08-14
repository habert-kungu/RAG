import { MongoClient } from "mongodb";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { createMetadataTagger } from "langchain/document_transformers/openai_functions";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import "dotenv/config";

async function fileLoader() {
  // setting up mongodb connections
  const client = new MongoClient(process.env.MONGODB_URI!);
  await client.connect();
  const collection = client
    .db("book_mongodb_chunks")
    .collection("chunked_data");

  const loader = new PDFLoader("./src/sample_files/mongodb.pdf");
  const pages = await loader.load();
  const cleanedPages = [];

  for (const page of pages) {
    if (page.pageContent.split(" ").length > 20) {
      cleanedPages.push(page);
    }
  }
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 150,
  });

  const schema = {
    properties: {
      title: { type: "string" },
      keywords: { type: "array", items: { type: "string" } },
      hasCode: { type: "boolean" },
    },
    required: ["title", "keywords", "hasCode"],
  };
  const llm = new ChatGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
    temperature: 0,
    model: "gemini-1.5-flash",
  });

  const document_transformer = createMetadataTagger({
    metaDataSchema: schema,
    llm: llm,
  });
  const docs = await document_transformer.transformDocuments(cleanedPages);
  const splitDocs = await textSplitter.splitDocuments(docs);
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
  });
  const vectorStore = await MongoDBAtlasVectorSearch.fromDocuments(
    splitDocs,
    embeddings,
    {
      collection: collection,
    },
  );
  await client.close();
}

fileLoader();
